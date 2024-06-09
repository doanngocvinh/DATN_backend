import os
import shutil
import subprocess
import zipfile
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .model.tools.video2anime import *
from datetime import datetime, timedelta
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector
from PIL import Image, ImageDraw, ImageFont
from app import schemas
from app.models import UserModel
import re

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('POSTGRES_URI')
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))

# Initialize FastAPI app
app = FastAPI(docs_url="/docs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,
    pool_timeout=30,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, email: str, password: str):
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if not user or not verify_password(password, user.password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Routes
@app.post("/token", response_model=schemas.Token)
def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(email=user.email, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# File handling and processing functions
def upload_to_r2(object_name, file_path):
    access_key_id = os.getenv('R2_ACCESS_KEY_ID')
    secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
    endpoint_url = os.getenv('R2_ENDPOINT_URL')
    bucket_name = os.getenv('R2_BUCKET_NAME')

    if not all([access_key_id, secret_access_key, endpoint_url, bucket_name]):
        raise ValueError("Missing one or more required environment variables: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET_NAME")

    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        region_name='auto',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4')
    )

    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}' successfully.")
        return object_name
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise

def convert_to_mp4(input_path: str, output_path: str):
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Conversion to MP4 successful: {output_path}")
        print(result.stdout.decode())
        print(result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")
        print(e.stdout.decode())
        print(e.stderr.decode())
        raise HTTPException(status_code=500, detail="Error converting video to MP4")

@app.post("/convert_to_mp4/")
async def convert_video(file: UploadFile = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_file_path = f"{timestamp}_{file.filename}"

    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_file_path = f"{timestamp}_converted.mp4"
    convert_to_mp4(input_file_path, output_file_path)

    try:
        os.remove(input_file_path)
        print(f"Deleted local file: {input_file_path}")
    except Exception as e:
        print(f"Error deleting local file: {e}")

    return {"output_file_path": output_file_path}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), type: str = Form(...)):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"{timestamp}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    args = parse_args(input_video_path=file_path, model_path=type)
    check_folder(args.output)
    func = Cartoonizer(args)
    info = func()
    info_split = info.split('/')[-1]
    convert = f'convert/{info_split}'
    convert_to_mp4(info, convert)

    object_name = upload_to_r2(object_name=info_split, file_path=convert)

    try:
        os.remove(file_path)
        os.remove(info)
        os.remove(convert)
        print(f"Deleted local files: {file_path} and {info}")
    except Exception as e:
        print(f"Error deleting local files: {e}")

    return {"object_name": object_name}

def generate_presigned_url(file_name: str) -> str:
    access_key_id = os.getenv('R2_ACCESS_KEY_ID')
    secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('R2_BUCKET_NAME')
    endpoint_url = os.getenv('R2_ENDPOINT_URL')

    if not all([access_key_id, secret_access_key, endpoint_url, bucket_name]):
        raise ValueError("Missing one or more required environment variables: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET_NAME")

    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        region_name='auto',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4')
    )

    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600
        )
        return presigned_url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail="Error generating presigned URL")

@app.get("/get_presigned_url/")
def get_presigned_url(file_name: str = Query(..., description="Name of the file for which to generate a presigned URL")):
    try:
        presigned_url = generate_presigned_url(file_name)
        return {"url": presigned_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_scenes(video_path, threshold=27.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return scene_list

def save_frames(video_path, scenes, output_dir='output_frames'):
    scene_times = []

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    for i, scene in enumerate(scenes):
        start_frame = scene[0].frame_num
        start_time = scene[0].get_seconds()
        scene_times.append(start_time)
        start_time_formatted = '{:02d}-{:02d}-{:02d}-{:03d}'.format(
            int(start_time // 3600),
            int((start_time % 3600) // 60),
            int(start_time % 60),
            int((start_time * 1000) % 1000)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret_val, frame = cap.read()
        if ret_val:
            output_file_path = os.path.join(output_dir, f"scene_{i+1}_start-{start_time_formatted}.jpg")
            cv2.imwrite(output_file_path, frame)

    cap.release()
    return scene_times

def parse_srt(srt_file_path):
    with open(srt_file_path, 'r') as file:
        srt_content = file.read()

    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}\n(.+?)(?:\n\n|\Z)', re.DOTALL)
    subtitles = []

    for match in pattern.finditer(srt_content):
        start_time = match.group(1)
        end_time = match.group(2)
        text = match.group(3).replace('\n', ' ')
        text = re.sub(r'<[^>]+>', '', text)
        text_lines = text.split(' - ')
        text_lines = [line.strip() for line in text_lines if line.strip()]
        subtitles.append((start_time, end_time, text_lines))

    return subtitles

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def add_subtitle_to_image(image_path, subtitles, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image {image_path}")

    filename = os.path.basename(image_path)
    time_match = re.search(r'(\d{2})-(\d{2})-(\d{2})-(\d{3})', filename)
    if not time_match:
        raise ValueError(f"Filename {filename} does not contain a valid timestamp")

    hours, minutes
