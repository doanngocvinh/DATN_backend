import os
import shutil
import subprocess
import zipfile
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Form
from fastapi.responses import * 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model.tools.video2anime import *
from datetime import datetime
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
import re
from PIL import Image, ImageDraw, ImageFont


app = FastAPI(docs_url="/docs")

load_dotenv()

class Item(BaseModel):
    text: str

def upload_to_r2(object_name, file_path):
    # Read credentials and endpoint from environment variables
    access_key_id = os.getenv('R2_ACCESS_KEY_ID')
    secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
    endpoint_url = os.getenv('R2_ENDPOINT_URL')
    bucket_name = os.getenv('R2_BUCKET_NAME')

    if not all([access_key_id, secret_access_key, endpoint_url, bucket_name]):
        raise ValueError("Missing one or more required environment variables: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET_NAME")

    # Create a session using your R2 credentials and endpoint
    session = boto3.session.Session()

    # Create the S3 client with the R2 endpoint
    s3_client = session.client(
        's3',
        region_name='auto',  # R2 does not have a region
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4')
    )

    try:
        # Upload the file
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
    # Construct the file path in the root directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_file_path = f"{timestamp}_{file.filename}"

    # Save the uploaded file
    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define the output file path
    output_file_path = f"{timestamp}_converted.mp4"

    # Convert the video to MP4
    convert_to_mp4(input_file_path, output_file_path)

    # Clean up the original file
    try:
        os.remove(input_file_path)
        print(f"Deleted local file: {input_file_path}")
    except Exception as e:
        print(f"Error deleting local file: {e}")

    return {"output_file_path": output_file_path}
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), type: str = Form(...)):
    # Construct the file path in the root directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"{timestamp}_{file.filename}"

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video using the cartoonizer
    args = parse_args(input_video_path=file_path, model_path=type)

    check_folder(args.output)
    func = Cartoonizer(args)
    info = func()  # Assuming this function processes the video and returns info about the output

    print(f'output video: {info}')

    info_split = info.split('/')[-1]

    convert = f'convert/{info_split}'

    convert_to_mp4(info, convert)

    # Upload the processed video to R2
    object_name = upload_to_r2(object_name=info_split, file_path=convert)

    try:
        os.remove(file_path)
        os.remove(info)
        os.remove(convert)
        print(f"Deleted local files: {file_path} and {info}")
    except Exception as e:
        print(f"Error deleting local files: {e}")
    
    # Stream the processed video back to the client
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
        region_name='auto',  # R2 does not have a region
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version='s3v4')
    )

    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600  # URL expires in 1 hour
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
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    # scene_manager.add_detector(ContentDetector(threshold=threshold)) # (delta_hue, delta_sat, delta_lum, delta_edges) = 1.0 0.5 1.0 0.2
    scene_manager.add_detector(AdaptiveDetector())

    # Start our video manager.
    video_manager.start()

    # Perform scene detection on video_manager.
    scene_manager.detect_scenes(frame_source=video_manager)

    # Obtain list of detected scenes.
    scene_list = scene_manager.get_scene_list()

    # We only need the video_manager for its duration so far, so close it.
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
        # Format the start time as h-m-s-ms
        start_time_formatted = '{:02d}-{:02d}-{:02d}-{:03d}'.format(
            int(start_time // 3600),
            int((start_time % 3600) // 60),
            int(start_time % 60),
            int((start_time * 1000) % 1000)
        )

        # Seek to the start frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret_val, frame = cap.read()
        if ret_val:
            # Include the start time in the filename
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
        text = match.group(3).replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML-like tags
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
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image {image_path}")

    # Get the frame's timestamp from its filename
    filename = os.path.basename(image_path)
    time_match = re.search(r'(\d{2})-(\d{2})-(\d{2})-(\d{3})', filename)
    if not time_match:
        raise ValueError(f"Filename {filename} does not contain a valid timestamp")

    hours, minutes, seconds, milliseconds = map(int, time_match.groups())
    frame_time_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    # Find the corresponding subtitle
    subtitle_text = None
    for start_time, end_time, text_lines in subtitles:
        start_time_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
        end_time_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))
        if start_time_seconds <= frame_time_seconds <= end_time_seconds:
            subtitle_text = '\n'.join(text_lines)
            break

    if subtitle_text:
        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Draw the subtitle on the image
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        text_width, text_height = textsize(subtitle_text, font=font)
        image_width, image_height = pil_image.size
        x = (image_width - text_width) // 2
        y = image_height - text_height - 10

        draw.text((x, y), subtitle_text, font=font, fill="white")

        # Convert the PIL image back to OpenCV format
        frame_with_subtitle = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Save the image with subtitles
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, frame_with_subtitle)

    else:
        # If no subtitle found, just copy the original frame
        shutil.copy(image_path, output_dir)

@app.post("/process_video/")
async def process_video( type: str, video_file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        video_path = f"temp_{video_file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Find scenes and save frames (assuming these functions are defined elsewhere)
        frames_dir = 'output_frames'
        scenes = find_scenes(video_path)
        scene_times = save_frames(video_path, scenes, output_dir=frames_dir)

        # Create a zip file of the output frames
        zip_filename = "output_frames.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(frames_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # Upload the zip file to R2
        r2_object_name = f"processed_videos/{zip_filename}"
        uploaded_object_name = upload_to_r2(r2_object_name, zip_filename)

        # Generate presigned URL for the uploaded file
        presigned_url = generate_presigned_url(uploaded_object_name)

        # Clean up the temporary files and directories
        os.remove(video_path)
        os.remove(zip_filename)
        shutil.rmtree(frames_dir)

        return JSONResponse(content={"message": "Processing complete", "download_url": presigned_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_video_srt/")
async def process_video_srt(type:str, video_file: UploadFile = File(...), srt_file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        video_path = f"temp_{video_file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Save the uploaded SRT file
        srt_path = f"temp_{srt_file.filename}"
        with open(srt_path, "wb") as buffer:
            shutil.copyfileobj(srt_file.file, buffer)

        # Parse the SRT file
        subtitles = parse_srt(srt_path)

        # Find scenes and save frames
        frames_dir = 'output_frames'
        scenes = find_scenes(video_path)
        scene_times = save_frames(video_path, scenes, output_dir=frames_dir)

        # Add subtitles to frames
        output_frames_sub_dir = 'output_frames_with_subtitles'  # Directory for frames with subtitles
        os.makedirs(output_frames_sub_dir, exist_ok=True)
        for image_name in os.listdir(frames_dir):
            image_path = os.path.join(frames_dir, image_name)
            add_subtitle_to_image(image_path, subtitles, output_frames_sub_dir)

        # Create a zip file of the output frames with subtitles
        zip_filename = "output_frames_with_subtitles.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(output_frames_sub_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # Upload the zip file to R2
        r2_object_name = f"processed_videos/{zip_filename}"
        uploaded_object_name = upload_to_r2(r2_object_name, zip_filename)

        # Generate presigned URL for the uploaded file
        presigned_url = generate_presigned_url(uploaded_object_name)

        # Clean up the temporary files and directories
        os.remove(video_path)
        os.remove(srt_path)
        os.remove(zip_filename)
        shutil.rmtree(frames_dir)
        shutil.rmtree(output_frames_sub_dir)

        return JSONResponse(content={"message": "Processing complete", "download_url": presigned_url, 'object_name': r2_object_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_and_upload_video/")
async def process_and_upload_video(type: str = Form(...), video_file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        video_path = f"temp_{timestamp}_{video_file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # Process the video using the cartoonizer
        args = parse_args(input_video_path=video_path, model_path=type)
        check_folder(args.output)
        func = Cartoonizer(args)
        info = func()  # Assuming this function processes the video and returns info about the output

        info_split = info.split('/')[-1]
        converted_video_path = f'convert/{info_split}'
        convert_to_mp4(info, converted_video_path)

        # Find scenes and save frames (assuming these functions are defined elsewhere)
        frames_dir = 'output_frames'
        scenes = find_scenes(converted_video_path)
        scene_times = save_frames(converted_video_path, scenes, output_dir=frames_dir)

        # Create a zip file of the output frames
        zip_filename = "output_frames.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(frames_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # Upload the zip file to R2
        r2_object_name = f"processed_videos/{zip_filename}"
        uploaded_object_name = upload_to_r2(r2_object_name, zip_filename)

        # Generate presigned URL for the uploaded file
        presigned_url = generate_presigned_url(uploaded_object_name)

        # Upload the processed video to R2
        video_object_name = upload_to_r2(f"processed_videos/{info_split}", converted_video_path)

        # Clean up the temporary files and directories
        os.remove(video_path)
        os.remove(zip_filename)
        os.remove(info)
        os.remove(converted_video_path)
        shutil.rmtree(frames_dir)

        return JSONResponse(content={"message": "Processing complete", "download_url": presigned_url, "video_object_name": video_object_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process_and_upload_video_srt/")
async def process_video_srt(type: str = Form(...), video_file: UploadFile = File(...), srt_file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        video_path = f"temp_{timestamp}_{video_file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # Save the uploaded SRT file
        srt_path = f"temp_{timestamp}_{srt_file.filename}"
        with open(srt_path, "wb") as buffer:
            shutil.copyfileobj(srt_file.file, buffer)

        # Process the video using the cartoonizer
        args = parse_args(input_video_path=video_path, model_path=type)
        check_folder(args.output)
        func = Cartoonizer(args)
        info = func()  # Assuming this function processes the video and returns info about the output

        info_split = info.split('/')[-1]
        converted_video_path = f'convert/{info_split}'
        convert_to_mp4(info, converted_video_path)

        # Parse the SRT file
        subtitles = parse_srt(srt_path)

        # Find scenes and save frames from the cartoonized video
        frames_dir = 'output_frames'
        scenes = find_scenes(converted_video_path)
        scene_times = save_frames(converted_video_path, scenes, output_dir=frames_dir)

        # Add subtitles to frames
        output_frames_sub_dir = 'output_frames_with_subtitles'  # Directory for frames with subtitles
        os.makedirs(output_frames_sub_dir, exist_ok=True)
        for image_name in os.listdir(frames_dir):
            image_path = os.path.join(frames_dir, image_name)
            add_subtitle_to_image(image_path, subtitles, output_frames_sub_dir)

        # Create a zip file of the output frames with subtitles
        zip_filename = "output_frames_with_subtitles.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(output_frames_sub_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # Upload the zip file to R2
        r2_object_name = f"processed_videos/{zip_filename}"
        uploaded_object_name = upload_to_r2(r2_object_name, zip_filename)

        # Generate presigned URL for the uploaded file
        presigned_url = generate_presigned_url(uploaded_object_name)

        # Clean up the temporary files and directories
        os.remove(video_path)
        os.remove(srt_path)
        os.remove(zip_filename)
        shutil.rmtree(frames_dir)
        shutil.rmtree(output_frames_sub_dir)

        return JSONResponse(content={"message": "Processing complete", "download_url": presigned_url, "object_name": r2_object_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

