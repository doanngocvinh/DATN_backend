import os
import shutil
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import * 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model.tools.video2anime import *
from datetime import datetime

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

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

    # Upload the processed video to R2
    object_name = upload_to_r2(object_name=info.split('/')[-1], file_path=info)

    try:
        os.remove(file_path)
        os.remove(info)
        print(f"Deleted local files: {file_path} and {info}")
    except Exception as e:
        print(f"Error deleting local files: {e}")

    print(object_name)
    
    # Stream the processed video back to the client
    return {"object_name": object_name}

@app.get("/get_presigned_url/")
def get_presigned_url(file_name: str):
    access_key_id = os.getenv('R2_ACCESS_KEY_ID')
    secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('R2_BUCKET_NAME')
    endpoint_url = os.getenv('R2_ENDPOINT_URL')

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
        return {"url": presigned_url}
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail="Error generating presigned URL")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

