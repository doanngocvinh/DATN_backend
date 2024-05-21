import os
import shutil
from fastapi import FastAPI, File, Path, UploadFile, Form
from fastapi.responses import * 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model.tools.video2anime import *

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")


class Item(BaseModel):
    text: str

@app.post("/files/")
async def create_file(file: UploadFile = File(...)):
    return {"file_size": file}


@app.post("/uploadfiles/")
async def create_upload_file(file: UploadFile = File(...), type: str = Form(...)):
    # Construct the file path in the root directory
    file_path = file.filename

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video using the cartoonizer
    arg = parse_args(input_video_path=file_path)

    check_folder(arg.output)
    func = Cartoonizer(arg)
    info = func()

    print(f'output video: {info}')

    return {"filename": file.filename, "type": type}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), type: str = Form(...)):
    # Construct the file path in the root directory
    file_path = file.filename

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video using the cartoonizer
    args = parse_args(input_video_path=file_path)

    check_folder(args.output)
    func = Cartoonizer(args)
    info = func()  # Assuming this function processes the video and returns info about the output

    print(f'output video: {info}')

    # Stream the processed video back to the client
    return StreamingResponse(open(info, 'rb'), media_type='video/mp4', headers={"Content-Disposition": f"attachment; filename={os.path.basename(info)}"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

