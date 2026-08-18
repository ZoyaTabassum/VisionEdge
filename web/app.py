from pathlib import Path
import shutil
import sys
import subprocess

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.gpu_video_detection import process_video

app = FastAPI(title="VisionEdge")

UPLOAD_DIR = BASE_DIR / "web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_PATH = (
    Path.home()
    / "Downloads"
    / "ffmpeg-9.0.1-essentials_build"
    / "ffmpeg-9.0.1-essentials_build"
    / "bin"
    / "ffmpeg.exe"
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>VisionEdge</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #1f2937;
        }

        .header {
            background: #111827;
            color: white;
            padding: 24px 50px;
        }

        .header h1 {
            margin: 0;
            font-size: 32px;
        }

        .header p {
            margin: 6px 0 0;
            color: #cbd5e1;
        }

        .container {
            max-width: 1100px;
            margin: 35px auto;
            padding: 0 20px;
        }

        .card {
            background: white;
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }

        .upload-box {
            border: 2px dashed #94a3b8;
            border-radius: 12px;
            padding: 35px;
            text-align: center;
        }

        input[type="file"] {
            margin: 20px 0;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 13px 28px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .metric {
            background: #f8fafc;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
        }

        .metric strong {
            display: block;
            font-size: 20px;
            margin-top: 6px;
        }

        .status {
            color: #15803d;
        }

        @media (max-width: 800px) {
            .grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>

<body>

<div class="header">
    <h1>VisionEdge</h1>
    <p>Hardware-Accelerated Video Object Detection</p>
</div>

<div class="container">

    <div class="card">

        <h2>Video Detection</h2>

        <div class="upload-box">

            <p>
                Select a video to process using the Intel Arc GPU.
            </p>

            <form action="/detect"
                  method="post"
                  enctype="multipart/form-data">

                <input
                    type="file"
                    name="video"
                    accept="video/*"
                    required
                >

                <br>

                <button type="submit">
                    Start Detection
                </button>

            </form>

        </div>

    </div>

    <div class="card">

        <h2>VisionEdge Pipeline</h2>

        <div class="grid">

            <div class="metric">
                Model
                <strong>YOLO11n</strong>
            </div>

            <div class="metric">
                Engine
                <strong>OpenVINO</strong>
            </div>

            <div class="metric">
                Device
                <strong>Intel Arc GPU</strong>
            </div>

            <div class="metric">
                Status
                <strong class="status">Ready</strong>
            </div>

        </div>

    </div>

</div>

</body>
</html>
"""


@app.post("/detect", response_class=HTMLResponse)
async def detect(video: UploadFile = File(...)):

    input_path = UPLOAD_DIR / video.filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    output_video, elapsed_time = process_video(
        str(input_path),
        "web_detection"
    )

    mp4_output = output_video.with_name(
        output_video.stem + "_browser.mp4"
    )

    subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-i",
            str(output_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(mp4_output)
        ],
        check=True
    )

    return f"""
<!DOCTYPE html>

<html>

<head>

    <title>VisionEdge Result</title>

    <style>

        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #1f2937;
        }}

        .header {{
            background: #111827;
            color: white;
            padding: 24px 50px;
        }}

        .container {{
            max-width: 1100px;
            margin: 35px auto;
            padding: 0 20px;
        }}

        .card {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}

        video {{
            width: 100%;
            max-height: 650px;
            background: black;
            border-radius: 12px;
        }}

        .stats {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}

        .stat {{
            background: #f1f5f9;
            padding: 15px 25px;
            border-radius: 10px;
        }}

        .button {{
            display: inline-block;
            margin-top: 25px;
            padding: 12px 22px;
            background: #2563eb;
            color: white;
            text-decoration: none;
            border-radius: 8px;
        }}

    </style>

</head>

<body>

<div class="header">

    <h1>VisionEdge</h1>
    <p>Detection Result</p>

</div>

<div class="container">

    <div class="card">

        <h2>Annotated Video</h2>

        <video controls>
            <source src="/video" type="video/mp4">
            Your browser does not support video playback.
        </video>

        <div class="stats">

            <div class="stat">
                <b>Model</b><br>
                YOLO11n
            </div>

            <div class="stat">
                <b>Engine</b><br>
                OpenVINO
            </div>

            <div class="stat">
                <b>Device</b><br>
                Intel Arc GPU
            </div>

            <div class="stat">
                <b>Processing Time</b><br>
                {elapsed_time:.2f} seconds
            </div>

        </div>

        <a class="button" href="/">
            Process Another Video
        </a>

    </div>

</div>

</body>

</html>
"""


@app.get("/video")
def get_video():

    output_dir = BASE_DIR / "output" / "web_detection"

    videos = list(output_dir.glob("*_browser.mp4"))

    if not videos:
        return HTMLResponse(
            "<h2>Processed video not found.</h2>",
            status_code=404
        )

    latest_video = max(
        videos,
        key=lambda path: path.stat().st_mtime
    )

    return FileResponse(
        latest_video,
        media_type="video/mp4",
        filename=latest_video.name
    )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )