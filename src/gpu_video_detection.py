from ultralytics import YOLO
import time
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n_openvino_model"
INPUT_VIDEO = BASE_DIR / "input" / "input_video.mp4"
OUTPUT_DIR = BASE_DIR / "output"

# Load OpenVINO model
model = YOLO(str(MODEL_PATH), task="detect")

print("========================================")
print("       VISIONEDGE VIDEO PIPELINE")
print("========================================")
print("Inference device: Intel Arc GPU via OpenVINO")
print(f"Input video: {INPUT_VIDEO}")
print()

start_time = time.perf_counter()

model.predict(
    source=str(INPUT_VIDEO),
    device="intel:gpu",
    save=True,
    project=str(OUTPUT_DIR),
    name="gpu_detection",
    exist_ok=True,
    conf=0.25,
    verbose=True
)

elapsed_time = time.perf_counter() - start_time

print()
print("========================================")
print("Video processing completed successfully!")
print(f"Total processing time: {elapsed_time:.2f} seconds")
print("Inference device: Intel Arc GPU via OpenVINO")
print(f"Output directory: {OUTPUT_DIR / 'gpu_detection'}")
print("========================================")