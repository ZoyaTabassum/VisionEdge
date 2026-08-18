from ultralytics import YOLO
import time
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n_openvino_model"
OUTPUT_DIR = BASE_DIR / "output"


# Load OpenVINO model once
model = YOLO(str(MODEL_PATH), task="detect")


def process_video(input_video: str, output_name: str = "web_detection"):
    """
    Process a video using YOLO11n + OpenVINO on Intel Arc GPU.

    Returns:
        tuple: (output_video_path, elapsed_time)
    """

    input_path = Path(input_video)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("       VISIONEDGE VIDEO PIPELINE")
    print("========================================")
    print("Inference device: Intel Arc GPU via OpenVINO")
    print(f"Input video: {input_path}")
    print()

    start_time = time.perf_counter()

    results = model.predict(
        source=str(input_path),
        device="intel:gpu",
        save=True,
        project=str(OUTPUT_DIR),
        name=output_name,
        exist_ok=True,
        conf=0.25,
        verbose=True
    )

    elapsed_time = time.perf_counter() - start_time

    # Ultralytics stores the generated video inside the output directory.
    output_files = list(output_dir.glob("*.avi")) + list(output_dir.glob("*.mp4"))

    if not output_files:
        raise FileNotFoundError(
            f"Detection completed but no output video was found in {output_dir}"
        )

    output_video = output_files[0]

    print()
    print("========================================")
    print("Video processing completed successfully!")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print("Inference device: Intel Arc GPU via OpenVINO")
    print(f"Output video: {output_video}")
    print("========================================")

    return output_video, elapsed_time


if __name__ == "__main__":
    INPUT_VIDEO = BASE_DIR / "input" / "input_video.mp4"
    process_video(str(INPUT_VIDEO), "gpu_detection")