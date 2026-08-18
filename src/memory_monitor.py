from ultralytics import YOLO
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n_openvino_model"
IMAGE_PATH = BASE_DIR / "bus.jpg"

model = YOLO(str(MODEL_PATH), task="detect")

print("========================================")
print(" VisionEdge Continuous Inference Monitor")
print("========================================")
print("Device: Intel Arc GPU via OpenVINO")
print(f"Model: {MODEL_PATH}")
print()

# Warm-up
model.predict(
    source=str(IMAGE_PATH),
    device="intel:gpu",
    verbose=False
)

times = []

for i in range(20):

    start = time.perf_counter()

    model.predict(
        source=str(IMAGE_PATH),
        device="intel:gpu",
        verbose=False
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    times.append(elapsed_ms)

    print(
        f"Run {i + 1:02d}: "
        f"{elapsed_ms:.2f} ms"
    )

average_ms = sum(times) / len(times)
min_ms = min(times)
max_ms = max(times)

print()
print("========================================")
print(f"Runs completed: {len(times)}")
print(f"Average inference: {average_ms:.2f} ms")
print(f"Minimum inference: {min_ms:.2f} ms")
print(f"Maximum inference: {max_ms:.2f} ms")
print("Continuous inference test completed.")
print("========================================")