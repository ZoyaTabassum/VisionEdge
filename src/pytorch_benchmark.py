from ultralytics import YOLO
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
IMAGE_PATH = BASE_DIR / "bus.jpg"

model = YOLO(str(MODEL_PATH), task="detect")

print("========================================")
print("   VisionEdge PyTorch Benchmark")
print("========================================")
print("Device: PyTorch CPU")
print(f"Model: {MODEL_PATH}")
print(f"Image: {IMAGE_PATH}")
print()

# Warm-up
model.predict(
    source=str(IMAGE_PATH),
    device="cpu",
    verbose=False
)

times = []

for i in range(5):

    start = time.perf_counter()

    model.predict(
        source=str(IMAGE_PATH),
        device="cpu",
        verbose=False
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    times.append(elapsed_ms)

    print(f"Run {i + 1}: {elapsed_ms:.2f} ms")

average_ms = sum(times) / len(times)
fps = 1000 / average_ms

print("----------------------------------------")
print(f"Average inference time: {average_ms:.2f} ms")
print(f"Approximate FPS: {fps:.2f}")
print("----------------------------------------")
print("PyTorch benchmark completed successfully.")