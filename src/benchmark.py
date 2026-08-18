from ultralytics import YOLO
import time

MODEL_PATH = "models/yolo11n_openvino_model"
IMAGE_PATH = "bus.jpg"

model = YOLO(MODEL_PATH, task="detect")

print("VisionEdge Performance Benchmark")
print("---------------------------------")
print("Device: Intel Arc GPU via OpenVINO")

# Warm-up
model.predict(
    source=IMAGE_PATH,
    device="intel:gpu",
    verbose=False
)

# Benchmark
runs = 5
times = []

for i in range(runs):
    start = time.perf_counter()

    model.predict(
        source=IMAGE_PATH,
        device="intel:gpu",
        verbose=False
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(f"Run {i + 1}: {elapsed * 1000:.2f} ms")

average = sum(times) / len(times)

print("---------------------------------")
print(f"Average inference time: {average * 1000:.2f} ms")
print(f"Approximate FPS: {1 / average:.2f}")
print("Benchmark completed successfully.")