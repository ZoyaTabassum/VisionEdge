import av
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "input" / "input_video.mp4"

print("========================================")
print("       VISIONEDGE PyAV TEST")
print("========================================")
print(f"Input video: {VIDEO_PATH}")
print()

container = av.open(str(VIDEO_PATH))

frame_count = 0

for frame in container.decode(video=0):

    frame_count += 1

    if frame_count == 1:
        print(f"First frame detected")
        print(f"Frame size: {frame.width} x {frame.height}")
        print(f"Pixel format: {frame.format.name}")

    if frame_count >= 100:
        break

container.close()

print()
print("========================================")
print(f"Frames successfully decoded: {frame_count}")
print("PyAV decoding test completed successfully.")
print("========================================")