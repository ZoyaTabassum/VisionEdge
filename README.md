\# VisionEdge – Hardware-Accelerated Video Object Detection



\## 1. Project Overview



VisionEdge is a hardware-accelerated video processing prototype for detecting objects in video using a YOLO object detection model.



The project was developed as part of the Axlero Solutions Advanced Python Engineering internship project. The original project specification focuses on a high-performance video pipeline involving model optimization, video processing, hardware acceleration, and real-time inference.



For this implementation, the available development hardware is an \*\*Intel Arc 140T GPU\*\*. Therefore, the hardware-accelerated inference path was implemented using \*\*Intel OpenVINO\*\* instead of NVIDIA TensorRT.



\## 2. Problem Statement



Processing video frame-by-frame for object detection can require significant computational resources. A video-processing system needs to efficiently read frames, perform AI inference, and generate an annotated output.



VisionEdge demonstrates a hardware-accelerated approach where a YOLO model is exported into optimized formats and executed using the Intel GPU.



\## 3. Objectives



The main objectives of this project are:



\* Perform object detection using a pretrained YOLO model.

\* Export the YOLO model to ONNX format.

\* Verify ONNX inference.

\* Export the model to OpenVINO format.

\* Detect available Intel hardware devices.

\* Compile the model for the Intel GPU.

\* Process a video frame-by-frame.

\* Detect objects and draw bounding boxes.

\* Measure video processing time.

\* Produce an annotated output video.



\## 4. Technologies Used



\* \*\*Python 3.13\*\*

\* \*\*Ultralytics YOLO11n\*\*

\* \*\*OpenCV\*\*

\* \*\*ONNX\*\*

\* \*\*OpenVINO\*\*

\* \*\*Intel Arc 140T GPU\*\*

\* \*\*Git\*\*

\* \*\*Visual Studio Code\*\*



\## 5. System Architecture



```text

&#x20;                Input Video

&#x20;                    |

&#x20;                    v

&#x20;                 OpenCV

&#x20;                    |

&#x20;                    v

&#x20;             YOLO11n Model

&#x20;                    |

&#x20;                    v

&#x20;            OpenVINO Runtime

&#x20;                    |

&#x20;                    v

&#x20;             Intel Arc GPU

&#x20;                    |

&#x20;                    v

&#x20;           Object Detection

&#x20;                    |

&#x20;                    v

&#x20;         Bounding Box Rendering

&#x20;                    |

&#x20;                    v

&#x20;            Output Video

```



\## 6. Model Pipeline



The model was developed through the following stages:



```text

YOLO11n PyTorch Model

&#x20;       |

&#x20;       v

&#x20;    ONNX Export

&#x20;       |

&#x20;       v

&#x20;   ONNX Inference

&#x20;       |

&#x20;       v

&#x20;OpenVINO Conversion

&#x20;       |

&#x20;       v

&#x20;Intel GPU Compilation

&#x20;       |

&#x20;       v

&#x20;Video Object Detection

```



\## 7. Hardware Detection



OpenVINO successfully detected the available hardware devices:



```text

CPU

GPU

NPU

```



The GPU inference implementation uses the Intel Arc GPU through OpenVINO.



\## 8. Object Detection Test



The initial YOLO test was performed using a sample image.



The model successfully detected:



```text

4 persons

1 bus

```



The YOLO model used for the project is:



```text

YOLO11n

```



\## 9. ONNX Model



The pretrained YOLO model was exported to:



```text

yolo11n.onnx

```



The ONNX model was successfully loaded and tested using ONNX Runtime.



The test confirmed that the ONNX model could perform object detection successfully.



\## 10. OpenVINO GPU Inference



The YOLO11n model was exported to OpenVINO format and compiled for the Intel Arc GPU.



The available OpenVINO devices on the development system were:



\- CPU

\- GPU

\- NPU



The GPU inference test successfully detected objects in the sample image.



A controlled 5-run benchmark was then performed using the Intel Arc GPU through OpenVINO.



\### Benchmark Result



| Metric | Result |

|---|---:|

| Inference Device | Intel Arc GPU via OpenVINO |

| Number of Runs | 5 |

| Average Inference Time | 24.74 ms |

| Approximate Inference FPS | 40.43 FPS |

| Best Recorded Run | 23.05 ms |

| Slowest Recorded Run | 26.78 ms |



The benchmark confirms that the exported YOLO11n model can be executed successfully on the Intel Arc GPU using OpenVINO.

### Performance Comparison



Recorded benchmark results:



| Pipeline | Average Inference Time | Approx. FPS |

|---|---:|---:|

| PyTorch CPU | 71.82 ms | 13.92 FPS |

| OpenVINO + Intel Arc GPU | 23.98 ms | 41.71 FPS |



The recorded OpenVINO + Intel Arc GPU pipeline achieved approximately 3.0× the FPS of the PyTorch CPU baseline under these benchmark conditions.

### Continuous Inference Stability



A 20-run continuous inference test was performed using the OpenVINO model on the Intel Arc GPU.



\- Runs completed: 20

\- Average inference time: 26.79 ms

\- Minimum inference time: 20.73 ms

\- Maximum inference time: 31.05 ms

\- Result: All 20 inference runs completed successfully without inference failure.



This test demonstrates continuous inference stability. It does not constitute a direct measurement of VRAM memory usage.


## 11. Video Processing



A personal MP4 video was used as the input video for testing.



The video-processing pipeline performs:



```text

Read video

&#x20;   ↓

Process frames

&#x20;   ↓

Run YOLO inference

&#x20;   ↓

Detect objects

&#x20;   ↓

Draw bounding boxes

&#x20;   ↓

Save processed video

```



The processed video successfully displayed detected objects with bounding boxes.



\## 12. GPU Video Processing Result



The video pipeline was executed using OpenVINO with the Intel Arc GPU.



The processed video successfully generated bounding boxes around detected objects.



Two successful test runs recorded the following total processing times:



\- Run 1: 34.41 seconds

\- Run 2: 37.04 seconds



The difference between runs is expected because video processing time can vary depending on system workload and runtime conditions.



The output video was successfully generated with object detections and bounding boxes.


\## 13. Project Structure



```text

VisionEdge/

│

├── models/

│   ├── yolo11n.pt

│   ├── yolo11n.onnx

│   └── yolo11n\_openvino\_model/

│

├── input/

│   └── input\_video.mp4

│

├── output/

│

├── src/

│   └── gpu\_video\_detection.py

│

├── screenshots/

│

├── requirements.txt

├── README.md

│

└── .venv/

```



\## 14. Running the Project



\### Step 1 – Create the virtual environment



```powershell

python -m venv .venv

```



\### Step 2 – Activate the environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### Step 3 – Install dependencies



```powershell

pip install -r requirements.txt

```



\### Step 4 – Run GPU video detection



```powershell

python .\\src\\gpu\_video\_detection.py

```



The program processes the input video using the OpenVINO model and Intel GPU.



\## 15. Results



The project successfully demonstrated:



\* YOLO object detection.

\* YOLO to ONNX export.

\* Successful ONNX inference.

\* OpenVINO model export.

\* Intel GPU detection through OpenVINO.

\* Successful compilation of the YOLO ONNX model for the Intel GPU.

\* Image inference using the Intel GPU.

\* Video object detection using the Intel GPU.

\* Generation of an annotated output video.

\* Measurement of video processing time.



\## 16. Hardware Adaptation



The original VisionEdge specification describes an NVIDIA-based architecture using technologies such as TensorRT, NVIDIA video decoding, CuPy, and WebRTC.



The development laptop used for this implementation contains an Intel Arc 140T GPU. Therefore, the hardware acceleration portion was adapted to use Intel OpenVINO.



This implementation demonstrates the same core concept of hardware-accelerated AI inference while using the hardware available in the development environment.



\## 17. Current Limitations



The current implementation is a prototype and does not yet implement every component described in the complete VisionEdge specification.



The following advanced components remain future development areas:



\* NVIDIA TensorRT implementation.

\* NVIDIA-specific hardware video decoding.

\* CuPy zero-copy GPU pipeline.

\* WebRTC live streaming.

\* Multi-stream orchestration.

\* React telemetry dashboard.

\* Dynamic model swapping.



\## 18. Future Improvements



Future versions can extend the prototype with:



\* Real-time camera or RTSP input.

\* Multiple simultaneous video streams.

\* FPS and GPU utilization monitoring.

\* Live WebRTC streaming.

\* Telemetry dashboard.

\* Asynchronous stream processing.

\* Dynamic model selection.

\* Further GPU optimization.



\## 19. Conclusion



VisionEdge demonstrates a working AI video-processing pipeline using YOLO11n, ONNX, OpenVINO, OpenCV, and an Intel Arc GPU.



The project successfully progresses from object detection on images to GPU-accelerated object detection on video and produces an annotated output video.



The implementation provides a foundation for extending the prototype toward a more advanced real-time edge computer vision system.

### PyAV Video Decoding



PyAV was used to decode the input video and extract raw video frames.



\- Frames successfully decoded: 100

\- Frame resolution: 480 × 864

\- Pixel format: yuv420p



The decoding test completed successfully.



