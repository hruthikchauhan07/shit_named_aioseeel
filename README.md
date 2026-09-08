#  Cylinder Counting System

Real-time vehicle, license-plate, and LPG-cylinder monitoring for the  loading area. The system reads six RTSP camera streams, identifies entry and exit vehicles, sends plate crops to an OCR service, counts cylinders from top and side views, and stores the resulting record and evidence images in MongoDB.

## What It Does

The application runs a coordinated multi-camera pipeline:

1. The entry (`cam5`) and exit (`cam3`) ANPR workers classify vehicle orientation and stabilize a license-plate reading.
2. A stabilized plate creates one processing job. Recent-plate, in-progress, and cooldown checks prevent duplicates.
3. The top processor uses `cam2` and `cam6` to detect cylinder positions, rows, columns, load pattern, and truck type.
4. The side processor uses `cam1` and `cam4` to estimate side geometry and provide a fallback count or evidence.
5. The watchdog completes jobs that exceed the configured timeout.
6. The completed job is written to MongoDB with ANPR, top-view, and side-view evidence paths.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Application entry point and worker-thread startup |
| `config.py` | RTSP URLs, model paths, thresholds, timeouts, and output directories |
| `state.py` | Shared models, streams, job state, and job completion |
| `plate_monitor.py` | Entry/exit orientation detection, plate detection, OCR, and job creation |
| `top_processor.py` | Top-camera cylinder detection and load classification |
| `side_processor.py` | Side-camera cylinder geometry and counting |
| `ocr.py` | FastAPI PaddleOCR service on port `9000` |
| `save_data.py` | MongoDB duplicate checks and document persistence |
| `watchdog.py` | Timeout and fallback handling for stuck jobs |
| `models/` | YOLO model files used by the pipeline |
| `anpr_images/`, `plates/` | Captured vehicle and plate images |
| `top_images/`, `top_outputs/` | Raw and annotated top-view evidence |
| `side_images/`, `side_outputs/` | Raw and annotated side-view evidence |
| `logs/` | Rotating `app.log` and `error.log` files |

## Requirements

- Python 3.10 or newer recommended
- A working OpenCV build with FFmpeg and RTSP support
- MongoDB running locally on `mongodb://localhost:27017/`
- Six reachable RTSP camera channels
- YOLO model files in `models/` or valid paths configured in `config.py`
- PaddleOCR with a compatible PaddlePaddle CPU or GPU installation
- For GPU OCR/inference, compatible NVIDIA drivers, CUDA, and cuDNN

There is currently no `requirements.txt`. Install the general Python dependencies in the selected virtual environment, then choose the appropriate PaddlePaddle package for the machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy requests pymongo scikit-learn scipy ultralytics fastapi uvicorn python-multipart
python -m pip install paddleocr
# Install paddlepaddle or paddlepaddle-gpu according to the official compatibility matrix.
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Configuration Before First Run

Edit `config.py` before connecting to production equipment:

1. Set `USERNAME`, `PASSWORD_RAW`, `NVR_IP`, and `PORT` for the NVR.
2. Confirm the six entries in `CAMERA_CONFIG` map to the intended physical cameras.
3. Update `PLATE_MODEL_PATH`, `TOP_MODEL_PATH`, `SIDE_MODEL_PATH`, and `ORIENTATION_MODEL_PATH` to valid local files. The checked-in defaults contain Windows-specific paths and do not point to every model in this repository.
4. Update `LOG_DIR` or make sure the configured directory is writable.
5. Confirm `OCR_URL` matches the host and port where `ocr.py` will run.
6. Review detection confidence, stabilization, cooldown, and timeout values for the camera installation.

Do not commit real camera credentials. The current configuration contains a plaintext password and should be treated as sensitive until moved to environment variables or another secret store.

## Camera Roles

| Camera | Role | Current use |
| --- | --- | --- |
| `cam1` | Side view 1 | Primary side processing camera |
| `cam2` | Top view 1 | Primary top processing camera |
| `cam3` | Exit ANPR | Exit plate and vehicle record; uses the rightmost plate candidate |
| `cam4` | Side view 2 | Side fallback/evidence camera |
| `cam5` | Entry ANPR | Entry plate and vehicle record; uses the leftmost plate candidate |
| `cam6` | Top view 2 | Top fallback/evidence camera |

The default RTSP URL format is:

```text
rtsp://<user>:<password>@<nvr-ip>:<port>/cam/realmonitor?channel=<channel>&subtype=0
```

Use `test-rtsp.py` to validate camera connectivity when the NVR uses a different URL format. That script currently contains a separate Infinova `/media/videoN` test format and should be configured before use.

## Running the Services

Start MongoDB first. In a second terminal, start the OCR API from the project directory:

```bash
source .venv/bin/activate
python ocr.py
```

Verify that the service responds before starting the main workers:

```bash
curl -X POST http://127.0.0.1:9000/ocr -F "file=@path/to/plate.jpg"
```

Then start the main application in another terminal:

```bash
source .venv/bin/activate
python main.py
```

Stop it with `Ctrl+C` or a `SIGTERM`. The application creates its output directories automatically and attempts a clean worker shutdown.

## Data and Evidence

Each completed job is stored in MongoDB database ``, collection `_datas`. Important fields include:

- `vehicleNumber`, `type` (`entry` or `exit`), and generated `code`
- `cylinderCount` and `cylinderCountColor` by cylinder size/color
- `truckType`, `top_count`, `side_count`, and `total_count`
- `truckImages` containing ANPR, primary-view, and multi-view image paths
- `createdAt` and `updatedAt`

Duplicate records are suppressed by vehicle number, record type, and a recent time window. Image paths are relative to the process working directory, so run the application from the project directory unless the paths are intentionally changed.

## Diagnostics and Utility Scripts

```bash
python test-rtsp.py          # Probe configured RTSP endpoints
python test_orientation.py   # Test the orientation model
python test-top.py           # Test top-view inference
python checkdb.py            # Find today's ANPR images without DB records
python check_endpoints.py    # Check configured external endpoints
```

For runtime issues, inspect:

- `logs/app.log` for normal pipeline activity
- `logs/error.log` for errors
- Console output for camera connection and model-loading failures

Useful log events include `JOB CREATED`, `JOB_FINISH`, `JOB TIMEOUT`, `OCR`, `Connected`, and `DUPLICATE DETECTED`.

## Troubleshooting

### The application exits while importing modules

Model loading, RTSP stream creation, and MongoDB initialization happen during module import. Check model paths, Python package installation, MongoDB availability, and camera accessibility first.

### Cameras continually reconnect

Confirm the NVR IP, credentials, port, channel mapping, firewall rules, and FFmpeg-enabled OpenCV build. Run `test-rtsp.py` and compare its URL format with the format in `config.py`.

### Plate jobs are created but no record is saved

Confirm the OCR API is running, MongoDB is reachable, and the current process has write access to the output and log directories. Search `logs/error.log` for the plate number and `Mongo Insert Failed`.

### Counts are unstable or zero

Check camera placement and lighting, then review `PLATE_CONF`, `RAW_STABLE_FRAMES`, `COUNT_STABLE_FRAMES`, `MIN_TOP_RAW_COUNT`, `MIN_SIDE_RAW_COLUMNS`, and `JOB_TIMEOUT_SEC` in `config.py`. Annotated detections in `top_outputs/` and `side_outputs/` are useful for diagnosing model and geometry errors.

## Operational Notes

- The application is designed for one active job at a time and uses shared in-memory state; do not run multiple `main.py` instances against the same camera set.
- `ocr.py` is configured for GPU OCR and one worker. Change that deliberately if deploying on CPU or adding concurrency.
- Trusted plate overrides can be supplied through `trusted_counts.py`; review that file before relying on historical counts.
- Back up MongoDB and the evidence directories according to the site's retention requirements.
