import cv2
import time
import state
from config import ORIENTATION_MODEL_PATH
from ultralytics import YOLO

print("Loading orientation model...")
model = YOLO(ORIENTATION_MODEL_PATH)
print(f"Model loaded. Classes: {model.names}")
print(f"Model type: {type(model)}")

# Wait for streams to connect
print("\nWaiting for camera streams to connect...")
time.sleep(3)

# Test with each camera
for cam_name in ["cam2", "cam3"]:
    print(f"\n{'='*60}")
    print(f"Testing camera: {cam_name}")
    print(f"{'='*60}")
    
    # Try to read multiple frames
    for attempt in range(5):
        ret, frame = state.streams[cam_name].read()
        if ret and frame is not None:
            print(f"✅ Frame {attempt+1} read successfully! Shape: {frame.shape}")
            
            # Resize for orientation model
            orientation_frame = cv2.resize(frame, (224, 224))
            print(f"Resized frame shape: {orientation_frame.shape}")
            
            # Run inference
            print("Running orientation inference...")
            results = model(orientation_frame, verbose=False)
            
            if results and len(results) > 0:
                # Check for probs attribute (classification model)
                if hasattr(results[0], 'probs') and results[0].probs is not None:
                    print(f"\n✅ Classification model detected!")
                    top1_idx = results[0].probs.top1
                    top1_conf = results[0].probs.top1conf.item()
                    class_name = model.names[top1_idx]
                    print(f"Top1 class: {class_name} (index: {top1_idx})")
                    print(f"Top1 confidence: {top1_conf:.4f}")
                    print(f"All probabilities: {results[0].probs.data}")
                    
                # Check for boxes attribute (detection model)
                elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    print(f"\n✅ Detection model detected!")
                    print(f"Number of detections: {len(results[0].boxes)}")
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        cls_ids = boxes.cls.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
                            print(f"  Detection {i}: class={model.names[int(cls_id)]}, conf={conf:.4f}")
                else:
                    print("\n❌ No probs or boxes found in results!")
            else:
                print("❌ No results from model!")
            break
        else:
            print(f"⚠️ Attempt {attempt+1}: Failed to read frame from {cam_name}")
            time.sleep(1)
    else:
        print(f"❌ Could not read any frames from {cam_name} after 5 attempts")

print("\n" + "="*60)
print("Test complete")