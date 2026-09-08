import os
import re
from pymongo import MongoClient
from datetime import datetime, timedelta

print("Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
db = client["IOCL"]
collection = db["iocl_datas"]

# Get today's date in YYYYMMDD format
today_str = datetime.now().strftime("%Y%m%d")

# Only check images from TODAY
image_dir = "anpr_images"
if not os.path.exists(image_dir):
    print(f"❌ Folder '{image_dir}' not found!")
    exit()

files = os.listdir(image_dir)
print(f"📁 Found {len(files)} total ANPR images.")

# Pattern to extract plate and date
pattern = re.compile(r"^(.+?)_(\d{8})_\d{6}_\d+\.jpg$")

today_images = []
for f in files:
    m = pattern.match(f)
    if m:
        plate = m.group(1)
        date_str = m.group(2)
        if date_str == today_str:
            today_images.append((plate, f))

print(f"📅 Found {len(today_images)} images from today ({today_str}).")

if not today_images:
    print("✅ No images from today. Nothing to check.")
    exit()

# Get all saved vehicle numbers from today
start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
saved_plates = set()
for doc in collection.find({"createdAt": {"$gte": start_of_day}}, {"vehicleNumber": 1}):
    saved_plates.add(doc.get("vehicleNumber"))

print(f"💾 Found {len(saved_plates)} saved vehicles today.")

missing = []
for plate, f in today_images:
    if plate not in saved_plates:
        missing.append((plate, f))

print(f"\n🔍 Found {len(missing)} images WITHOUT a corresponding DB record today:")

for plate, f in missing:
    print(f"  🚨 {plate} -> {f}")

# Optional: also show the images that have a record (for sanity)
if len(missing) == 0:
    print("✅ All today's images have matching records! No missing trucks detected.")
else:
    print(f"\n⚠️  Total missing today: {len(missing)}")