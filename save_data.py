from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import uuid
import logging

MONGO_URI = "mongodb://localhost:27017/"

logger = logging.getLogger(__name__)

# We still print the initial connection message to console,
# but we'll also log it.
try:
    client = MongoClient(MONGO_URI)
    db = client["IOCL"]
    collection = db["iocl_datas"]
    msg = "MongoDB connected successfully"
    logger.info(msg)
    print(msg)  # keep for console, but it's not critical
except Exception as e:
    msg = f"MongoDB connection failed: {e}"
    logger.error(msg)
    print(msg)


def check_duplicate_entry(
    vehicle_number: str,
    record_type: str,
    time_window_minutes: int = 15
) -> bool:
    try:
        cutoff_time = datetime.now() - timedelta(
            minutes=time_window_minutes
        )

        existing = collection.find_one({
            "vehicleNumber": vehicle_number,
            "type": record_type,
            "createdAt": {"$gte": cutoff_time}
        })

        if existing:
            logger.warning(
                f"DUPLICATE DETECTED | "
                f"plate={vehicle_number} | "
                f"type={record_type} | "
                f"existing_id={existing.get('_id')} | "
                f"createdAt={existing.get('createdAt')}"
            )
            return True

        return False

    except Exception as e:
        logger.exception(f"Duplicate check failed: {e}")
        return False


def save_iocl_data(
    anpr_image_path,
    top_image_paths,          # Now accepts list or single string
    side_image_paths,         # Now accepts list or single string
    vehicle_number,
    time,
    top_count,
    side_count,
    total_count,
    record_type,
    is_jumbo=False,
    cylinder_color_detected="UNKNOWN",
    truck_type="UNKNOWN",
    kg14_count=0,
    kg19_count=0,
    kg47_count=0,
    kg425_count=0
):
    try:
        # ─── Use logger instead of print ────────────────────────────────
        logger.info("----- IOCL DATA INSERT START -----")
        logger.info(f"Vehicle Number: {vehicle_number}")
        logger.info(f"Type: {record_type}")
        logger.info(f"Top Count: {top_count}")
        logger.info(f"Side Count: {side_count}")
        logger.info(f"Total Count: {total_count}")
        logger.info(f"Is Jumbo: {is_jumbo}")
        logger.info(f"Detected Color: {cylinder_color_detected}")
        logger.info(f"Truck Type: {truck_type}")

        if check_duplicate_entry(vehicle_number, record_type):
            logger.warning(
                f"DUPLICATE FOUND: {vehicle_number} ({record_type}) "
                f"already exists in last 45 minutes"
            )
            logger.info("----- IOCL DATA INSERT SKIPPED (DUPLICATE) -----\n")
            return None

        # Normalize inputs to lists
        if isinstance(top_image_paths, str):
            top_image_paths = [top_image_paths] if top_image_paths else []
        if isinstance(side_image_paths, str):
            side_image_paths = [side_image_paths] if side_image_paths else []
        # Also allow None
        if top_image_paths is None:
            top_image_paths = []
        if side_image_paths is None:
            side_image_paths = []

        # Primary images (first of each, or None)
        top_view_primary = top_image_paths[0] if top_image_paths else None
        side_view_primary = side_image_paths[0] if side_image_paths else None

        code = f"IOCL{str(uuid.uuid4().int)[:5]}"

        cylinder_count = {
            "kg5": 0,
            "kg19": 0,
            "kg425": 0,
            "kg14": 0,
            "kg47": 0
        }

        if truck_type.startswith("425 KG +"):
            cylinder_count["kg425"] = int(kg425_count)
            cylinder_count["kg19"] = int(kg19_count)
            cylinder_count["kg14"] = int(kg14_count)
            cylinder_count["kg47"] = int(kg47_count)

        elif truck_type == "425 KG JUMBO TRUCK":
            cylinder_count["kg425"] = int(kg425_count)

        elif truck_type in ("MIXED LOAD", "MIXED"):
            cylinder_count["kg14"] = int(kg14_count)
            cylinder_count["kg19"] = int(kg19_count)

        elif truck_type == "MIXED_19_47":
            cylinder_count["kg47"] = int(kg47_count)
            cylinder_count["kg19"] = int(kg19_count)

        elif truck_type == "19 KG TRUCK":
            cylinder_count["kg19"] = int(total_count)

        elif truck_type == "14.5 KG TRUCK":
            cylinder_count["kg14"] = int(total_count)

        elif truck_type == "UNKNOWN":
            cylinder_count["kg14"] = int(kg14_count)
            cylinder_count["kg19"] = int(kg19_count)
            cylinder_count["kg425"] = int(kg425_count)
            cylinder_count["kg47"] = int(kg47_count)
        else:
            # Fallback: if none of the above, use total_count as kg14 (legacy)
            cylinder_count["kg14"] = int(total_count)

        cylinder_color = {
            "red": 0,
            "blue": 0,
            "white": 0,
            "purple": 0,
            "red_blue": 0
        }

        if truck_type.startswith("425 KG +"):
            cylinder_color["blue"] = kg19_count
            cylinder_color["white"] = kg425_count

        elif truck_type in ("MIXED LOAD", "MIXED"):
            cylinder_color["red"] = kg14_count
            cylinder_color["blue"] = kg19_count

        elif truck_type == "MIXED_19_47":
            cylinder_color["red_blue"] = kg47_count
            cylinder_color["blue"] = kg19_count

        elif cylinder_color_detected == "RED":
            cylinder_color["red"] = total_count

        elif cylinder_color_detected == "BLUE":
            cylinder_color["blue"] = total_count

        elif cylinder_color_detected == "WHITE":
            cylinder_color["white"] = total_count

        if truck_type == "UNKNOWN":
            if kg425_count > 0:
                cylinder_color["white"] = kg425_count
            elif kg14_count > 0:
                cylinder_color["red"] = kg14_count
            elif kg19_count > 0:
                cylinder_color["blue"] = kg19_count

        logger.info(
            "[MONGO_COUNTS] "
            f"truck_type={truck_type} "
            f"kg425={kg425_count} "
            f"kg14={kg14_count} "
            f"kg19={kg19_count} "
            f"kg47={kg47_count} "
            f"total={total_count}"
        )

        logger.info(
            f"[MONGO_COUNTS] cylinderCount={cylinder_count}"
        )

        document = {
            "code": code,
            "vehicleNumber": vehicle_number,
            "vehicleImage": anpr_image_path,
            "cylinderCount": cylinder_count,
            "cylinderCountColor": cylinder_color,
            "truckType": truck_type,
            "sap": {
                "customerName": "",
                "sapBillNo": "",
                "vehicleNumber": vehicle_number,
                "kg5": 0,
                "kg19": 0,
                "kg425": 0,
                "kg14": 0,
                "kg47": 0
            },
            "truckImages": {
                "anpr": anpr_image_path,
                "topView": top_view_primary,
                "topViews": top_image_paths,
                "sideView": side_view_primary,
                "sideViews": side_image_paths
            },
            "addedBy": "AI System",
            "modifiedBy": "AI System",
            "createdAt": time if isinstance(time, datetime) else datetime.now(),
            "updatedAt": datetime.now(),
            "plant": "IOCL",
            "type": record_type,
            "side_count": side_count,
            "top_count": top_count,
            "total_count": total_count
        }

        logger.debug("Document prepared: %s", document)

        result = collection.insert_one(document)

        logger.info(
            f"Mongo insert success | "
            f"id={result.inserted_id} | "
            f"plate={vehicle_number}"
        )

        logger.info("----- IOCL DATA INSERT END -----\n")

        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"Mongo Insert Failed: {e}", exc_info=True)
        logger.info("----- IOCL DATA INSERT FAILED -----\n")
        return None