# plate_validation.py

import re
from typing import Tuple, Optional

from config import (
    VALID_STATE_CODES,
    STATE_CORRECTION_MAP,
    CHAR_CORRECTION_MAP,
    PLATE_REGEX,
    BH_PLATE_REGEX
)

# ─────────────────────────────────────────────────────────────
# EXTRA SPECIAL FORMAT SUPPORT
# Example: TN123456, KA000001
# Format: 2 letters + 6 digits
# ─────────────────────────────────────────────────────────────
SPECIAL_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{6}$")

def strip_india_marking(plate: str) -> str:
    """
    Remove OCR artefacts caused by the vertical IND logo
    printed between the district code and series.

    Examples:
        TN28INDBF2801  -> TN28BF2801
        TN28INOBF2801  -> TN28BF2801
        TN28NOBF2801   -> TN28BF2801
        TN28NDBF2801   -> TN28BF2801
        PY05H4322IND   -> PY05H4322
        PY05H4322INDIA -> PY05H4322
    """
    if not plate:
        return plate

    original = plate
    plate = plate.upper().strip()

    # Remove suffix artefacts
    plate = re.sub(r"(INDIA|IND|INO|ND|NO)$", "", plate)

    # Remove embedded IND artefacts after district code
    plate = re.sub(r"^([A-Z]{2}\d{2})(IND|INO|ND|NO)([A-Z0-9]+)$", r"\1\3", plate)

    if plate != original:
        print(f"strip_india_marking: '{original}' -> '{plate}'")

    return plate

# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_plate(plate: str) -> Tuple[bool, str]:
    if not plate:
        return False, "empty"

    plate = plate.upper().strip()
    plate = re.sub(r"[-\s]", "", plate)
    plate = strip_india_marking(plate)

    if len(plate) < 6:
        return False, "too short"

    # BH SERIES
    if BH_PLATE_REGEX.match(plate):
        return True, "BH-series"

    # SPECIAL FORMAT (2 letters + 6 digits)
    if SPECIAL_PLATE_REGEX.match(plate):
        state_code = plate[:2]
        if state_code not in VALID_STATE_CODES:
            return False, f"unknown state code '{state_code}'"
        return True, "special-format"

    # STANDARD INDIAN FORMAT
    if not PLATE_REGEX.match(plate):
        return False, "format mismatch"

    state_code = plate[:2]
    if state_code not in VALID_STATE_CODES:
        return False, f"unknown state code '{state_code}'"

    return True, "ok"

# ─────────────────────────────────────────────────────────────
# ENHANCED CORRECTION
# ─────────────────────────────────────────────────────────────
def correct_plate(plate: str) -> Optional[str]:
    """
    Correct OCR errors in Indian number plates.

    Supports:
        STANDARD: AA DD A DDDD, AA DD AA DDDD, AA DD AAA DDDD
        SPECIAL:   AA DDDDDD

    Also handles common missing‑character errors (e.g., TN28B39970 → TN28BB9970).
    """
    if not plate or len(plate) < 2:
        return None

    # Normalize
    plate = plate.upper().strip()
    plate = re.sub(r"[-\s]", "", plate)
    plate = strip_india_marking(plate)

    length = len(plate)
    chars = list(plate)

    if len(chars) < 2:
        return None

    # State code correction
    chars[0] = STATE_CORRECTION_MAP.get(chars[0], chars[0])
    chars[1] = STATE_CORRECTION_MAP.get(chars[1], chars[1])

    def correct_segment(chars_list, expect_letter):
        corrected = []
        for ch in chars_list:
            if expect_letter:
                if ch in CHAR_CORRECTION_MAP and CHAR_CORRECTION_MAP[ch].isalpha():
                    corrected.append(CHAR_CORRECTION_MAP[ch])
                elif ch.isalpha():
                    corrected.append(ch)
                else:
                    corrected.append(ch)  # let validation fail later
            else:
                if ch in CHAR_CORRECTION_MAP and CHAR_CORRECTION_MAP[ch].isdigit():
                    corrected.append(CHAR_CORRECTION_MAP[ch])
                elif ch.isdigit():
                    corrected.append(ch)
                else:
                    corrected.append(ch)
        return ''.join(corrected)

    # ─── SPECIAL FORMAT ────────────────────────────────────────
    if length == 8:
        seg_state = chars[0:2]
        seg_number = chars[2:8]
        state = correct_segment(seg_state, expect_letter=True)
        number = correct_segment(seg_number, expect_letter=False)
        corrected_plate = state + number
        valid, _ = validate_plate(corrected_plate)
        return corrected_plate if valid else None

    # ─── STANDARD FORMATS ──────────────────────────────────────
    if length == 9:
        seg_state = chars[0:2]
        seg_district = chars[2:4]
        seg_series = chars[4:5]
        seg_number = chars[5:9]
    elif length == 10:
        seg_state = chars[0:2]
        seg_district = chars[2:4]
        seg_series = chars[4:6]
        seg_number = chars[6:10]
    elif length == 11:
        seg_state = chars[0:2]
        seg_district = chars[2:4]
        seg_series = chars[4:7]
        seg_number = chars[7:11]
    else:
        # ─── HANDLE COMMON MISSING‑CHARACTER CASES ──────────────
        # If plate has 10 chars and ends with 9970, might be missing one letter.
        if length == 10 and plate.endswith("9970") and re.match(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$", plate):
            # Try inserting a letter (guess from the existing one)
            original_series_letter = plate[4]
            candidates = []
            # Try inserting the same letter, and also common alternatives
            for letter in [original_series_letter, 'B', 'S', '3', '5', '8', '9']:
                candidate = plate[:4] + letter + plate[4:]
                valid, _ = validate_plate(candidate)
                if valid:
                    candidates.append(candidate)
            if candidates:
                return candidates[0]  # return first valid one
        return None

    # Apply corrections
    state = correct_segment(seg_state, expect_letter=True)
    district = correct_segment(seg_district, expect_letter=False)
    series = correct_segment(seg_series, expect_letter=True)
    number = correct_segment(seg_number, expect_letter=False)

    corrected_plate = state + district + series + number

    valid, _ = validate_plate(corrected_plate)
    return corrected_plate if valid else None

# ─────────────────────────────────────────────────────────────
# LEGACY
# ─────────────────────────────────────────────────────────────
def extract_partial_plate(plate: str) -> Optional[str]:
    return None