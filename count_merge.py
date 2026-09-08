"""
count_merge.py

Reconciles the per-column cylinder counts detected from the TOP view
against the per-column counts detected from the SIDE view, and computes
the final total cylinder count.

-----------------------------------------------------------------------
ALGORITHM SUMMARY (Data‑Driven, No Hard‑Coded Assumptions)
-----------------------------------------------------------------------
... (unchanged header) ...
"""

import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ─── USER-TUNABLE CONSTANTS ──────────────────────────────────────────────
# How many leading columns define the "full column" value (use first 4 columns)
START_WINDOW = 4

# How many trailing columns to preserve as-is (degradation, use last 4 columns)
EDGE_WINDOW = 4

# 🆕 Minimum run length of low columns to be considered genuine partial
MIN_GENUINE_PARTIAL_RUN = 2

# 🆕 Maximum cylinders (global cap)
MAX_CYLINDER_COUNT = 350

# ─── NO HARD-CODED MAX COLUMNS OR TOTAL COUNT ────────────────────────────
# We trust the data and do not cap columns or total count.


def get_fallback_layers_for_region(region_label: Optional[str] = None) -> int:
    """
    Return a reasonable fallback layer count when side view fails for a specific region.

    - STRAIGHT (14.5 kg) region: typically 3 full layers.
    - ZIG-ZAG (19 kg) region: typically 2 layers.
    - MIXED or unknown: default to 2 (conservative).
    """
    if region_label == "STRAIGHT":
        return 3
    elif region_label == "ZIG-ZAG":
        return 2
    else:
        return 2


def get_fallback_layers_for_truck_type(truck_type: str) -> int:
    """
    Global fallback when no region information is available.
    """
    if truck_type == "14.5 KG TRUCK":
        return 3
    elif truck_type in ("19 KG TRUCK", "MIXED LOAD", "MIXED_19_47"):
        return 2
    else:
        return 2


def _full_column_value(side_col_counts: dict) -> int:
    """
    Determine the 'full column' cylinder count, based on the first
    START_WINDOW columns detected by the side view (left edge of the
    load, assumed to be fully visible / not occluded).
    """
    if not side_col_counts:
        return 0

    sorted_keys = sorted(side_col_counts.keys())
    start_keys = sorted_keys[:START_WINDOW] if len(sorted_keys) >= START_WINDOW else sorted_keys

    if not start_keys:
        return 0

    return max(side_col_counts[k] for k in start_keys)


def reconcile_side_columns(
    top_num_cols: int,
    side_col_counts: dict,
    side_num_centers_in_lastrow: int = 0,
    edge_window: int = EDGE_WINDOW,
) -> tuple[int, dict]:
    """
    Reconcile side view columns to the authoritative top view column count.

    This is a DATA-DRIVEN approach:
      1. The side dict is TRUNCATED or PADDED to exactly `effective_num_cols`.
      2. Runs of 2+ consecutive columns with counts < full_value are
         marked as GENUINE PARTIAL and NOT repaired.
      3. Single low columns are repaired to full_value (occlusion).
      4. Only the first and last `edge_window` columns are preserved as-is.

    Returns:
        (effective_num_cols, assumed_col_counts) where assumed_col_counts is a
        dict of length effective_num_cols with corrected values.
    """
    if not side_col_counts:
        return top_num_cols, {}

    # ─── CLEAN ──────────────────────────────────────────────────────────────
    cleaned = {
        int(k): min(3, int(v))
        for k, v in side_col_counts.items()
    }
    sorted_keys = sorted(cleaned.keys())
    assumed = {i: cleaned[k] for i, k in enumerate(sorted_keys)}
    side_num_cols = len(assumed)

    if side_num_cols == 0:
        return top_num_cols, {}

    # ─── AUTHORITATIVE COLUMNS ─────────────────────────────────────────────
    # We use the maximum of top_num_cols, side's own count, and bottom row centers.
    # This ensures we don't lose columns that one view missed.
    authoritative_cols = max(
        int(top_num_cols),
        int(side_num_centers_in_lastrow),
        int(side_num_cols)
    )
    effective_num_cols = authoritative_cols  # No hard-coded cap
    top_num_cols = effective_num_cols

    logger.info(
        "[count_merge] Authoritative column selection | "
        f"top_cols={top_num_cols} | side_cols={side_num_cols} | "
        f"bottom_row_centers={side_num_centers_in_lastrow} | effective_cols={effective_num_cols}"
    )

    # ─── FULL COLUMN VALUE ──────────────────────────────────────────────────
    # Derived from the first START_WINDOW columns (heuristic, not hard-coded)
    start_vals = [assumed[i] for i in range(min(START_WINDOW, side_num_cols))]
    full_value = max(start_vals) if start_vals else 3
    full_value = min(3, int(full_value))

    # ─── TRUNCATE OR PAD TO effective_num_cols ─────────────────────────────
    if len(assumed) > effective_num_cols:
        # Truncate to the effective number of columns
        assumed = {i: assumed[i] for i in range(effective_num_cols) if i in assumed}

    elif len(assumed) < effective_num_cols:
        # Pad missing columns with full_value
        missing = effective_num_cols - len(assumed)
        left_keep = min(edge_window, len(assumed))
        right_keep = min(edge_window, max(0, len(assumed) - left_keep))

        left_vals = [assumed[i] for i in range(left_keep)]
        right_vals = [assumed[len(assumed) - right_keep + i] for i in range(right_keep)]
        middle_vals = [assumed[i] for i in range(left_keep, len(assumed) - right_keep)]

        rebuilt = {}
        idx = 0
        for v in left_vals:
            rebuilt[idx] = v
            idx += 1
        for v in middle_vals:
            rebuilt[idx] = v
            idx += 1
        for _ in range(missing):
            rebuilt[idx] = full_value
            idx += 1
        for v in right_vals:
            rebuilt[idx] = v
            idx += 1
        assumed = rebuilt

    n = len(assumed)

    # ─── DETECT GENUINE PARTIAL RUNS (Data‑Driven) ────────────────────────
    # If there are 2 or more consecutive columns with counts < full_value,
    # treat them as genuinely partial (not occluded).
    vals = [assumed[i] for i in sorted(assumed.keys())]
    genuine_partial_runs = []
    low_run_start = -1
    low_run_count = 0

    for i in range(n):
        if vals[i] < full_value:
            if low_run_start == -1:
                low_run_start = i
            low_run_count += 1
        else:
            if low_run_count >= MIN_GENUINE_PARTIAL_RUN:
                genuine_partial_runs.append((low_run_start, i - 1))
            low_run_start = -1
            low_run_count = 0

    if low_run_count >= MIN_GENUINE_PARTIAL_RUN:
        genuine_partial_runs.append((low_run_start, n - 1))

    # ─── DETECT DEGRADATION START ──────────────────────────────────────────
    degradation_start = n
    consecutive_underfull = 0

    for i in range(n - 1, -1, -1):
        if vals[i] < full_value:
            consecutive_underfull += 1
        else:
            consecutive_underfull = 0

        if consecutive_underfull >= 2:
            degradation_start = i
            break

    if degradation_start == n and n > 0 and vals[-1] < full_value:
        degradation_start = n - 1

    # ─── REPAIR ONLY OCCLUDED MIDDLE (NOT genuine partial) ──────────────
    left_boundary = min(edge_window, n)

    for i in range(left_boundary, degradation_start):
        # Skip if in a genuine partial run
        skip = False
        for start, end in genuine_partial_runs:
            if start <= i <= end:
                skip = True
                break
        if not skip and assumed[i] < full_value:
            assumed[i] = full_value

    # ─── LOGGING ────────────────────────────────────────────────────────────
    logger.info(
        f"[count_merge] reconcile_side_columns | "
        f"top_cols={top_num_cols} | side_cols={side_num_cols} | "
        f"full_value={full_value} | "
        f"genuine_partial_runs={genuine_partial_runs} | "
        f"degradation_start={degradation_start} | "
        f"assumed={assumed}"
    )

    return top_num_cols, assumed


def compute_subtr(assumed_col_counts: dict, top_num_rows: int) -> int:
    """
    Compute the subtraction factor (SUBTR) = sum of missing cylinders per column × top_num_rows.
    """
    if not assumed_col_counts:
        return 0

    values = [
        min(3, int(v))
        for v in assumed_col_counts.values()
    ]

    # The full value is the most common count (not hard-coded)
    full_value = Counter(values).most_common(1)[0][0]
    full_value = min(3, int(full_value))

    missing_centers = sum(
        max(0, full_value - v)
        for v in values
    )

    subtr = missing_centers * top_num_rows

    logger.debug(
        f"[count_merge] compute_subtr | "
        f"full_value={full_value} "
        f"missing_centers={missing_centers} "
        f"top_num_rows={top_num_rows} "
        f"SUBTR={subtr}"
    )

    return int(subtr)


def merge_region_counts(
    top_num_cols: int,
    top_num_rows: int,
    side_num_rows: int,
    side_col_counts: dict,
    side_num_centers_in_lastrow: int = 0,
    fallback_layers: Optional[int] = None,
    top_region_count: Optional[int] = None,  # 🆕 if provided, use this when side_col_counts empty
) -> dict:
    """
    Full pipeline for ONE region (or the whole load if not MIXED):

      1. Reconcile side_col_counts to top_num_cols.
      2. Compute full_volume = top_num_cols * top_num_rows * side_num_rows.
      3. Compute SUBTR and TOTAL = full_volume - SUBTR.
      4. Cap total at 350 if it exceeds 350.

    Parameters:
        top_region_count: If side_col_counts is empty, use this as the total count
                          (derived from top view) instead of computing full_volume.
    """
    if top_num_cols <= 0 or top_num_rows <= 0:
        return {"assumed_col_counts": {}, "subtr": 0, "total_count": 0}

    # ─── FALLBACK: if side_num_rows is 0, use the region‑specific fallback ────
    if side_num_rows <= 0:
        if fallback_layers is None:
            fallback_layers = 2
        # Use top_region_count if available, else fallback to geometric estimate
        if top_region_count is not None and top_region_count > 0:
            total = min(top_region_count, MAX_CYLINDER_COUNT)
            logger.warning(
                f"[count_merge] side_num_rows <= 0. Using top_region_count={total} "
                f"(top_num_cols={top_num_cols} * top_num_rows={top_num_rows})"
            )
            return {
                "assumed_col_counts": dict(side_col_counts),
                "subtr": 0,
                "total_count": int(total),
            }
        else:
            total = int(top_num_cols * top_num_rows * fallback_layers)
            logger.warning(
                f"[count_merge] side_num_rows <= 0. Using fallback layers={fallback_layers} "
                f"(top_num_cols={top_num_cols} * top_num_rows={top_num_rows}) -> total={total}"
            )
            return {
                "assumed_col_counts": dict(side_col_counts),
                "subtr": 0,
                "total_count": total,
            }

    # ─── RECONCILE SIDE COLUMNS ──────────────────────────────────────────────
    top_num_cols, assumed = reconcile_side_columns(
        top_num_cols,
        side_col_counts,
        side_num_centers_in_lastrow=side_num_centers_in_lastrow,
    )

    # 🆕 If side_col_counts was empty, assumed will be empty. In that case,
    # we should use the top_region_count if provided.
    if not assumed and top_region_count is not None:
        total = min(top_region_count, MAX_CYLINDER_COUNT)
        logger.info(
            f"[count_merge] side_col_counts empty, using top_region_count={total} "
            f"(top_num_cols={top_num_cols}, top_num_rows={top_num_rows})"
        )
        return {
            "assumed_col_counts": {},
            "subtr": 0,
            "total_count": int(total),
        }

    # ─── COMPUTE FULL VOLUME ──────────────────────────────────────────────
    full_volume = top_num_cols * top_num_rows * side_num_rows

    logger.info("=" * 50)
    logger.info("[count_merge] REGION MERGE START")

    logger.info(
        f"[count_merge] Inputs -> "
        f"top_cols={top_num_cols}, "
        f"top_rows={top_num_rows}, "
        f"side_rows={side_num_rows}"
    )

    logger.info(
        f"[count_merge] Raw side_col_counts={side_col_counts}"
    )

    logger.info(
        f"[count_merge] Assumed/Reconciled columns={assumed}"
    )

    logger.info(
        f"[count_merge] Full volume calculation: "
        f"{top_num_cols} * {top_num_rows} * {side_num_rows} "
        f"= {full_volume}"
    )

    # ─── COMPUTE SUBTR AND TOTAL ──────────────────────────────────────
    subtr = compute_subtr(assumed, top_num_rows)
    total = full_volume - subtr
    total = max(0, int(total))

    # ─── Global cap: never exceed 350 ─────────────────────────────────
    if total > MAX_CYLINDER_COUNT:
        logger.warning(f"[count_merge] Total {total} exceeds max {MAX_CYLINDER_COUNT} → capped")
        total = MAX_CYLINDER_COUNT

    logger.info(f"[count_merge] SUBTR={subtr}")
    logger.info(f"[count_merge] FINAL TOTAL = {full_volume} - {subtr} = {total}")

    logger.info("=" * 50)

    logger.info(
        f"[count_merge] top_cols={top_num_cols}, top_rows={top_num_rows}, "
        f"side_rows={side_num_rows}, assumed_cols={assumed}, SUBTR={subtr} "
        f"-> TOTAL={total}"
    )

    return {
        "assumed_col_counts": assumed,
        "subtr": subtr,
        "total_count": total,
    }


def merge_mixed_regions(
    top_regions: list,
    side_regions: list,
    side_num_rows: int,
) -> dict:
    """
    For MIXED (14.5 + 19) loads: process each top region independently
    against the matching side region (matched by overlapping x-range /
    order), sum the totals.

    top_regions: list of dicts with keys:
        label, start_x, end_x, count, num_rows, num_cols, row_counts

    side_regions: list of dicts with keys:
        label, start_x, end_x, num_rows, num_cols, col_counts (dict)

    Returns:
        {
            "total_count": int,
            "kg14_total": int,
            "kg19_total": int,
            "region_results": [ ... per-region merge_region_counts output ... ],
        }
    """
    total_count = 0
    kg14_total = 0
    kg19_total = 0
    region_results = []

    for top_region in top_regions:
        t_start, t_end = top_region["start_x"], top_region["end_x"]
        t_label = top_region["label"]
        t_num_cols = top_region.get("num_cols", 0)
        t_num_rows = top_region.get("num_rows", 0)
        t_lastrow_centers = top_region.get("num_centers_in_lastrow", 0)
        t_count = top_region.get("count", 0)   # 🆕 top view count for this region

        # Find best-matching side region by x-range overlap.
        best_side = None
        best_overlap = -1
        for side_region in side_regions:
            s_start, s_end = side_region["start_x"], side_region["end_x"]
            overlap = min(t_end, s_end) - max(t_start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_side = side_region

        if best_side is not None:
            side_col_counts = best_side.get("col_counts", {})
            region_side_rows = best_side.get("num_rows", side_num_rows)
            side_lastrow = best_side.get("num_centers_in_lastrow", 0)
        else:
            side_col_counts = {}
            region_side_rows = side_num_rows
            side_lastrow = 0

        # ─── REGION‑AWARE FALLBACK ──────────────────────────────────────────
        if t_label == "STRAIGHT":
            fallback_layers = 3
        elif t_label == "ZIG-ZAG":
            fallback_layers = 2
        else:
            fallback_layers = 2

        # 🆕 Pass the top region count so we can use it when side col_counts empty
        result = merge_region_counts(
            top_num_cols=t_num_cols,
            top_num_rows=t_num_rows,
            side_num_rows=region_side_rows,
            side_col_counts=side_col_counts,
            side_num_centers_in_lastrow=side_lastrow,
            fallback_layers=fallback_layers,
            top_region_count=t_count,   # pass top count as fallback
        )

        result["label"] = t_label
        result["start_x"] = t_start
        result["end_x"] = t_end
        region_results.append(result)

        total_count += result["total_count"]
        if t_label == "STRAIGHT":
            kg14_total += result["total_count"]
        else:
            kg19_total += result["total_count"]

    return {
        "total_count": int(total_count),
        "kg14_total": int(kg14_total),
        "kg19_total": int(kg19_total),
        "region_results": region_results,
    }


def compute_final_total(job) -> dict:
    """
    Top-level entry point called from state.finish_job().

    Inspects `job` (a state.Job instance) and dispatches to the correct
    merge strategy based on job.truck_type / job.kind, handling all the
    documented edge cases:

      1. 425-only (JUMBO)              -> side bypassed, total = top count.
      2. 425 + other (JUMBO_MIXED)     -> total = 425_count + merged "other".
      3. MIXED (14.5 + 19 regions)     -> per-region merge, summed.
      4. NORMAL (single pattern)       -> single merge_region_counts call.

    Returns:
        {
            "total_count": int,
            "details": {...},  # for logging / debugging
        }
    """
    truck_type = job.truck_type or ""

    if truck_type == "19 KG TRUCK":
        # Pure 19 kg: 2 layers, so total = kg19_count * 2
        total = int(job.kg19_count * 2)
        return {
            "total_count": total,
            "details": {
                "case": "PURE_19KG",
                "top_count": job.kg19_count,
                "height": 2,
            },
        }

    # CASE 1: 425 KG ONLY -- side already bypassed (side_final=0)
    if truck_type == "425 KG JUMBO TRUCK":
        total = int(job.kg425_count)
        return {
            "total_count": total,
            "details": {"case": "JUMBO_425_ONLY", "kg425_count": job.kg425_count},
        }

    # CASE 3: MIXED 14.5 + 19 -- per-region merge
    if truck_type == "MIXED LOAD" and job.top_regions:
        side_regions = getattr(job, "side_regions", []) or []
        side_num_rows = getattr(job, "side_num_rows", 0) or 0

        result = merge_mixed_regions(
            top_regions=job.top_regions,
            side_regions=side_regions,
            side_num_rows=side_num_rows,
        )

        return {
            "total_count": result["total_count"],
            "details": {
                "case": "MIXED_REGIONS",
                "kg14_total": result["kg14_total"],
                "kg19_total": result["kg19_total"],
                "region_results": result["region_results"],
            },
        }

    # CASE 2: 425 + OTHER (JUMBO_MIXED)
    if truck_type.startswith("425 KG +"):
        side_col_counts = getattr(job, "side_col_counts", {}) or {}
        side_num_rows = getattr(job, "side_num_rows", 0) or 0

        if job.top_regions:
            # Secondary load was itself MIXED -> per-region merge.
            side_regions = getattr(job, "side_regions", []) or []
            other_result = merge_mixed_regions(
                job.top_regions, side_regions, side_num_rows
            )
            other_total = other_result["total_count"]
            details = {
                "case": "JUMBO_MIXED_REGIONS",
                "kg425_count": job.kg425_count,
                "kg14_total": other_result["kg14_total"],
                "kg19_total": other_result["kg19_total"],
                "region_results": other_result["region_results"],
            }
        else:
            # Secondary load is a single pattern
            if truck_type == "425 KG + 14.5 KG TRUCK":
                fallback_layers = 3
            else:
                fallback_layers = 2

            # 🆕 pass job.top_final as top_region_count fallback
            other_result = merge_region_counts(
                top_num_cols=max(job.top_num_cols, getattr(job, "side_num_centers_in_lastrow", 0) or 0),
                top_num_rows=job.top_num_rows,
                side_num_rows=side_num_rows,
                side_col_counts=side_col_counts,
                side_num_centers_in_lastrow=getattr(job, "side_num_centers_in_lastrow", 0) or 0,
                fallback_layers=fallback_layers,
                top_region_count=job.top_final,   # fallback to top count if side empty
            )
            other_total = other_result["total_count"]
            details = {
                "case": "JUMBO_MIXED_SINGLE",
                "kg425_count": job.kg425_count,
                "other_result": other_result,
            }

        total = int(job.kg425_count) + int(other_total)
        details["total_count"] = total

        return {
            "total_count": total,
            "details": details,
        }

    # CASE 4: NORMAL single-pattern load (14.5 or 19, STRAIGHT/ZIG-ZAG)
    side_col_counts = getattr(job, "side_col_counts", {}) or {}
    side_num_rows = getattr(job, "side_num_rows", 0) or 0

    if truck_type == "14.5 KG TRUCK":
        fallback_layers = 3
    elif truck_type == "19 KG TRUCK":
        fallback_layers = 2
    else:
        fallback_layers = 2

    result = merge_region_counts(
        top_num_cols=max(job.top_num_cols, getattr(job, "side_num_centers_in_lastrow", 0) or 0),
        top_num_rows=job.top_num_rows,
        side_num_rows=side_num_rows,
        side_col_counts=side_col_counts,
        side_num_centers_in_lastrow=getattr(job, "side_num_centers_in_lastrow", 0) or 0,
        fallback_layers=fallback_layers,
        top_region_count=job.top_final,   # 🆕 fallback to top count if side empty
    )

    return {
        "total_count": int(result["total_count"]),
        "details": {"case": "NORMAL_SINGLE", "merge_result": result},
    }