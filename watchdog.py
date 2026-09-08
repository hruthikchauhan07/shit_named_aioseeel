import logging
import time
import numpy as np
import state
from config import JOB_TIMEOUT_SEC

logger = logging.getLogger(__name__)


def job_watchdog():
    """Monitor and timeout stuck jobs"""
    logger.info("=" * 60)
    logger.info("WATCHDOG STARTED - Monitoring for stuck jobs")
    logger.info(f"Timeout: {JOB_TIMEOUT_SEC} seconds")
    logger.info("=" * 60)
    
    last_status_log = time.time()
    
    while not state.STOP_EVENT.is_set():
        time.sleep(5)
        
        job = state.current_job
        if job is None:
            continue

        elapsed = time.time() - job.started_at
        
        # Warning at half timeout
        if elapsed > JOB_TIMEOUT_SEC / 2 and not job.completed:
            logger.warning(f"[watchdog] ⚠️ Job {job.plate} running for {elapsed:.1f}s (half timeout)")
            logger.warning(f"[watchdog] Top: stable={job.top_stable}, final={job.top_final}, raw_counts={list(job.top_raw_counts) if job.top_raw_counts else []}")
            logger.warning(f"[watchdog] Side: stable={job.side_stable}, final={job.side_final}, raw_counts={list(job.side_raw_counts) if job.side_raw_counts else []}")
        
        # Timeout at full timeout
        if elapsed > JOB_TIMEOUT_SEC:
            logger.warning("=" * 60)
            logger.warning(f"[watchdog] ⏰ JOB TIMEOUT for {job.plate} after {elapsed:.1f}s")
            
            # ─── FORCE TOP COMPLETION ──────────────────────────────────────
            if not job.top_stable and job.top_final is None:
                if job.top_raw_counts:
                    median_count = int(np.median(list(job.top_raw_counts)))
                    job.top_final = median_count
                    job.top_stable = True
                    logger.warning(f"[watchdog] Forced top count to median of raw counts: {median_count}")
                else:
                    # If no raw counts, use side total as fallback
                    if job.side_stable and job.side_col_counts:
                        side_total = sum(job.side_col_counts.values())
                        job.top_final = side_total
                        job.top_stable = True
                        logger.warning(f"[watchdog] Forced top count to side total: {side_total}")
                    else:
                        job.top_final = 0
                        job.top_stable = True
                        logger.warning("[watchdog] No raw counts and no side data, forced top_final=0")
            
            # ─── FORCE SIDE COMPLETION ──────────────────────────────────────
            if not job.side_stable and job.side_final is None:
                if job.side_raw_counts:
                    median_count = int(np.median(list(job.side_raw_counts)))
                    job.side_final = median_count
                    job.side_stable = True
                    logger.warning(f"[watchdog] Forced side count to median of raw counts: {median_count}")
                else:
                    # Default side layers based on truck type if known
                    if job.truck_type == "14.5 KG TRUCK":
                        job.side_final = 3
                    elif job.truck_type == "19 KG TRUCK":
                        job.side_final = 2
                    else:
                        job.side_final = 3  # safe default
                    job.side_stable = True
                    logger.warning(f"[watchdog] Forced side count to default: {job.side_final}")
            
            # ─── Force finish ──────────────────────────────────────────────
            logger.warning(f"[watchdog] Forcing job completion for {job.plate}")
            state.finish_job(job)
            logger.warning("=" * 60)