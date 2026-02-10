import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# ensure repo root is on sys.path so local package imports work when running
# the script directly (e.g. `python scripts/extract_cq500.py`)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CACHE, RAW_ZIPS, RAW, META

# Preferred Keywords for the CQ500 extraction 
PREFER_KEYWORDS = [
    "plain_thin", 
    "plainthin", 
    "thin",
    "2,55",
    "2.5",
    "plain",
]

# Keywords we don't want in the CQ500 extraction
EXCLUDE_KEYWORDS = [
    "scout", "localizer", 
    "cta", "angio", 
    "contrast", "pefusion"]

# DICOM file extentions 
DICOM_EXTS = (".dcm", ".dicom")

# ============================================================================
# LOGGING HELPER
# ============================================================================

LOG_FILE: Path = None


def setup_logging() -> None:
    """Initialize logging file in META directory."""
    global LOG_FILE
    META.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = META / f"extract_log_{timestamp}.txt"


def log(msg: str) -> None:
    """
    Log message to both console and file.
    
    Args:
        msg: Message to log
    """
    print(msg, flush=True)
    if LOG_FILE:
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}", flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_temp_root() -> Path:
    """
    Get or create temporary extraction folder.
    
    Tries to use CACHE from src/config.py if available,
    otherwise falls back to RAW / "_tmp_extract"
    
    Returns:
        Path to temporary root folder
    """
    tmp_root = CACHE / "tmp_extract"
    tmp_root.mkdir(parents=True, exist_ok=True)
    log(f"Using temp folder: {tmp_root}")
    return tmp_root


# ============================================================================
# RESUME SAFETY CHECKS
# ============================================================================

def is_patient_extracted(patient_id: str) -> bool:
    """
    Check if a patient has been fully extracted and is safe to skip.
    
    A patient is considered extracted if:
    - raw/<patient_id>/selected_series.json exists
    - CT_SELECTED folder exists and contains files
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        True if patient is fully extracted, False otherwise
    """
    out_dir = RAW / patient_id
    selected_json = out_dir / "selected_series.json"
    ct_selected = out_dir / "CT_SELECTED"
    
    # Check if JSON exists
    if not selected_json.exists():
        return False
    
    # Check if CT_SELECTED folder exists and has files
    if not ct_selected.exists() or not ct_selected.is_dir():
        return False
    
    # Check if folder has any files
    try:
        if not any(ct_selected.iterdir()):
            return False
    except (OSError, PermissionError):
        return False
    
    return True


def cleanup_incomplete_patient(patient_id: str) -> None:
    """
    Remove incomplete patient extraction to allow reprocessing.
    
    Args:
        patient_id: Patient identifier
    """
    out_dir = RAW / patient_id
    if out_dir.exists():
        log(f"  Cleaning up incomplete extraction for {patient_id}")
        shutil.rmtree(out_dir, ignore_errors=True)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track extraction progress and statistics."""
    
    def __init__(self, total_patients: int):
        self.total_patients = total_patients
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.start_time = time.time()
        
    def patient_start(self, patient_id: str, zip_path: Path) -> float:
        """
        Log start of patient extraction.
        
        Args:
            patient_id: Patient identifier
            zip_path: Path to patient zip file
            
        Returns:
            Start time for elapsed calculation
        """
        self.processed += 1
        remaining = self.total_patients - self.processed
        log(f"\n{'='*70}")
        log(f"[{self.processed}/{self.total_patients}] START {patient_id}")
        log(f"  Zip: {zip_path}")
        log(f"  Remaining: {remaining}")
        log(f"{'='*70}")
        return time.time()
    
    def stage(self, stage_name: str) -> None:
        """
        Log a processing stage.
        
        Args:
            stage_name: Name of the stage
        """
        log(f"  ➜ {stage_name}...")
    
    def patient_done(self, patient_id: str, start_time: float, slice_count: int = 0) -> None:
        """
        Log completion of patient extraction.
        
        Args:
            patient_id: Patient identifier
            start_time: Start time from patient_start()
            slice_count: Number of DICOM slices copied
        """
        elapsed = time.time() - start_time
        log(f"  ✓ DONE {patient_id}")
        log(f"  Elapsed: {elapsed:.2f}s | Slices copied: {slice_count}")
    
    def patient_skipped(self, patient_id: str) -> None:
        """
        Log skipped patient.
        
        Args:
            patient_id: Patient identifier
        """
        self.skipped += 1
        log(f"  ⊘ SKIPPED {patient_id} (already extracted)")
    
    def patient_failed(self, patient_id: str, error: str) -> None:
        """
        Log failed patient.
        
        Args:
            patient_id: Patient identifier
            error: Error message
        """
        self.failed += 1
        log(f"  ✗ FAILED {patient_id}: {error}")
    
    def print_summary(self) -> None:
        """Print final summary statistics."""
        total_time = time.time() - self.start_time
        log(f"\n{'='*70}")
        log(f"EXTRACTION COMPLETE")
        log(f"{'='*70}")
        log(f"Total patients: {self.total_patients}")
        log(f"Processed: {self.processed - self.skipped}")
        log(f"Skipped (already extracted): {self.skipped}")
        log(f"Failed: {self.failed}")
        log(f"Total time: {total_time:.2f}s")
        log(f"Log file: {LOG_FILE}")
        log(f"{'='*70}\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def norm(s: str) -> str: 
    """Normalize strings so matching is easier"""
    return "".join((s or "").lower().strip().split()) 


def is_dicom(p: Path) -> bool: 
    """Make sure it is a DICOM file"""
    return p.is_file() and p.suffix.lower() in DICOM_EXTS 


def count_dicoms(folder: Path) -> int: 
    """Count DICOM files in folder recursively"""
    return sum(1 for f in folder.rglob("*") if is_dicom(f)) 


def choose_best_series(series_folders): 
    """Choose the best series based on keywords and DICOM count"""
    log(f"    Scanning {len(series_folders)} series folders for best match...")
    candidates = [] 

    for i, sf in enumerate(series_folders):
        log(f"    Checking series {i+1}/{len(series_folders)}: {sf.name[:50]}")
        name = norm(sf.name) 

        if any(k in name for k in EXCLUDE_KEYWORDS): 
            log(f"      → Excluded (matches exclude keyword)")
            continue 

        dicom_count = count_dicoms(sf) 
        if dicom_count == 0: 
            log(f"      → No DICOMs found")
            continue

        log(f"      → Found {dicom_count} DICOMs")
        
        tier = 999
        for j, key in enumerate(PREFER_KEYWORDS): 
            if key in name: 
                tier = j 
                break 

        candidates.append((tier, -dicom_count, sf, dicom_count))
        log(f"      → Added to candidates (tier={tier}, count={dicom_count})")

    if not candidates:
        log(f"    No candidates found!")
        return None
    
    candidates.sort() 
    best_tier, _, best_folder, best_count = candidates[0]
    log(f"    Selected best series: {best_folder.name[:50]} (tier={best_tier}, count={best_count})")
    return best_folder, best_count, best_tier 


# ============================================================================
# CORE EXTRACTION LOGIC
# ============================================================================

def already_extracted(patient_id: str) -> bool:
    """
    Check if a patient has been fully extracted (backward compatibility).
    
    This function is kept for backward compatibility but now uses
    is_patient_extracted() which has stricter validation.
    """
    return is_patient_extracted(patient_id)


def extract_one_patient(zip_path: Path, tmp_root: Path, progress: ProgressTracker) -> tuple:
    """
    Extract a single patient zip file.
    
    Args:
        zip_path: Path to patient zip file
        tmp_root: Path to temporary root folder
        progress: Progress tracker instance
        
    Returns:
        Tuple of (patient_id, status, slice_count)
    """
    patient_id = zip_path.stem
    start_time = progress.patient_start(patient_id, zip_path)

    try:
        # Check if already extracted
        if is_patient_extracted(patient_id):
            progress.patient_skipped(patient_id)
            return patient_id, "SKIP_ALREADY_EXTRACTED", 0
        
        # Check if incomplete and needs cleanup
        out_dir = RAW / patient_id
        if out_dir.exists():
            log(f"  Found incomplete extraction for {patient_id}, will reprocess")
            cleanup_incomplete_patient(patient_id)
        
        # Stage 1: Prepare temp directory
        progress.stage("prepare temp")
        tmp_dir = tmp_root / patient_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Output folders
        out_dir = RAW / patient_id
        selected_dir = out_dir / "CT_SELECTED"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2: Extract the zip to temp
        progress.stage("unzip")
        log(f"    Unzipping {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        log(f"    Unzip complete")

        # Stage 3: Find dicom files and choose best series
        progress.stage("choose_best_series")
        log(f"    Scanning for DICOM files...")
        dicoms = [p for p in tmp_dir.rglob("*") if is_dicom(p)]
        log(f"    Found {len(dicoms)} DICOM files")
        
        if not dicoms:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            progress.patient_failed(patient_id, "No DICOM files found")
            return patient_id, "FAIL_NO_DICOMS", 0
        
        # Candidate series folder = parent of dicom files
        series_folders = sorted({p.parent for p in dicoms})
        log(f"    Found {len(series_folders)} unique series folders")

        # Choose best series using keywords
        best = choose_best_series(series_folders)
        if best is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            progress.patient_failed(patient_id, "No matching series found")
            return patient_id, "FAIL_NO_MATCHING_SERIES", 0
        
        best_folder, dicom_count, tier = best

        # Stage 4: Copy dicoms and put into CT_SELECTED
        progress.stage("copy selected dicoms")
        log(f"    Copying {dicom_count} DICOMs to CT_SELECTED...")
        copied = 0
        for f in best_folder.rglob("*"):
            if not is_dicom(f):
                continue
            dest = selected_dir / f.name

            if dest.exists():
                continue
            shutil.copy2(f, dest)
            copied += 1

        if copied == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            progress.patient_failed(patient_id, "No DICOM files copied")
            return patient_id, "FAIL_NO_DICOMS_COPIED", 0

        log(f"    Copied {copied} DICOMs")

        # Stage 5: Save selection metadata
        progress.stage("write selected_series.json")
        meta = {
            "patient_id": patient_id,
            "zip_file": str(zip_path),
            "selected_series_folder": best_folder.name,
            "selected_series_path_in_temp": str(best_folder),
            "preference_tier": tier,
            "dicoms_in_selected_series": dicom_count,
            "dicoms_copied": copied,
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_dir / "selected_series.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        # Stage 6: Cleanup temp
        progress.stage("cleanup temp")
        shutil.rmtree(tmp_dir, ignore_errors=True)

        progress.patient_done(patient_id, start_time, copied)
        return patient_id, "OK", copied

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        progress.patient_failed(patient_id, error_msg)
        log(f"    Exception details: {error_msg}")
        import traceback
        log(f"    Traceback: {traceback.format_exc()}")
        # Cleanup incomplete extraction
        cleanup_incomplete_patient(patient_id)
        tmp_dir = tmp_root / patient_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return patient_id, f"ERROR:{type(e).__name__}", 0


def main(limit=None):
    """Run extraction.

    Args:
        limit: int or None. If None process all new zips, otherwise process up to `limit`.
    """
    
    # Setup logging
    setup_logging()
    
    log(f"\n{'='*70}")
    log(f"CQ500 EXTRACTION STARTED")
    log(f"{'='*70}")
    log(f"Input (zips): {RAW_ZIPS}")
    log(f"Output (RAW): {RAW}")
    log(f"Meta: {META}")
    log(f"Log file: {LOG_FILE}")
    
    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    zips = sorted(Path(RAW_ZIPS).rglob("CQ500-CT-*.zip"))
    if not zips:
        log(f"ERROR: No CQ500 zips found Under: {RAW_ZIPS}")
        return

    # Count already extracted
    extracted_count = sum(1 for z in zips if is_patient_extracted(z.stem))
    remaining_count = len(zips) - extracted_count
    
    log(f"\nTotal zips found: {len(zips)}")
    log(f"Already extracted: {extracted_count}")
    log(f"Remaining: {remaining_count}")
    
    # keep only zips that are not already extracted
    todo = [zp for zp in zips if not is_patient_extracted(zp.stem)]
    if limit is not None:
        todo = todo[:limit]

    log(f"Will process (limit={limit}): {len(todo)}\n")

    tmp_root = get_temp_root()

    # Initialize progress tracker
    progress = ProgressTracker(len(todo))
    
    run_log = []
    for zp in todo:
        pid, status, copied = extract_one_patient(zp, tmp_root, progress)
        run_log.append({"patient_id": pid, "zip": str(zp), "status": status, "copied": copied})

    # Print summary
    progress.print_summary()

    # Save JSON log for compatibility
    log_path = META / f"extract_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(run_log, fp, indent=2)

    log(f"Saved JSON log to: {log_path}")
    log("DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CQ500 zips to RAW)")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Number of new patients to process (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all new zips (equivalent to --limit None)")
    args = parser.parse_args()

    # If --all specified, ensure limit is None
    limit_arg = None if args.all else args.limit

    try:
        main(limit=limit_arg)
    except Exception as e:
        log(f"Fatal error: {e}")
        print(f"Fatal error: {e}", file=sys.stderr)
        raise