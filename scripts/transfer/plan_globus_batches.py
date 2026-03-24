#!/usr/bin/env python3
"""
Plan and verify Globus (or manual) transfer batches for FLARE breast data.

Does NOT run Globus or SSH — only scans disks, compares ID sets, writes manifests.

Repo path conventions (see ml/breast/mammo/config_dbt.py, ml/breast/mri/config_mri.py,
src/config.py):

  DBT (Delta defaults):
    FLARE_DBT_DATA_ROOT  -> .../mammo_breast
    raw/train/batch1, raw/val/batch1, raw/test/batch1
    cache/train/npz + bcs_dbt_manifest.json

  Breast MRI / EA1141 (env overrides):
    FLARE_BREAST_MRI_ROOT -> .../breast_mri (or under scratch tree)
    raw/, cache/npz/, meta/mri_ea1141_manifest.csv

Examples (Windows, from repo root):

  python scripts/transfer/plan_globus_batches.py plan \\
    --modality mri \\
    --local-roots D:/ea1141/incoming D:/ea1141/stable \\
    --delta-export D:/exports/delta_patient_ids.txt \\
    --staging D:/globus_staging \\
    --batch-size 10

  python scripts/transfer/plan_globus_batches.py verify-landing \\
    --batch-manifest D:/globus_staging/batch_001/manifest.json \\
    --delta-root /scratch/bckk/flare/mri_breast/raw/ea1141_train
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _root_for_path(path: Path, roots: Sequence[Path]) -> str:
    pr = path.resolve()
    for r in roots:
        rr = Path(r).resolve()
        try:
            pr.relative_to(rr)
            return str(rr)
        except ValueError:
            continue
    return ""


def _is_probably_patient_id(name: str) -> bool:
    s = name.strip()
    if not s or s.startswith("."):
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,}$", s))


def discover_patient_dirs(
    roots: Sequence[Path],
    *,
    patient_id_regex: Optional[str] = None,
) -> Dict[str, Path]:
    """Map patient_id -> first seen directory (top-level under each root)."""
    rx = re.compile(patient_id_regex) if patient_id_regex else None
    out: Dict[str, Path] = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if rx:
                if not rx.fullmatch(name):
                    continue
            elif not _is_probably_patient_id(name):
                continue
            out.setdefault(name, child)
    return out


def count_dicoms_under(path: Path, limit: int = 5000) -> int:
    n = 0
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".dcm":
            n += 1
            if n >= limit:
                break
    return n


def flag_finished_mri_patient(
    patient_dir: Path,
    *,
    min_dicoms: int = 30,
    min_exam_subdirs: int = 1,
) -> Tuple[bool, str]:
    """
    Heuristic: patient "finished" if enough DICOMs exist under tree.
    Tune min_dicoms for your site; does not read NBIA download state.
    """
    if not patient_dir.is_dir():
        return False, "not_a_dir"
    n = count_dicoms_under(patient_dir)
    if n >= min_dicoms:
        return True, f"ok_dicoms>={min_dicoms} (approx={n})"
    subdirs = [c for c in patient_dir.iterdir() if c.is_dir()]
    if len(subdirs) >= min_exam_subdirs and n > 0:
        return False, f"partial_dicoms={n}"
    return False, f"too_few_dicoms={n}"


def load_delta_export(path: Path) -> Set[str]:
    """
    Load patient (or exam) IDs from:
      - .txt: non-empty, non-# lines
      - .csv: columns patient_id, PatientID, patient, id, subject_id
      - .json: list of strings or dicts with patient_id / PatientID / ...
    """
    path = path.resolve()
    if not path.exists():
        return set()
    suf = path.suffix.lower()
    if suf == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON must be a list")
        ids: Set[str] = set()
        for item in raw:
            if isinstance(item, str):
                ids.add(item.strip())
            elif isinstance(item, dict):
                for k in ("patient_id", "PatientID", "patient", "id"):
                    v = item.get(k)
                    if v:
                        ids.add(str(v).strip())
                        break
        return {i for i in ids if i}
    if suf == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = set()
        for row in rows:
            for k in ("patient_id", "PatientID", "patient", "id", "subject_id"):
                v = row.get(k)
                if v and str(v).strip():
                    ids.add(str(v).strip())
                    break
        return ids
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def chunk_sorted(items: Sequence[str], batch_size: int) -> List[List[str]]:
    xs = sorted(items)
    return [xs[i : i + batch_size] for i in range(0, len(xs), batch_size)]


@dataclass
class PatientEntry:
    patient_id: str
    local_path: str
    source_root: str
    finished_guess: Optional[bool] = None
    finished_note: Optional[str] = None


def write_batch_manifest(
    batch_dir: Path,
    batch_index: int,
    entries: List[PatientEntry],
    *,
    modality: str,
    extra: Optional[dict] = None,
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "modality": modality,
        "batch_index": batch_index,
        "patient_count": len(entries),
        "patients": [asdict(e) for e in entries],
    }
    if extra:
        payload["extra"] = extra
    (batch_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    csv_path = batch_dir / "patients.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["patient_id", "local_path", "source_root", "finished_guess", "finished_note"]
        )
        for e in entries:
            w.writerow(
                [
                    e.patient_id,
                    e.local_path,
                    e.source_root,
                    e.finished_guess,
                    e.finished_note or "",
                ]
            )


def stage_batch_directories(batch_dir: Path, entries: List[PatientEntry], *, copy: bool) -> None:
    for e in entries:
        src = Path(e.local_path)
        dst = batch_dir / e.patient_id
        if dst.exists():
            continue
        if copy:
            shutil.copytree(src, dst)
        else:
            try:
                dst.symlink_to(src, target_is_directory=True)
            except OSError as ex:
                print(
                    "Symlink failed (Windows may need Developer Mode or admin). "
                    "Use --stage-copy instead.",
                    file=sys.stderr,
                )
                raise RuntimeError(f"{dst} -> {src}: {ex}") from ex


def verify_landing(manifest_path: Path, delta_root: Path) -> Tuple[int, List[str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    patients = data.get("patients") or []
    missing: List[str] = []
    for p in patients:
        pid = p.get("patient_id")
        if not pid:
            continue
        if not (delta_root / pid).is_dir():
            missing.append(pid)
    return len(patients), missing


def cmd_plan(args: argparse.Namespace) -> None:
    roots = [Path(p) for p in args.local_roots]
    rx = args.patient_id_regex or None
    local_map = discover_patient_dirs(roots, patient_id_regex=rx)
    remote: Set[str] = set()
    if args.delta_export:
        remote = load_delta_export(Path(args.delta_export))

    local_ids = set(local_map.keys())
    missing = sorted(local_ids - remote) if remote else sorted(local_ids)
    already = sorted(local_ids & remote) if remote else []

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    summary = {
        "modality": args.modality,
        "local_roots": [str(Path(r).resolve()) for r in roots],
        "local_patient_count": len(local_ids),
        "delta_export": args.delta_export or None,
        "delta_id_count": len(remote),
        "missing_on_delta_count": len(missing),
        "already_on_delta_count": len(already),
        "batch_size": args.batch_size,
    }
    (staging / "plan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.only_finished:
        filtered: Dict[str, Path] = {}
        allow = set(missing) if remote else local_ids
        for pid, pdir in local_map.items():
            if pid not in allow:
                continue
            ok, _ = flag_finished_mri_patient(
                pdir,
                min_dicoms=args.min_dicoms,
                min_exam_subdirs=args.min_exam_subdirs,
            )
            if ok:
                filtered[pid] = pdir
        missing = sorted(filtered.keys())
        local_map = filtered

    batches = chunk_sorted(missing, args.batch_size)
    for i, batch_ids in enumerate(batches, start=1):
        bdir = staging / f"batch_{i:03d}"
        entries: List[PatientEntry] = []
        for pid in batch_ids:
            pdir = local_map[pid]
            fg: Optional[bool] = None
            note: Optional[str] = None
            if args.modality == "mri":
                fg, note = flag_finished_mri_patient(
                    pdir,
                    min_dicoms=args.min_dicoms,
                    min_exam_subdirs=args.min_exam_subdirs,
                )
            entries.append(
                PatientEntry(
                    patient_id=pid,
                    local_path=str(pdir.resolve()),
                    source_root=_root_for_path(pdir, roots),
                    finished_guess=fg,
                    finished_note=note,
                )
            )
        write_batch_manifest(
            bdir,
            i,
            entries,
            modality=args.modality,
            extra={
                "flare_note": "Transfer with Globus; on Delta, align delta_root with "
                "FLARE_BREAST_MRI_RAW or FLARE_DBT_RAW_* before preprocess."
            },
        )
        if args.stage_copy or args.stage_symlink:
            stage_batch_directories(
                bdir,
                entries,
                copy=args.stage_copy,
            )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(batches)} batches under {staging}")


def cmd_verify(args: argparse.Namespace) -> None:
    n, missing = verify_landing(Path(args.batch_manifest), Path(args.delta_root))
    print(f"Manifest patients: {n}")
    if missing:
        print(f"Missing ({len(missing)}):", ", ".join(missing[:80]))
        if len(missing) > 80:
            print("...")
    else:
        print("All patient folders present under delta_root.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FLARE: plan Globus batches (scan, diff, manifests); verify landing on Delta."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Scan local dirs, diff vs Delta export, write batch_*.")
    p_plan.add_argument("--modality", choices=("mri", "dbt"), required=True)
    p_plan.add_argument(
        "--local-roots",
        nargs="+",
        required=True,
        help="Top-level dirs whose immediate subfolders are patient IDs (e.g. incoming stable).",
    )
    p_plan.add_argument(
        "--delta-export",
        default="",
        help="TXT, CSV, or JSON of IDs already on Delta (empty = plan all local as missing).",
    )
    p_plan.add_argument(
        "--staging",
        required=True,
        help="Output directory for batch_001/, plan_summary.json, etc.",
    )
    p_plan.add_argument("--batch-size", type=int, default=10)
    p_plan.add_argument(
        "--patient-id-regex",
        default="",
        help="If set, only folder names matching this fullmatch regex count as patients.",
    )
    p_plan.add_argument(
        "--only-finished",
        action="store_true",
        help="MRI heuristic: only patients with enough DICOMs (see --min-dicoms).",
    )
    p_plan.add_argument("--min-dicoms", type=int, default=30)
    p_plan.add_argument("--min-exam-subdirs", type=int, default=1)
    p_plan.add_argument(
        "--stage-copy",
        action="store_true",
        help="Copy each patient tree into batch_N/patient_id (heavy).",
    )
    p_plan.add_argument(
        "--stage-symlink",
        action="store_true",
        help="Symlink patient dir into batch_N (light; may fail on Windows without dev mode).",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_ver = sub.add_parser(
        "verify-landing",
        help="Check manifest.json patient folders exist under delta_root.",
    )
    p_ver.add_argument("--batch-manifest", type=Path, required=True)
    p_ver.add_argument(
        "--delta-root",
        type=Path,
        required=True,
        help="Parent dir containing one folder per patient_id (e.g. .../raw/train/batch1).",
    )
    p_ver.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
