/**
 * Patient-folder upload: BraTS-style MRI names vs future CT names.
 * *-seg.nii.gz is ignored (labels).
 */

export type DetectedFolderKind = "mri_brats" | "ct" | "unknown";

export const MRI_SEQUENCE_KEYS = ["t1n", "t1c", "t2w", "t2f"] as const;
export type MriSequenceKey = (typeof MRI_SEQUENCE_KEYS)[number];

/** BraTS filename suffix → human-readable modality (checklist labels). */
export const MRI_SEQUENCE_LABELS: Record<MriSequenceKey, string> = {
  t1n: "T1 (native)",
  t1c: "T1 contrast",
  t2w: "T2",
  t2f: "FLAIR",
};

const SEG_IGNORE = /-seg\.nii\.gz$/i;

export function shouldIgnoreFolderFile(name: string): boolean {
  return SEG_IGNORE.test(name.trim());
}

function normPath(p: string): string {
  return p.replace(/\\/g, "/").toLowerCase();
}

function fileSearchNames(f: File): string[] {
  const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
  return [f.name, rel || ""].filter(Boolean).map(normPath);
}

/** Match BraTS-style *-t1n.nii.gz (case-insensitive) on basename or relative path. */
export function findSequenceFile(files: File[], tail: string): File | null {
  const t = tail.toLowerCase();
  for (const f of files) {
    for (const p of fileSearchNames(f)) {
      const base = p.split("/").pop() || p;
      if (shouldIgnoreFolderFile(base)) continue;
      if (p.endsWith(t) || base.endsWith(t)) return f;
    }
  }
  return null;
}

/** Future CT folder: *-ct.nii.gz (spec TBD by CT team). */
export function findCtLikeFiles(files: File[]): File[] {
  const out: File[] = [];
  for (const f of files) {
    const base = normPath(f.name).split("/").pop() || "";
    if (shouldIgnoreFolderFile(base)) continue;
    if (base.includes("-ct.nii.gz") || base.endsWith("-ct.nii")) out.push(f);
  }
  return out;
}

export function analyzePatientFolder(files: File[]): {
  kind: DetectedFolderKind;
  mriSequences: Record<MriSequenceKey, File | null>;
  ctCandidates: File[];
  ignoredSegCount: number;
} {
  let ignoredSegCount = 0;
  const filtered: File[] = [];
  for (const f of files) {
    if (shouldIgnoreFolderFile(f.name)) {
      ignoredSegCount++;
      continue;
    }
    filtered.push(f);
  }

  const mriSequences = {
    t1n: findSequenceFile(filtered, "-t1n.nii.gz"),
    t1c: findSequenceFile(filtered, "-t1c.nii.gz"),
    t2w: findSequenceFile(filtered, "-t2w.nii.gz"),
    t2f: findSequenceFile(filtered, "-t2f.nii.gz"),
  } satisfies Record<MriSequenceKey, File | null>;

  const ctCandidates = findCtLikeFiles(filtered);

  let kind: DetectedFolderKind = "unknown";
  if (mriSequences.t1n) kind = "mri_brats";
  else if (ctCandidates.length > 0) kind = "ct";

  return { kind, mriSequences, ctCandidates, ignoredSegCount };
}

export function mriBraTSFolderComplete(mri: Record<MriSequenceKey, File | null>): boolean {
  return MRI_SEQUENCE_KEYS.every((k) => mri[k] != null);
}
