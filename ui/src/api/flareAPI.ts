export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "https://reassign-guiding-grass.ngrok-free.dev";
const NGROK_HEADERS = { "ngrok-skip-browser-warning": "true" } as const;

export type CancerType = "brain" | "breast";
export type ResultClass = "Normal" | "Benign" | "Malignant";

export type ClassProbability = { label: string; value: number };

export type PredictResponse = {
  cancer_type: CancerType;
  prediction: ResultClass;
  confidence: number;
  localization_url?: string | null;
  probabilities?: ClassProbability[] | null;
  gradcam_ready?: boolean;
  caseId?: string | null;
  review_required?: boolean;
};

/** Shape returned by /api/mri/predict when the BRISC/BraTS-style API is enabled (folder + future single). */
export type MriPredictClassification = {
  label: "Abnormal" | "Normal" | string;
  confidence: number;
  tumor_px?: number;
};

export type MriPredictSegmentation = {
  original_url: string | null;
  mask_url: string | null;
  overlay_url: string | null;
  tumor_pixel_count?: number;
};

export type MriFolderPredictView = {
  kind: "mri_api_v2";
  classification: MriPredictClassification;
  segmentation: MriPredictSegmentation;
  /** Raw input slice URL from API when distinct from segmentation.original_url */
  input_image_url?: string | null;
  caseId?: string | null;
  review_required?: boolean;
};

export type LegacyPredictView = {
  kind: "legacy";
  pred: PredictResponse;
};

export type CancerScanResult = MriFolderPredictView | LegacyPredictView;

export type CreateCaseRequest = {
  first_name: string;
  last_name: string;
  dob: string;
  medical_id: string;
  location: string;
  cancer_type: CancerType;
  prediction: ResultClass;
  confidence: number;
  localization_url?: string | null;
};

export async function predictScan(params: {
  cancerType: CancerType;
  file: File;
  hospitalId: string;
  firstName: string;
  lastName: string;
  dob: string;
  medicalId: string;
}): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("hospitalId", params.hospitalId);
  form.append("first_name", params.firstName);
  form.append("last_name", params.lastName);
  form.append("dob", params.dob);
  form.append("patient_id", params.medicalId);
  const res = await fetch(
    `${API_BASE}/predict?cancer_type=${encodeURIComponent(params.cancerType)}`,
    { method: "POST", headers: NGROK_HEADERS, body: form }
  );
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Predict failed (${res.status}): ${msg}`);
  }
  return res.json();
}

/** Turn relative /static/... paths into absolute URLs for <img src>. */
function absolutizeAssetUrl(path: string | null | undefined): string | null | undefined {
  if (path == null || path === "") return path;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = API_BASE.replace(/\/$/, "");
  return path.startsWith("/") ? `${base}${path}` : `${base}/${path}`;
}

/** Maps Flask /api/mri/predict JSON to the slim shape the cancer demo uses. */
export function mapMriApiJsonToPredictResponse(cancerType: CancerType, body: Record<string, unknown>): PredictResponse {
  let rc = (body.result_class as string) || "Normal";
  if (!["Normal", "Benign", "Malignant"].includes(rc)) {
    const pl = String(body.pred_label || "").toLowerCase();
    rc = ["abnormal", "tumor", "malignant"].includes(pl) ? "Malignant" : "Normal";
  }
  const seg = (body.segmentation as Record<string, unknown>) || {};
  const loc = (seg.overlay_url as string) || (seg.mask_url as string) || (body.gradcam_url as string);
  const out: PredictResponse = {
    cancer_type: cancerType,
    prediction: rc as ResultClass,
    confidence: Number(body.confidence ?? 0),
    localization_url: absolutizeAssetUrl(loc) ?? null,
    gradcam_ready: Boolean(body.gradcam_ready),
    caseId: (body.caseId as string) ?? null,
    review_required: Boolean(body.review_required),
  };
  if (body.probabilities != null) out.probabilities = body.probabilities as PredictResponse["probabilities"];
  return out;
}

function segUrl(v: unknown): string | null {
  if (v == null || v === "") return null;
  const s = String(v);
  return absolutizeAssetUrl(s) ?? null;
}

/** Prefer nested classification/segmentation when present; else legacy FLARE dict. */
export function parseMriPredictBody(body: Record<string, unknown>): CancerScanResult {
  const cls = body.classification;
  if (cls && typeof cls === "object" && cls !== null && "label" in cls) {
    const c = cls as Record<string, unknown>;
    const rawSeg = body.segmentation;
    const s =
      rawSeg && typeof rawSeg === "object" && rawSeg !== null ? (rawSeg as Record<string, unknown>) : {};
    const fromSegOriginal = segUrl(s.original_url);
    const fromRootInput = segUrl(body.input_image_url);
    const originalUrl = fromSegOriginal ?? fromRootInput ?? null;
    return {
      kind: "mri_api_v2",
      input_image_url: fromRootInput,
      caseId: (body.caseId as string) ?? null,
      review_required: Boolean(body.review_required),
      classification: {
        label: String(c.label ?? "Normal"),
        confidence: Math.min(1, Math.max(0, Number(c.confidence ?? 0))),
        tumor_px: typeof c.tumor_px === "number" ? c.tumor_px : undefined,
      },
      segmentation: {
        original_url: originalUrl,
        mask_url: segUrl(s.mask_url),
        overlay_url: segUrl(s.overlay_url),
        tumor_pixel_count: (() => {
          const v = s.tumor_pixel_count;
          const n = typeof v === "number" ? v : typeof v === "string" ? parseInt(v, 10) : NaN;
          return Number.isFinite(n) ? n : undefined;
        })(),
      },
    };
  }
  return { kind: "legacy", pred: mapMriApiJsonToPredictResponse("brain", body) };
}

/**
 * BraTS patient folder: POST /api/mri/predict with multipart fields t1n, t1c, t2w, t2f (+ patient info).
 */
export async function predictMriBraTSFolder(params: {
  sequences: { t1n: File; t1c: File; t2w: File; t2f: File };
  hospitalId: string;
  firstName: string;
  lastName: string;
  dob: string;
  medicalId: string;
}): Promise<CancerScanResult> {
  const form = new FormData();
  form.append("patient_id", params.medicalId);
  form.append("modality", "brain_brats");
  form.append("hospitalId", params.hospitalId);
  form.append("first_name", params.firstName);
  form.append("last_name", params.lastName);
  form.append("dob", params.dob);
  const { t1n, t1c, t2w, t2f } = params.sequences;
  form.append("t1n", t1n);
  form.append("t1c", t1c);
  form.append("t2w", t2w);
  form.append("t2f", t2f);

  const res = await fetch(`${API_BASE}/api/mri/predict`, {
    method: "POST",
    headers: NGROK_HEADERS,
    body: form,
  });
  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok) {
    const err = (body.error as string) || `Predict failed (${res.status})`;
    throw new Error(err);
  }
  return parseMriPredictBody(body);
}

export async function predictCtFile(
  file: File,
  patientId: string,
  hospitalId: string,
  firstName?: string,
  lastName?: string,
  dob?: string
): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append("file", file);
  form.append("patient_id", patientId);
  form.append("hospitalId", hospitalId);
  if (firstName) form.append("first_name", firstName);
  if (lastName) form.append("last_name", lastName);
  if (dob) form.append("dob", dob);
  const res = await fetch(`${API_BASE}/api/ct/predict`, {
    method: "POST",
    headers: NGROK_HEADERS,
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`CT predict failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<Record<string, unknown>>;
}

export async function predictFusion(
  ctFile: File | null,
  mriFile: File | null,
  patientId: string,
  hospitalId: string
): Promise<Record<string, unknown>> {
  const form = new FormData();
  if (ctFile) form.append("ct_file", ctFile);
  if (mriFile) form.append("mri_file", mriFile);
  form.append("patient_id", patientId);
  form.append("hospitalId", hospitalId);
  const res = await fetch(`${API_BASE}/api/fusion/predict`, {
    method: "POST",
    headers: NGROK_HEADERS,
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Fusion predict failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<Record<string, unknown>>;
}

export async function saveCase(payload: CreateCaseRequest) {
  const res = await fetch(`${API_BASE}/cases`, {
    method: "POST",
    headers: { ...NGROK_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Save case failed (${res.status})`);
  return res.json();
}

export async function fetchCases() {
  const res = await fetch(`${API_BASE}/cases`, { headers: NGROK_HEADERS });
  if (!res.ok) throw new Error(`Fetch cases failed: ${res.status}`);
  return res.json();
}

// ── Geotracker ────────────────────────────────────────────────────────────────

export type HospitalSummary = {
  hospitalId: string;
  name: string;
  latitude: number;
  longitude: number;
  approvedAbnormalCount: number;
  pendingCount: number;
  severityColor: string;
  trend: "up" | "flat" | "attention";
};

export type GeoSummary = {
  hospitals: HospitalSummary[];
  totals: { pending: number; approvedAbnormal: number };
};

export async function fetchGeoSummary(): Promise<GeoSummary> {
  const res = await fetch(`${API_BASE}/api/geotracker/summary`, { headers: NGROK_HEADERS });
  if (!res.ok) throw new Error(`Geo summary failed: ${res.status}`);
  return res.json();
}

export type OutbreakStatus = {
  outbreak_level: string;
  outbreak_color: string;
  triggered: boolean;
  total_approved_abnormal: number;
  expected_baseline: number;
  percent_of_baseline: number;
  hospital_counts: Record<string, number>;
  hotspot_hospitals: string[];
  population: number;
  threshold_elevated: number;
  threshold_critical: number;
};

export async function fetchOutbreakStatus(): Promise<OutbreakStatus> {
  const res = await fetch(`${API_BASE}/api/outbreak/status`, { headers: NGROK_HEADERS });
  if (!res.ok) throw new Error("Failed to fetch outbreak status");
  return res.json();
}

export type ReviewCase = {
  caseId: string;
  hospitalId: string;
  hospitalName: string;
  patient_id: string;
  result_class: string;
  confidence: number;
  review_status: "pending" | "approved" | "rejected";
  createdAt: string;
  segmentation?: { overlay_url?: string } | null;
  input_image_url?: string | null;
};

export async function fetchPendingReviews(): Promise<{ cases: ReviewCase[] }> {
  const res = await fetch(`${API_BASE}/api/reviews/pending`, { headers: NGROK_HEADERS });
  if (!res.ok) throw new Error(`Fetch reviews failed: ${res.status}`);
  return res.json();
}

export async function approveReview(
  caseId: string,
  opts: { reviewerName: string; signature: string }
): Promise<void> {
  const reviewerName = opts.reviewerName.trim();
  const signature = opts.signature.trim();
  const res = await fetch(`${API_BASE}/api/reviews/${caseId}/approve`, {
    method: "POST",
    headers: { ...NGROK_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({
      reviewerName,
      signature,
      reviewerId: reviewerName,
    }),
  });
  if (!res.ok) throw new Error(`Approve failed: ${res.status}`);
}

export async function rejectReview(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/reviews/${caseId}/reject`, {
    method: "POST",
    headers: { ...NGROK_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({ reviewerId: "demo-reviewer", reason: "rejected by reviewer" }),
  });
  if (!res.ok) throw new Error(`Reject failed: ${res.status}`);
}

// ── EHR ───────────────────────────────────────────────────────────────────────

export type EhrRecord = {
  caseId: string;
  patient_id: string;
  firstName: string | null;
  lastName: string | null;
  dob: string | null;
  medicalId?: string | null;
  hospitalId: string;
  hospitalName: string;
  modality: string;
  result_class: string;
  confidence: number;
  review_status: string;
  /** Backend review/EHR flag for abnormal screening queue */
  is_abnormal?: boolean;
  createdAt: string;
  segmentation?: { overlay_url?: string } | null;
  input_image_url?: string | null;
  gradcam_url?: string | null;
  reject_reason?: string | null;
};

export async function fetchEhrRecords(): Promise<{ records: EhrRecord[] }> {
  const res = await fetch(`${API_BASE}/api/ehr`, { headers: NGROK_HEADERS });
  if (!res.ok) throw new Error(`Fetch EHR failed: ${res.status}`);
  return res.json();
}
