const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000";

export type CancerType = "brain" | "breast";
export type ResultClass = "Normal" | "Benign" | "Malignant";

export type PredictResponse = {
  cancer_type: CancerType;
  prediction: ResultClass;
  confidence: number;
  localization_url?: string | null;
};

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
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Predict failed (${res.status}): ${msg}`);
  }
  return res.json();
}

export async function saveCase(payload: CreateCaseRequest) {
  const res = await fetch(`${API_BASE}/cases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Save case failed (${res.status})`);
  return res.json();
}

export async function fetchCases() {
  const res = await fetch(`${API_BASE}/cases`);
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
  const res = await fetch(`${API_BASE}/api/geotracker/summary`);
  if (!res.ok) throw new Error(`Geo summary failed: ${res.status}`);
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
  const res = await fetch(`${API_BASE}/api/reviews/pending`);
  if (!res.ok) throw new Error(`Fetch reviews failed: ${res.status}`);
  return res.json();
}

export async function approveReview(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/reviews/${caseId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewerId: "demo-reviewer" }),
  });
  if (!res.ok) throw new Error(`Approve failed: ${res.status}`);
}

export async function rejectReview(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/reviews/${caseId}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
  createdAt: string;
  segmentation?: { overlay_url?: string } | null;
  input_image_url?: string | null;
  gradcam_url?: string | null;
  reject_reason?: string | null;
};

export async function fetchEhrRecords(): Promise<{ records: EhrRecord[] }> {
  const res = await fetch(`${API_BASE}/api/ehr`);
  if (!res.ok) throw new Error(`Fetch EHR failed: ${res.status}`);
  return res.json();
}
