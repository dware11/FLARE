const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type CancerType = "brain" | "breast";
export type ResultClass = "Normal" | "Benign" | "Malignant";

export type PredictResponse = {
  cancer_type: CancerType;
  prediction: ResultClass;
  confidence: number; // 0..1
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
  confidence: number; // 0..1
  localization_url?: string | null;
};

export async function predictScan(params: {
  cancerType: CancerType;
  file: File;
}): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", params.file);

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

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Save case failed (${res.status}): ${msg}`);
  }

  return res.json();
}

export async function fetchCases() {
  const res = await fetch(`${API_BASE}/cases`);
  if (!res.ok) throw new Error(`Fetch cases failed: ${res.status}`);
  return res.json();
}