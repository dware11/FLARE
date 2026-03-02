const CT_API_BASE =
  import.meta.env.VITE_CT_API_BASE ?? 'http://localhost:5000';

console.log('CT API BASE:', CT_API_BASE);

export interface CtPatientsResponse {
  patients: string[];
}

export async function getCtPatients(): Promise<CtPatientsResponse | string[]> {
  const res = await fetch(`${CT_API_BASE}/api/patients`);
  if (!res.ok) {
    throw new Error(`GET /api/patients failed (${res.status})`);
  }
  return res.json();
}

export interface CtPredictRequest {
  patient_id: string;
}

export interface CtPredictResponse {
  patient_id: string;
  ct: unknown;
  fused: unknown;
}

export async function runCtPredict(
  patient_id: string,
): Promise<CtPredictResponse> {
  const res = await fetch(`${CT_API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id } satisfies CtPredictRequest),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`POST /api/predict failed (${res.status}) ${text}`);
  }

  return res.json();
}

export function getCtCamUrl(patient_id: string): string {
  return `${CT_API_BASE}/api/cam/${encodeURIComponent(patient_id)}.png`;
}

