export const CT_API_BASE =
  import.meta.env.VITE_CT_API_BASE ?? 'https://reassign-guiding-grass.ngrok-free.dev';
const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' } as const;

console.log('CT API BASE:', CT_API_BASE);

export interface CtPatientsResponse {
  patients: string[];
}

export async function getCtPatients(): Promise<CtPatientsResponse | string[]> {
  const res = await fetch(`${CT_API_BASE}/api/patients`, { headers: NGROK_HEADERS });
  if (!res.ok) {
    throw new Error(`GET /api/patients failed (${res.status})`);
  }
  return res.json();
}

export interface CtPredictResponse {
  patient_id?: string;
  modality?: string;
  pred_label?: string;
  result_class?: string;
  confidence?: number;
  p_normal?: number;
  p_abnormal?: number;
  cam_url?: string | null;
  hospitalId?: string;
  hospitalName?: string;
  error?: string;
}

export async function runCtPredict(
  file: File,
  patient_id: string,
  hospitalId: string = 'H001',
): Promise<CtPredictResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('patient_id', patient_id);
  form.append('hospitalId', hospitalId);

  const res = await fetch(`${CT_API_BASE}/api/ct/predict`, {
    method: 'POST',
    headers: NGROK_HEADERS,
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`POST /api/ct/predict failed (${res.status}) ${text}`);
  }

  return res.json();
}

export function getCtCamUrl(patient_id: string): string {
  return `${CT_API_BASE}/api/cam/${encodeURIComponent(patient_id)}.png`;
}

