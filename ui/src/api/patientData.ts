import { apiFetch } from './http'

export interface PatientRecord {
  id: string
  date: string
  result: string
}

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? 'https://reassign-guiding-grass.ngrok-free.dev'

export function fetchPatientHistory(token: string) {
  return apiFetch<PatientRecord[]>(
    `${API_BASE}/api/patients`,
    token
  )
}
