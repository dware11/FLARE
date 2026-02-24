import { apiFetch } from './http'

export interface PatientRecord {
  id: string
  date: string
  result: string
}

export function fetchPatientHistory(token: string) {
  return apiFetch<PatientRecord[]>(
    'http://localhost:3000/api/patients',
    token
  )
}
