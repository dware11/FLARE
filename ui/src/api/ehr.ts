import { apiFetch } from './http'
import type { CaseResult } from './types'

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:5000'

export function fetchMyCases(token: string) {
  return apiFetch<CaseResult[]>(`${API_BASE}/api/cases`, token)
}
