import { apiFetch } from './http'

export interface OutbreakData {
  region: string
  cases: number
}

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? 'https://reassign-guiding-grass.ngrok-free.dev'

export function fetchOutbreaks(token: string) {
  return apiFetch<OutbreakData[]>(
    `${API_BASE}/api/outbreak/status`,
    token
  )
}
