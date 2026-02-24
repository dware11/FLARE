import { apiFetch } from './http'

export interface OutbreakData {
  region: string
  cases: number
}

export function fetchOutbreaks(token: string) {
  return apiFetch<OutbreakData[]>(
    'http://localhost:3000/api/outbreaks',
    token
  )
}
