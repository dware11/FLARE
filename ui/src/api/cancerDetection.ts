import type { CreateCaseRequest, CaseResult } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000'

export async function createCaseAndRunModel(
  token: string,
  payload: CreateCaseRequest,
  files: File[]
): Promise<CaseResult> {
  const form = new FormData()
  form.append('payload', JSON.stringify(payload))
  files.forEach((f) => form.append('files', f))

  const res = await fetch(`${API_BASE}/api/cases`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: form,
  })

  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `Request failed: ${res.status}`)
  }

  return res.json()
}

