

//central place for API HTTP requests
export async function apiFetch<T>(
  url: string,
  token: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`, //sends secure JWT token
      ...(options.headers || {})
    }
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(
      `API Error ${response.status}: ${errorText}`
    )
  }

  return response.json() as Promise<T>
}
