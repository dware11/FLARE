export type UserRole = 'medical' | 'admin'

export interface AuthUser {
  id: string
  email: string
  role: UserRole
  token: string
}
