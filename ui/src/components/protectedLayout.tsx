import Header from './header'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
}

export default function ProtectedLayout({ children }: Props) {
  return (
    <>
      <Header />
      <main style={{ padding: 16 }}>
        {children}
      </main>
    </>
  )
}
