import { Box, CircularProgress } from '@mui/material'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'

export default function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuth0()

  if (isLoading) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: '#0b0f19',
        }}
      >
        <CircularProgress sx={{ color: '#ff5c5c' }} />
      </Box>
    )
  }

  return isAuthenticated ? <Outlet /> : <Navigate to="/" replace />
}

