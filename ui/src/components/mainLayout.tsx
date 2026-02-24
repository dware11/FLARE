 import { Outlet } from 'react-router-dom'
import { Box, Toolbar } from '@mui/material'
import Header from '../components/header'

export default function MainLayout() {
  return (
    <Box sx={{ minHeight: '100vh' }}>
      <Header />
      {/* This Toolbar acts as a spacer equal to AppBar height */}
      <Toolbar />
      <Outlet />
    </Box>
  )
}
