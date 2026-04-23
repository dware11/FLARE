
import { Box, Button, Link, Typography } from '@mui/material'
import { useAuth0 } from '@auth0/auth0-react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import FlareLogo from '../assets/FLARElogo.png'

export default function Login() {
  const { loginWithRedirect, isAuthenticated } = useAuth0()
  const navigate = useNavigate()

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/home')
    }
  }, [isAuthenticated])

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background:
          'radial-gradient(circle at bottom right, #18171c 5%, #17181c 45%, #15171b 75%, #0b0f19 100%)',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        
        <Box
          component="img"
          src={FlareLogo}
          alt="FLARE logo"
          sx={{
            height: 250,
          
            mb: 5,
          }}
        />

        
    <Button 
    onClick={() => {
      loginWithRedirect()
    }}
    sx={{
            backgroundColor: '#ff5c5c',
            color: '#ffffff',
            px: 6,
            py: 1.5,
            fontSize: '1.1rem',
            textTransform: 'none',
            borderRadius: '8px',
            '&:hover': {
              backgroundColor: '#ffffff',
              color: '#000000'
            },
          }}>
      Log In
    </Button>
    <Typography
      sx={{
        mt: 2.5,
        color: 'rgba(255,255,255,0.72)',
        fontSize: '0.85rem',
      }}
    >
      By signing in you agree to our{' '}
      <Link
        component="button"
        type="button"
        onClick={() => navigate('/privacy')}
        sx={{
          color: '#ff5c5c',
          fontSize: '0.85rem',
          textDecorationColor: 'rgba(255,92,92,0.75)',
          '&:hover': {
            color: '#ff8080',
            textDecorationColor: '#ff8080',
          },
        }}
      >
        Privacy Policy
      </Link>
    </Typography>
    </Box>
    </Box>
  )
}
