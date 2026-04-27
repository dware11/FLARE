import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import {
  AppBar,
  Toolbar,
  Box,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  Dialog,
  DialogContent,
  Button,
} from '@mui/material'
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown'
import CloseIcon from '@mui/icons-material/Close'
import RefreshIcon from '@mui/icons-material/Refresh'

export default function Header() {
  const navigate = useNavigate()
  const { user, logout } = useAuth0()

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [logoutOpen, setLogoutOpen] = useState(false)

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => setAnchorEl(null)

  const go = (path: string) => {
    handleMenuClose()
    navigate(path)
  }

  const openLogoutDialog = () => {
    handleMenuClose()
    setLogoutOpen(true)
  }

  const closeLogoutDialog = () => {
    setLogoutOpen(false)
  }

  const refreshApp = () => {
    handleMenuClose()
    try {
      sessionStorage.removeItem('flare_last_result')
      sessionStorage.removeItem('flare_form_state')
    } catch { /* ignore */ }
    window.dispatchEvent(new CustomEvent('flare:refresh-app'))
  }

  const confirmLogout = () => {
    setLogoutOpen(false)
    try {
      sessionStorage.removeItem('flare_last_result')
      sessionStorage.removeItem('flare_form_state')
    } catch { /* ignore */ }
    logout({ logoutParams: { returnTo: window.location.origin } })
  }

  return (
    <>
      {/* ================= HEADER ================= */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          backgroundColor: '#000000',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid #000000',
        }}
      >
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
          {/* LEFT: LOGO */}
          <Box sx={{ cursor: 'pointer' }} onClick={() => navigate('/home')}>
            <Typography sx={{ fontWeight: 700, letterSpacing: '0.12em' }}>
              FLARE
            </Typography>
          </Box>

          {/* RIGHT: AVATAR (NOT CLICKABLE) + DROPDOWN */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Avatar
              src={user?.picture}
              alt={user?.name ?? 'Profile'}
              sx={{ width: 34, height: 34, cursor: 'default' }}
            />

            <IconButton onClick={handleMenuOpen} sx={{ color: '#fff' }} size="small">
              <KeyboardArrowDownIcon />
            </IconButton>

            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              PaperProps={{
                sx: {
                  mt: 1.5,
                  backgroundColor: '#0f1117',
                  color: '#fff',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 2,
                  minWidth: 220,
                },
              }}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            >
              <MenuItem onClick={() => go('/home')}>
                Home
              </MenuItem>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)' }} />
              <MenuItem onClick={() => go('/cancer-detection')}>
                AI Cancer Detection
              </MenuItem>

              <MenuItem onClick={() => go('/geo-tracker')}>
                Geo Tracker
              </MenuItem>

              <MenuItem onClick={() => go('/outbreak-tracker')}>
                Outbreak Analytics
              </MenuItem>

              <MenuItem onClick={() => go('/ehr-database')}>
                EHR Database
              </MenuItem>

              <MenuItem onClick={() => go('/brain-ct-demo')}>
                Brain CT Demo
              </MenuItem>

              <Box sx={{ display: 'flex', justifyContent: 'center', py: 0.5 }}>
                <IconButton
                  onClick={refreshApp}
                  size="small"
                  sx={{ color: '#fff' }}
                  aria-label="Refresh app"
                >
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Box>

              <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)' }} />

              {/* 🔴 LOG OUT OPENS MODAL */}
              <MenuItem
                onClick={openLogoutDialog}
                sx={{ color: '#ff5c5c' }}
              >
                Log Out
              </MenuItem>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)' }} />
              <MenuItem onClick={() => go('/privacy')}>
                Privacy & Terms
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* ================= LOGOUT CONFIRM MODAL ================= */}
      <Dialog
        open={logoutOpen}
        onClose={closeLogoutDialog}
        maxWidth="sm"
        PaperProps={{
          sx: {
            backgroundColor: '#0f1117',
            color: '#fff',
            borderRadius: '14px',
            border: '1px solid rgba(255,255,255,0.1)',
            px: 2,
            py: 3,
            minWidth: 420,
          },
        }}
      >
        <DialogContent>
          {/* Close Icon */}
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <IconButton onClick={closeLogoutDialog} sx={{ color: '#fff' }}>
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Logo / Title */}
          <Box sx={{ textAlign: 'center', mb: 2 }}>
            <Typography sx={{ fontWeight: 700, letterSpacing: '0.15em', mb: 1 }}>
              FLARE
            </Typography>
            <Divider sx={{ borderColor: 'rgba(255,92,92,0.6)', mb: 3 }} />
          </Box>

          {/* Message */}
          <Typography
            sx={{
              textAlign: 'center',
              fontSize: '1.2rem',
              mb: 4,
              color: 'rgba(255,255,255,0.9)',
            }}
          >
            Are you sure you want to log out?
          </Typography>

          {/* Actions */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Button
              variant="contained"
              onClick={confirmLogout}
              sx={{
                backgroundColor: '#ff5c5c',
                py: 1.4,
                fontSize: '1rem',
                textTransform: 'none',
                borderRadius: '10px',
                '&:hover': { backgroundColor: '#ff3b3b' },
              }}
            >
              Log Out
            </Button>

            <Button
              onClick={closeLogoutDialog}
              sx={{
                color: '#fff',
                textTransform: 'none',
                fontSize: '1rem',
                opacity: 0.85,
              }}
            >
              Cancel
            </Button>
          </Box>
        </DialogContent>
      </Dialog>
    </>
  )
}
