
import { Box, Button, Card, CardContent, Typography } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import HubOutlinedIcon from '@mui/icons-material/HubOutlined'
import CloudOutlinedIcon from '@mui/icons-material/CloudOutlined'
import LayersOutlinedIcon from '@mui/icons-material/LayersOutlined'
import FlareBackground from '../assets/flare-background.png'

export default function Home() {
  const navigate = useNavigate()

  return (
    <Box
      sx={{
        width: '100vw',
        minHeight: '100vh',
        background:
          'radial-gradient(circle at bottom right, #1b2335 0%, #0b0f19 60%)',
        pb: 10,
      }}
    >
      {/* HERO SECTION */}
      <Box
        sx={{
          minHeight: '85vh',
          px: { xs: 3, md: 10 },
          pt: { xs: 12, md: 14 }, // pushes content below header
          pb: 10,
          backgroundImage: `url(${FlareBackground})`,
          backgroundRepeat: 'no-repeat',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <Typography
          variant="h2"
          sx={{
            fontWeight: 500,
            lineHeight: 1.08,
            color: '#fff',
            mb: 3,
          }}
        >
          Innovative Analysis
          <br />
          in Cancer Detection
        </Typography>

        <Typography
          sx={{
            color: '#cfd3dc',
            fontSize: '1.15rem',
            lineHeight: 1.7,
            mb: 4,
            maxWidth: 520,
          }}
        >
          Helping lives and advancing
          <br />
          MedTech through each scan
        </Typography>

        <Button
          variant="contained"
          onClick={() => navigate('/cancer-detection')}
          sx={{
            backgroundColor: '#ff5c5c',
            px: 2.5,   // was 4 → shorter width
    py: 1.2,   // was 1.5 → slightly shorter height
    fontSize: '0.95rem',
    textTransform: 'none',
    borderRadius: '8px',
    width: 'fit-content',
            '&:hover': { backgroundColor: '#ff3b3b' },
          }}
        >
          Get Started
        </Button>
      </Box>

      {/* SECTION: Obtain Accurate + Feature Cards (side-by-side) */}
      <Box
        sx={{
          px: { xs: 3, md: 10 },
          mt: 7,
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: '1fr 1.6fr' },
          gap: { xs: 4, lg: 8 },
          alignItems: 'stretch',
        }}
      >
        {/* LEFT: Obtain accurate awareness card */}
        <Box
          sx={{
            border: '1px solid rgba(255,255,255,0.10)',
            borderRadius: '18px',
            p: { xs: 4, md: 6 },
            minHeight: { lg: 520 },
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            background:
              'radial-gradient(circle at left center, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 60%, transparent 100%)',
          }}
        >
          <Typography
            sx={{
              color: '#fff',
              fontSize: { xs: '2rem', md: '2.4rem' },
              lineHeight: 1.2,
              fontWeight: 500,
              mb: 3,
            }}
          >
            Obtain accurate
            <br />
            awareness of cancer
            <br />
            and disease outbreaks
          </Typography>

          <Typography
            sx={{
              color: 'rgba(255,255,255,0.55)',
              fontSize: '0.95rem',
              lineHeight: 1.9,
              maxWidth: 540,
            }}
          >
            FLARE is a multi-modal, AI-assisted decision-support tool designed to assist
            clinicians in interpreting medical imaging across multiple types. It is designed
            to mirror how clinicians reason. Rather than analyzing just one scan at a time,
            FLARE is designed to accept multiple imaging studies from the same patient and
            process them through.
          </Typography>
        </Box>

        {/* RIGHT: 3 Feature Cards */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
            gap: 3,
            alignItems: 'stretch',
          }}
        >
          <FeatureCard
            title="Cancer Detection"
            description="Analyze medical images utilizing AI accuracy prediction to detect cancerous lesions."
            icon={<HubOutlinedIcon sx={{ color: '#ff5c5c', fontSize: 36 }} />}
            onClick={() => navigate('/cancer-detection')}
          />

          <FeatureCard
            title="Outbreak Tracker"
            description="Monitor local or regional disease outbreaks with current visual data."
            icon={<CloudOutlinedIcon sx={{ color: '#ff5c5c', fontSize: 36 }} />}
            onClick={() => navigate('/outbreak-tracker')}
          />

          <FeatureCard
            title="EHR Database"
            description="View patient data and FLARE user information."
            icon={<LayersOutlinedIcon sx={{ color: '#ff5c5c', fontSize: 36 }} />}
            onClick={() => navigate('/ehr-database')}
          />
        </Box>
      </Box>
    </Box>
  )
}

function FeatureCard({
  title,
  description,
  icon,
  onClick,
}: {
  title: string
  description: string
  icon: React.ReactNode
  onClick: () => void
}) {
  return (
    <Card
      onClick={onClick}
      sx={{
        height: '100%',
        backgroundColor: 'rgba(0,0,0,0.30)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '18px',
        cursor: 'pointer',
        transition: 'all 0.25s ease',
        boxShadow: '0 12px 30px rgba(0,0,0,0.35)',
        '&:hover': {
          transform: 'translateY(-6px)',
          boxShadow: '0 18px 45px rgba(0,0,0,0.55)',
          borderColor: 'rgba(255,255,255,0.14)',
        },
      }}
    >
      <CardContent sx={{ p: 4 }}>
        {icon}

        <Typography
          sx={{
            mt: 3,
            mb: 1,
            fontWeight: 700,
            color: '#9bb1ff',
            fontSize: '1.05rem',
          }}
        >
          {title}
        </Typography>

        <Typography
          sx={{
            fontSize: '0.9rem',
            color: 'rgba(255,255,255,0.55)',
            lineHeight: 1.7,
            minHeight: 72,
          }}
        >
          {description}
        </Typography>

        <Box
          sx={{
            mt: 3,
            height: '3px',
            width: '100%',
            backgroundColor: 'rgba(255,255,255,0.10)',
            borderRadius: '2px',
          }}
        />
      </CardContent>
    </Card>
  )
}
