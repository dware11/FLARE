import { Alert, Box, Button, Card, CardContent, Typography } from '@mui/material'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { useNavigate } from 'react-router-dom'

type SectionProps = {
  title: string
  body: string
}

function PrivacySection({ title, body }: SectionProps) {
  return (
    <Card
      sx={{
        backgroundColor: 'rgba(0,0,0,0.30)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '18px',
        boxShadow: '0 12px 30px rgba(0,0,0,0.35)',
      }}
    >
      <CardContent sx={{ p: { xs: 2.5, md: 3.5 } }}>
        <Typography
          sx={{
            color: '#ff5c5c',
            fontSize: { xs: '1.05rem', md: '1.2rem' },
            fontWeight: 700,
            mb: 1.25,
          }}
        >
          {title}
        </Typography>
        <Typography
          sx={{
            color: 'rgba(255,255,255,0.88)',
            fontSize: '0.96rem',
            lineHeight: 1.8,
          }}
        >
          {body}
        </Typography>
      </CardContent>
    </Card>
  )
}

export default function Privacy() {
  const navigate = useNavigate()

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100%',
        background: 'radial-gradient(circle at bottom right, #1b2335 0%, #0b0f19 60%)',
        px: { xs: 2, md: 8 },
        py: { xs: 5, md: 8 },
      }}
    >
      <Box sx={{ maxWidth: 980, mx: 'auto' }}>
        <Typography
          variant="h4"
          sx={{
            color: '#fff',
            fontWeight: 700,
            letterSpacing: '0.03em',
            mb: 2.5,
          }}
        >
          Privacy Policy
        </Typography>

        <Alert
          severity="warning"
          sx={{
            mb: 3,
            borderRadius: '12px',
            color: '#fff',
            backgroundColor: 'rgba(255,92,92,0.16)',
            border: '1px solid rgba(255,92,92,0.65)',
            '& .MuiAlert-icon': {
              color: '#ff5c5c',
            },
          }}
        >
          By using FLARE, you acknowledge and agree to the terms described on this page.
          This includes understanding that AI predictions are decision-support only and
          require qualified clinical review.
        </Alert>

        <Box sx={{ display: 'grid', gap: 2.2 }}>
          <PrivacySection
            title="Data Collection & Uploads"
            body="When a user uploads a medical scan to FLARE, the file is handled in memory for model inference and is not stored as a persistent upload artifact by the application. In a production deployment, all uploaded imaging files would undergo automated metadata and DICOM tag stripping prior to inference to prevent inadvertent transmission of embedded patient identifiers."
          />
          <PrivacySection
            title="Prediction Data & Identity"
            body="AI prediction outputs are not stored with patient identity data. Any model-generated result data is kept separate from direct patient identifiers."
          />
          <PrivacySection
            title="Audit Logging"
            body="FLARE records operational audit events such as API activity and prediction metadata needed for system monitoring and debugging. FLARE does not log filenames, protected health information (PHI), or direct patient identifiers."
          />
          <PrivacySection
            title="Authentication (Auth0)"
            body="Authentication is provided through Auth0. This means user login credentials are handled by Auth0's authentication flow rather than stored directly by FLARE."
          />
          <PrivacySection
            title="Research Demo Disclaimer"
            body="FLARE is a research demonstration platform and is not represented as a covered HIPAA entity."
          />
          <PrivacySection
            title="Clinical Safety — AI Is Not A Substitute For Clinical Judgment"
            body="Model outputs are decision-support only. All AI predictions must be reviewed and validated by a qualified radiologist or clinician before clinical use."
          />
          <PrivacySection
            title="Contact"
            body="For privacy, terms, or data-handling questions, contact: d.ware9873@gmail.com"
          />
        </Box>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(-1)}
          sx={{
            mt: 2.5,
            color: '#ff5c5c',
            borderColor: 'rgba(255,92,92,0.7)',
            textTransform: 'none',
            borderRadius: '10px',
            '&:hover': {
              borderColor: '#ff5c5c',
              backgroundColor: 'rgba(255,92,92,0.10)',
            },
          }}
          variant="outlined"
        >
          Back
        </Button>
      </Box>
    </Box>
  )
}
