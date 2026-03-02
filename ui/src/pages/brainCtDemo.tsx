import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  MenuItem,
  TextField,
  Typography,
} from '@mui/material'
import { getCtPatients, runCtPredict, getCtCamUrl } from '../api/ctApi'

export default function BrainCtDemo() {
  const [patients, setPatients] = useState<string[]>([])
  const [patientId, setPatientId] = useState('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [loadingPatients, setLoadingPatients] = useState(false)
  const [error, setError] = useState('')

  const camUrl = useMemo(() => {
    if (!patientId) return ''
    // cache-bust so Grad-CAM refreshes when re-running on same patient
    return `${getCtCamUrl(patientId)}?t=${Date.now()}`
  }, [patientId, result])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        setError('')
        setLoadingPatients(true)

        const data = await getCtPatients()

        // Normalize shapes:
        // - { patients: string[] }
        // - string[]
        let ids: string[] = []
        if (Array.isArray(data)) {
          ids = data
            .map((x: any) =>
              typeof x === 'string' ? x : x?.patient_id ?? x?.id ?? '',
            )
            .filter(Boolean)
        } else if (data && typeof data === 'object' && 'patients' in data) {
          const maybe = (data as any).patients
          if (Array.isArray(maybe)) {
            ids = maybe.filter((x) => typeof x === 'string')
          }
        }

        if (!cancelled) {
          setPatients(ids)
          if (!patientId && ids[0]) setPatientId(ids[0])
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.message ?? 'Failed to load patients')
        }
      } finally {
        if (!cancelled) setLoadingPatients(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function onRun() {
    if (!patientId) return

    setError('')
    setResult(null)
    setLoading(true)

    try {
      const r = await runCtPredict(patientId)
      setResult(r)
    } catch (e: any) {
      setError(e?.message ?? 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ px: { xs: 3, md: 10 }, py: 5 }}>
      <Typography
        sx={{ color: '#fff', fontSize: '1.7rem', fontWeight: 800, mb: 1 }}
      >
        Brain CT Demo
      </Typography>
      <Typography sx={{ color: 'rgba(255,255,255,0.65)', mb: 3 }}>
        Select a patient from the processed CT manifest, run analysis, and view
        the prediction and Grad-CAM overlay.
      </Typography>

      {error && (
        <Alert
          severity="error"
          sx={{ mb: 3, backgroundColor: 'rgba(255,255,255,0.05)', color: '#fff' }}
        >
          {error}
        </Alert>
      )}

      <Card
        sx={{
          backgroundColor: 'rgba(0,0,0,0.30)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3,
          color: '#fff',
          mb: 4,
        }}
      >
        <CardContent sx={{ p: 3, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <TextField
            select
            label="Patient"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            disabled={loading || loadingPatients || !patients.length}
            sx={{
              minWidth: 260,
              '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
              '& label': { color: 'rgba(255,255,255,0.65)' },
              '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
            }}
          >
            {patients.map((id) => (
              <MenuItem key={id} value={id}>
                {id}
              </MenuItem>
            ))}
          </TextField>

          <Button
            variant="contained"
            disabled={!patientId || loading || loadingPatients}
            onClick={onRun}
            sx={{
              backgroundColor: '#ff5c5c',
              textTransform: 'none',
              borderRadius: 2,
              px: 4,
              py: 1.2,
              '&:hover': { backgroundColor: '#ff3b3b' },
            }}
          >
            {loading ? 'Running…' : 'Run Analysis'}
          </Button>

          {loadingPatients && (
            <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>
              Loading patients…
            </Typography>
          )}

          {!loadingPatients && !patients.length && !error && (
            <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>
              No patients found in manifest.
            </Typography>
          )}
        </CardContent>
      </Card>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1.1fr 1.1fr' },
          gap: 3,
        }}
      >
        <Card
          sx={{
            backgroundColor: 'rgba(0,0,0,0.30)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 3,
            color: '#fff',
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography
              sx={{ fontWeight: 900, color: '#9bb1ff', mb: 1 }}
            >
              Prediction
            </Typography>
            <Divider
              sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }}
            />
            <Box
              component="pre"
              sx={{
                m: 0,
                fontFamily: 'monospace',
                fontSize: '0.8rem',
                backgroundColor: 'rgba(0,0,0,0.45)',
                borderRadius: 2,
                p: 2,
                maxHeight: 340,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {result
                ? JSON.stringify(result, null, 2)
                : 'No result yet. Run analysis to see CT + fused outputs.'}
            </Box>
          </CardContent>
        </Card>

        <Card
          sx={{
            backgroundColor: 'rgba(0,0,0,0.30)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 3,
            color: '#fff',
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography
              sx={{ fontWeight: 900, color: '#9bb1ff', mb: 1 }}
            >
              Grad-CAM
            </Typography>
            <Divider
              sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }}
            />

            {patientId ? (
              <Box
                component="img"
                src={camUrl}
                alt="Grad-CAM overlay"
                sx={{
                  width: '100%',
                  borderRadius: 2,
                  border: '1px solid rgba(255,255,255,0.12)',
                }}
              />
            ) : (
              <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>
                Select a patient and run analysis to view the Grad-CAM overlay.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}

