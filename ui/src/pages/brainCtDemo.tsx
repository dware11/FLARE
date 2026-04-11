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
import { getCtPatients, runCtPredict, CT_API_BASE } from '../api/ctApi'

function absolutizeCamUrl(path: string | null | undefined): string | null {
  if (path == null || path === '') return null
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  const base = CT_API_BASE.replace(/\/$/, '')
  return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`
}

export default function BrainCtDemo() {
  const [patients, setPatients] = useState<string[]>([])
  const [patientId, setPatientId] = useState('')
  const [npzFile, setNpzFile] = useState<File | null>(null)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingPatients, setLoadingPatients] = useState(false)
  const [error, setError] = useState('')

  const camDisplayUrl = useMemo(() => {
    const raw = result?.cam_url
    if (typeof raw !== 'string' || !raw) return ''
    const u = absolutizeCamUrl(raw)
    if (!u) return ''
    return `${u}?t=${Date.now()}`
  }, [result])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        setError('')
        setLoadingPatients(true)

        const data = await getCtPatients()

        let ids: string[] = []
        if (Array.isArray(data)) {
          ids = data
            .map((x: unknown) =>
              typeof x === 'string' ? x : (x as { patient_id?: string; id?: string })?.patient_id ?? (x as { id?: string })?.id ?? '',
            )
            .filter(Boolean)
        } else if (data && typeof data === 'object' && 'patients' in data) {
          const maybe = (data as { patients: unknown }).patients
          if (Array.isArray(maybe)) {
            ids = maybe.filter((x): x is string => typeof x === 'string')
          }
        }

        if (!cancelled) {
          setPatients(ids)
          if (!patientId && ids[0]) setPatientId(ids[0])
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load patients')
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
    if (!patientId.trim() || !npzFile) return

    setError('')
    setResult(null)
    setLoading(true)

    try {
      const r = await runCtPredict(npzFile, patientId, 'H001')
      setResult(r as Record<string, unknown>)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
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
        Select or enter a patient ID, upload a preprocessed CT volume (.npz), run analysis, and view
        the prediction and Grad-CAM from <code style={{ color: '#9bb1ff' }}>result.cam_url</code>.
      </Typography>

      {error && (
        <Alert
          severity="error"
          sx={{
            mb: 3,
            backgroundColor: 'rgba(255,0,0,0.08)',
            border: '1px solid rgba(255,92,92,0.6)',
            color: '#fff',
          }}
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
        <CardContent sx={{ p: 3, display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start' }}>
          {patients.length > 0 ? (
            <TextField
              select
              label="Patient"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              disabled={loading || loadingPatients}
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
          ) : (
            <TextField
              label="Patient ID"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              disabled={loading || loadingPatients}
              sx={{
                minWidth: 260,
                '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
                '& label': { color: 'rgba(255,255,255,0.65)' },
                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
              }}
            />
          )}

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Button variant="outlined" component="label" disabled={loading} sx={{ color: '#fbbf24', borderColor: 'rgba(251,191,36,0.5)', textTransform: 'none' }}>
              Choose .npz
              <input
                type="file"
                accept=".npz"
                hidden
                onChange={(e) => setNpzFile(e.target.files?.[0] ?? null)}
              />
            </Button>
            {npzFile && (
              <Typography sx={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.85rem', maxWidth: 320, wordBreak: 'break-all' }}>
                {npzFile.name}
              </Typography>
            )}
          </Box>

          <Button
            variant="contained"
            disabled={!patientId.trim() || !npzFile || loading || loadingPatients}
            onClick={onRun}
            sx={{
              backgroundColor: '#ff5c5c',
              textTransform: 'none',
              borderRadius: 2,
              px: 4,
              py: 1.2,
              alignSelf: 'center',
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
              No patients found in manifest — enter a patient ID above.
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
                backgroundColor: 'rgba(0,0,0,0.7)',
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
                : 'No result yet. Upload an NPZ and run analysis.'}
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

            {camDisplayUrl ? (
              <Box
                component="img"
                src={camDisplayUrl}
                alt="Grad-CAM overlay"
                sx={{
                  width: '100%',
                  borderRadius: 2,
                  border: '1px solid rgba(255,255,255,0.12)',
                }}
              />
            ) : (
              <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>
                After a successful run, the Grad-CAM image appears here when <code style={{ color: '#9bb1ff' }}>cam_url</code> is returned.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}
