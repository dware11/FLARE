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
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { getCtPatients, runCtPredict, CT_API_BASE } from '../api/ctApi'

type UploadMode = 'single' | 'volume'

function absolutizeCamUrl(path: string | null | undefined): string | null {
  if (path == null || path === '') return null
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  const base = CT_API_BASE.replace(/\/$/, '')
  return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`
}

function isVolumeFile(f: File | null): boolean {
  if (!f) return false
  const n = f.name.toLowerCase()
  return n.endsWith('.npz') || n.endsWith('.zip')
}

function isSingleSliceFile(f: File | null): boolean {
  if (!f) return false
  const n = f.name.toLowerCase()
  return n.endsWith('.jpg') || n.endsWith('.jpeg') || n.endsWith('.png')
}

function fileMatchesMode(f: File | null, mode: UploadMode): boolean {
  if (!f) return false
  return mode === 'volume' ? isVolumeFile(f) : isSingleSliceFile(f)
}

function fmtPercent01(n: unknown): string {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—'
  return `${(n * 100).toFixed(2)}%`
}

function strVal(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return String(v)
  return '—'
}

function formatPredictionSummary(r: Record<string, unknown> | null): string {
  if (!r) return '—'
  const rc = r.result_class
  if (typeof rc === 'string' && rc.trim()) return rc
  const pl = r.pred_label
  if (pl === 0 || pl === '0' || pl === 'Normal' || pl === 'normal') return 'Normal'
  if (pl === 1 || pl === '1' || pl === 'Abnormal' || pl === 'abnormal') return 'Abnormal'
  if (pl != null && String(pl) !== '') return String(pl)
  return '—'
}

function ResultRow({ label, value }: { label: string; value: string }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        gap: 1,
        py: 0.75,
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        '&:last-of-type': { borderBottom: 'none' },
      }}
    >
      <Typography sx={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.88rem' }}>{label}</Typography>
      <Typography sx={{ color: '#fff', fontSize: '0.95rem', fontWeight: 600, textAlign: 'right', wordBreak: 'break-word' }}>
        {value}
      </Typography>
    </Box>
  )
}

export default function BrainCtDemo() {
  const [patients, setPatients] = useState<string[]>([])
  const [patientId, setPatientId] = useState('')
  const [uploadMode, setUploadMode] = useState<UploadMode>('volume')
  const [ctFile, setCtFile] = useState<File | null>(null)
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

  const canRun = Boolean(
    patientId.trim() && ctFile && fileMatchesMode(ctFile, uploadMode) && !loading && !loadingPatients
  )

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
    if (!canRun || !ctFile) return

    setError('')
    setResult(null)
    setLoading(true)

    try {
      const r = await runCtPredict(ctFile, patientId, 'H001')
      setResult(r as Record<string, unknown>)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const predSummary = useMemo(() => formatPredictionSummary(result), [result])

  return (
    <Box sx={{ px: { xs: 3, md: 10 }, py: 5 }}>
      <Typography sx={{ color: '#fff', fontSize: '1.7rem', fontWeight: 800, mb: 1 }}>
        Brain CT Demo
      </Typography>
      <Typography sx={{ color: 'rgba(255,255,255,0.65)', mb: 2 }}>
        Standalone CT pipeline preview: same <code style={{ color: '#9bb1ff' }}>POST /api/ct/predict</code> as main app.
        Use <strong>CT volume</strong> for full sequence (NPZ / ZIP) or <strong>single image</strong> for a quick slice test
        (JPG/PNG) — not for full k-slice reporting.
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
          mb: 3,
        }}
      >
        <CardContent sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box>
            <Typography sx={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.85rem', fontWeight: 600, mb: 1 }}>Upload mode</Typography>
            <ToggleButtonGroup
              exclusive
              value={uploadMode}
              onChange={(_, v: UploadMode | null) => {
                if (v == null) return
                setUploadMode(v)
                setCtFile(null)
                setResult(null)
                setError('')
              }}
              disabled={loading || loadingPatients}
              sx={{
                '& .MuiToggleButton-root': {
                  color: 'rgba(255,255,255,0.75)',
                  textTransform: 'none',
                  borderColor: 'rgba(255,255,255,0.2)',
                },
                '& .MuiToggleButton-root.Mui-selected': {
                  backgroundColor: 'rgba(255,92,92,0.22)',
                  color: '#fff',
                },
              }}
            >
              <ToggleButton value="single">Single image</ToggleButton>
              <ToggleButton value="volume">CT volume / DICOM ZIP</ToggleButton>
            </ToggleButtonGroup>
            <Typography sx={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.8rem', mt: 1 }}>
              {uploadMode === 'volume'
                ? 'Recommended: .npz (preprocessed volume) or .zip (DICOM series) for multi-slice inference.'
                : 'Quick test: one JPG/PNG slice. Not a substitute for full volume input.'}
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start' }}>
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
              <Button
                variant="outlined"
                component="label"
                disabled={loading}
                sx={{ color: '#fbbf24', borderColor: 'rgba(251,191,36,0.5)', textTransform: 'none' }}
              >
                {uploadMode === 'volume' ? 'Choose .npz or .zip' : 'Choose JPG/PNG'}
                <input
                  type="file"
                  accept={uploadMode === 'volume' ? '.npz,.zip' : '.jpg,.jpeg,.png'}
                  hidden
                  onChange={(e) => {
                    setCtFile(e.target.files?.[0] ?? null)
                    setResult(null)
                    setError('')
                  }}
                />
              </Button>
              {ctFile && (
                <Typography sx={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.85rem', maxWidth: 360, wordBreak: 'break-all' }}>
                  {ctFile.name}
                </Typography>
              )}
            </Box>

            <Button
              variant="contained"
              disabled={!canRun}
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

            {loadingPatients && <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>Loading patients…</Typography>}

            {!loadingPatients && !patients.length && !error && (
              <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>No patients found in manifest — enter a patient ID above.</Typography>
            )}
          </Box>
        </CardContent>
      </Card>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1.1fr 1.1fr' },
          gap: 3,
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Card
            sx={{
              backgroundColor: 'rgba(0,0,0,0.30)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 3,
              color: '#fff',
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Typography sx={{ fontWeight: 900, color: '#9bb1ff', mb: 1 }}>CT results</Typography>
              <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }} />

              {!result && (
                <Typography sx={{ color: 'rgba(255,255,255,0.6)' }}>No result yet. Choose files and run analysis.</Typography>
              )}

              {result && (
                <Box>
                  {typeof result.error === 'string' && result.error && (
                    <Alert severity="warning" sx={{ mb: 2, color: '#fff' }}>
                      {result.error}
                    </Alert>
                  )}
                  <ResultRow
                    label="Prediction"
                    value={predSummary}
                  />
                  <ResultRow
                    label="Confidence"
                    value={typeof result.confidence === 'number' && Number.isFinite(result.confidence) ? fmtPercent01(result.confidence) : strVal(result.confidence)}
                  />
                  <ResultRow label="p_normal" value={fmtPercent01(result.p_normal)} />
                  <ResultRow label="p_abnormal" value={fmtPercent01(result.p_abnormal)} />
                  <ResultRow label="input_format" value={strVal(result.input_format)} />
                  <ResultRow
                    label="cam_url"
                    value={typeof result.cam_url === 'string' && result.cam_url ? result.cam_url : '—'}
                  />
                </Box>
              )}
            </CardContent>
          </Card>

          {result && Object.keys(result).length > 0 && (
            <Box component="details" sx={{ color: 'rgba(255,255,255,0.75)' }}>
              <Box
                component="summary"
                sx={{ cursor: 'pointer', color: '#9bb1ff', fontSize: '0.9rem', fontWeight: 600, mb: 1, userSelect: 'none' }}
              >
                Raw API response (JSON)
              </Box>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  backgroundColor: 'rgba(0,0,0,0.5)',
                  borderRadius: 2,
                  p: 2,
                  maxHeight: 280,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {JSON.stringify(result, null, 2)}
              </Box>
            </Box>
          )}
        </Box>

        <Card
          sx={{
            backgroundColor: 'rgba(0,0,0,0.30)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 3,
            color: '#fff',
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 900, color: '#9bb1ff', mb: 1 }}>Grad-CAM (cam_url)</Typography>
            <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }} />

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
                After a successful run, the heatmap appears here when the API returns <code style={{ color: '#9bb1ff' }}>cam_url</code>.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}
