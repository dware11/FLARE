import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Typography,
  TextField,
  InputAdornment,
  MenuItem,
  Chip,
  Button,
  Divider,
  Drawer,
  IconButton,
  Card,
  CardContent,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material'
import { fetchEhrRecords, approveReview, rejectReview, type EhrRecord } from '../api/flareAPI'
import SearchIcon from '@mui/icons-material/Search'
import CloseIcon from '@mui/icons-material/Close'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import DownloadIcon from '@mui/icons-material/Download'
import VisibilityIcon from '@mui/icons-material/Visibility'

type BrainCancerLabel = 'Glioma' | 'Meningioma' | 'Pituitary' | 'Normal'
type ResultClass = 'Normal' | 'Benign' | 'Malignant'
type ScanModality = 'MRI' | 'CT' | 'Fusion' | 'Mammography' | 'Ultrasound'

type PatientRecord = {
  id: string
  firstName: string
  lastName: string
  dob: string
  medicalId: string
  location: string
  cancerType: BrainCancerLabel
  modality: ScanModality
  scanDate: string
  aiResult: ResultClass
  confidence: number // 0-100
  reviewStatus: string
  notes?: string
  gradCamUrl?: string
  originalImageUrl?: string
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'https://reassign-guiding-grass.ngrok-free.dev'

function toAbsoluteUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined
  const t = url.trim()
  if (t.startsWith('http://') || t.startsWith('https://')) return t
  const base = API_BASE.replace(/\/$/, '')
  const path = t.startsWith('/') ? t : `/${t}`
  return `${base}${path}`
}

function toModality(modality: string): ScanModality {
  const x = (modality || '').toLowerCase()
  if (x.includes('mamm')) return 'Mammography'
  if (x.includes('ultra')) return 'Ultrasound'
  if (x.includes('fusion')) return 'Fusion'
  if (x.includes('ct')) return 'CT'
  return 'MRI'
}

function toResultClass(result_class: string): ResultClass {
  const u = (result_class || '').toLowerCase()
  if (u === 'malignant') return 'Malignant'
  if (u === 'benign') return 'Benign'
  return 'Normal'
}

function mapCancerLabel(raw: string | null | undefined): BrainCancerLabel {
  const u = (raw || '').toLowerCase()
  if (u === 'glioma') return 'Glioma'
  if (u === 'meningioma') return 'Meningioma'
  if (u === 'pituitary') return 'Pituitary'
  return 'Normal'
}

function mapEhrToPatientRecord(r: EhrRecord): PatientRecord {
  const created = r.createdAt?.split('T')[0] ?? ''
  const overlay = r.segmentation?.overlay_url
  const gradCamRaw = overlay ?? r.gradcam_url ?? undefined
  const st = (r.review_status || '').toLowerCase()
  return {
    id: r.caseId,
    firstName: r.firstName ?? '',
    lastName: r.lastName ?? '',
    dob: r.dob ?? '',
    medicalId: (r.medicalId ?? r.patient_id) || '',
    location: r.hospitalName ?? '',
    cancerType: mapCancerLabel(r.cancer_type),
    modality: toModality(r.modality),
    scanDate: created,
    aiResult: toResultClass(r.result_class),
    confidence: Math.round(Number(r.confidence) * 100),
    reviewStatus: st,
    notes:
      r.reject_reason ??
      (st === 'approved'
        ? 'Approved by reviewer.'
        : st === 'rejected'
          ? 'Rejected by reviewer.'
          : undefined),
    gradCamUrl: toAbsoluteUrl(gradCamRaw),
    originalImageUrl: toAbsoluteUrl(r.input_image_url),
  }
}

function resultChipColor(result: ResultClass) {
  switch (result) {
    case 'Normal':
      return { bg: 'rgba(34,197,94,0.16)', border: 'rgba(34,197,94,0.35)', text: '#86efac' }
    case 'Benign':
      return { bg: 'rgba(59,130,246,0.16)', border: 'rgba(59,130,246,0.35)', text: '#93c5fd' }
    case 'Malignant':
      return { bg: 'rgba(239,68,68,0.16)', border: 'rgba(239,68,68,0.35)', text: '#fca5a5' }
  }
}

const fieldSx = {
  '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
  '& label': { color: 'rgba(255,255,255,0.65)' },
  '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
}

export default function EhrDatabase() {
  const navigate = useNavigate()
  const [records, setRecords] = useState<PatientRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState('')

  const [query, setQuery] = useState('')
  const [cancerFilter, setCancerFilter] = useState<BrainCancerLabel | 'All'>('All')
  const [resultFilter, setResultFilter] = useState<ResultClass | 'All'>('All')
  const [selected, setSelected] = useState<PatientRecord | null>(null)

  const [signOpen, setSignOpen] = useState(false)
  const [signDialogMode, setSignDialogMode] = useState<'approve' | 'reject'>('approve')
  const [reviewerName, setReviewerName] = useState('')
  const [digitalSignature, setDigitalSignature] = useState('')
  const [signModalError, setSignModalError] = useState('')
  const [reviewActing, setReviewActing] = useState(false)
  const [drawerReviewMsg, setDrawerReviewMsg] = useState('')
  const [drawerReviewErr, setDrawerReviewErr] = useState('')

  const loadRecords = useCallback(async () => {
    setFetchError('')
    setLoading(true)
    try {
      const { records: rows } = await fetchEhrRecords()
      const mapped = (rows ?? []).map(mapEhrToPatientRecord)
      setRecords(mapped)
      setSelected((prev) => {
        if (!prev) return null
        return mapped.find((x) => x.id === prev.id) ?? prev
      })
    } catch (e: unknown) {
      setFetchError(e instanceof Error ? e.message : 'Failed to load EHR records.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadRecords()
  }, [loadRecords])

  useEffect(() => {
    setDrawerReviewMsg('')
    setDrawerReviewErr('')
  }, [selected?.id])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()

    return records.filter((r) => {
      const matchesQuery =
        !q ||
        `${r.firstName} ${r.lastName}`.toLowerCase().includes(q) ||
        r.medicalId.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)

      const matchesCancer = cancerFilter === 'All' ? true : r.cancerType === cancerFilter
      const matchesResult = resultFilter === 'All' ? true : r.aiResult === resultFilter

      return matchesQuery && matchesCancer && matchesResult
    })
  }, [records, query, cancerFilter, resultFilter])

  const canActOnSelected = Boolean(selected?.id && selected.reviewStatus === 'pending')

  function openApproveModal() {
    if (!selected?.id || selected.reviewStatus !== 'pending') return
    setSignDialogMode('approve')
    setReviewerName('')
    setDigitalSignature('')
    setSignModalError('')
    setSignOpen(true)
  }

  function openRejectModal() {
    if (!selected?.id || selected.reviewStatus !== 'pending') return
    setSignDialogMode('reject')
    setReviewerName('')
    setDigitalSignature('')
    setSignModalError('')
    setSignOpen(true)
  }

  async function confirmSignDialog() {
    if (!selected?.id) return
    const name = reviewerName.trim()
    const sig = digitalSignature.trim()
    if (!name || !sig) {
      setSignModalError('Reviewer name and digital signature are required.')
      return
    }
    setSignModalError('')
    setReviewActing(true)
    setDrawerReviewErr('')
    setDrawerReviewMsg('')
    try {
      if (signDialogMode === 'approve') {
        await approveReview(selected.id, { reviewerName: name, signature: sig })
        setDrawerReviewMsg('Case approved.')
      } else {
        await rejectReview(selected.id, { reviewerName: name, signature: sig })
        setDrawerReviewMsg('Case rejected.')
      }
      setSignOpen(false)
      await loadRecords()
      try {
        window.dispatchEvent(new Event('flare:refresh-app'))
      } catch {
        /* ignore */
      }
    } catch (e: unknown) {
      setDrawerReviewErr(
        e instanceof Error
          ? e.message
          : signDialogMode === 'approve'
            ? 'Approve failed.'
            : 'Reject failed.'
      )
    } finally {
      setReviewActing(false)
    }
  }

  return (
    <Box
      sx={{
        width: '100vw',
        minHeight: '100vh',
        px: { xs: 2, md: 6 },
        py: 4,
        color: '#fff',
        background: 'radial-gradient(circle at bottom right, #1b2335 0%, #0b0f19 60%)',
      }}
    >
      <Dialog
        open={signOpen}
        onClose={() => !reviewActing && setSignOpen(false)}
        PaperProps={{
          sx: {
            backgroundColor: '#0f1117',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 2,
            minWidth: 360,
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 800 }}>
          {signDialogMode === 'reject' ? 'Reject case' : 'Approve case'}
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
          {signModalError && (
            <Alert severity="error" sx={{ backgroundColor: 'rgba(239,68,68,0.15)', color: '#fff' }}>
              {signModalError}
            </Alert>
          )}
          <TextField
            label="Reviewer Name"
            required
            value={reviewerName}
            onChange={(e) => setReviewerName(e.target.value)}
            fullWidth
            sx={fieldSx}
          />
          <TextField
            label="Digital Signature — type your full name to confirm"
            required
            value={digitalSignature}
            onChange={(e) => setDigitalSignature(e.target.value)}
            fullWidth
            sx={fieldSx}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={() => setSignOpen(false)}
            disabled={reviewActing}
            sx={{ color: 'rgba(255,255,255,0.7)' }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={reviewActing}
            onClick={() => void confirmSignDialog()}
            sx={{
              backgroundColor: '#ff5c5c',
              textTransform: 'none',
              '&:hover': { backgroundColor: '#ff3b3b' },
            }}
          >
            {signDialogMode === 'reject' ? 'Confirm Rejection' : 'Confirm Approval'}
          </Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2, mb: 3 }}>
        <Box>
          <Typography sx={{ fontSize: '1.7rem', fontWeight: 800, letterSpacing: '0.02em' }}>
            EHR Database
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.65)', mt: 0.5, maxWidth: 640, lineHeight: 1.6 }}>
            Patient scan history and AI results. Pending screening cases are approved or rejected here
            (authoritative for this demo).
          </Typography>
        </Box>

        <Stack direction="row" spacing={1.5}>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            sx={{
              borderColor: 'rgba(255,255,255,0.18)',
              color: '#fff',
              textTransform: 'none',
              borderRadius: 2,
              '&:hover': { borderColor: 'rgba(255,255,255,0.35)' },
            }}
            onClick={() => alert('Mock export: connect to backend / CSV export later')}
          >
            Export
          </Button>

          <Button
            variant="contained"
            startIcon={<UploadFileIcon />}
            sx={{
              backgroundColor: '#ff5c5c',
              textTransform: 'none',
              borderRadius: 2,
              '&:hover': { backgroundColor: '#ff3b3b' },
            }}
            onClick={() => navigate('/cancer-detection')}
          >
            New Scan
          </Button>
        </Stack>
      </Box>

      <Card
        sx={{
          mb: 3,
          backgroundColor: 'rgba(0,0,0,0.30)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3,
          boxShadow: '0 12px 30px rgba(0,0,0,0.35)',
        }}
      >
        <CardContent sx={{ p: 2.5 }}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '1.4fr 0.8fr 0.8fr' },
              gap: 2,
              alignItems: 'center',
            }}
          >
            <TextField
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by patient name, medical ID, or case ID"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ color: 'rgba(255,255,255,0.55)' }} />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiInputBase-root': {
                  color: '#fff',
                  backgroundColor: 'rgba(255,255,255,0.04)',
                  borderRadius: 2,
                },
                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
              }}
            />

            <TextField
              select
              label="Cancer Type"
              value={cancerFilter}
              onChange={(e) => setCancerFilter(e.target.value as BrainCancerLabel | 'All')}
              sx={{
                '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
                '& label': { color: 'rgba(255,255,255,0.65)' },
                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
              }}
            >
              <MenuItem value="All">All</MenuItem>
              <MenuItem value="Glioma">Glioma</MenuItem>
              <MenuItem value="Meningioma">Meningioma</MenuItem>
              <MenuItem value="Pituitary">Pituitary</MenuItem>
              <MenuItem value="Normal">Normal</MenuItem>
            </TextField>

            <TextField
              select
              label="AI Result"
              value={resultFilter}
              onChange={(e) => setResultFilter(e.target.value as ResultClass | 'All')}
              sx={{
                '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
                '& label': { color: 'rgba(255,255,255,0.65)' },
                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
              }}
            >
              <MenuItem value="All">All</MenuItem>
              <MenuItem value="Normal">Normal</MenuItem>
              <MenuItem value="Benign">Benign</MenuItem>
              <MenuItem value="Malignant">Malignant</MenuItem>
            </TextField>
          </Box>
        </CardContent>
      </Card>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress sx={{ color: '#ff5c5c' }} />
        </Box>
      )}

      {fetchError && (
        <Alert severity="error" sx={{ mb: 2, backgroundColor: 'rgba(239,68,68,0.12)', color: '#fff' }}>
          {fetchError}
        </Alert>
      )}

      <TableContainer
        component={Paper}
        sx={{
          backgroundColor: 'rgba(0,0,0,0.30)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <Table sx={{ opacity: loading ? 0.4 : 1, pointerEvents: loading ? 'none' : 'auto' }}>
          <TableHead>
            <TableRow sx={{ backgroundColor: 'rgba(255,255,255,0.03)' }}>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Patient</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Medical ID</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Cancer Type</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Modality</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Scan Date</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>AI Result</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }}>Confidence</TableCell>
              <TableCell sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700 }} align="right">
                Action
              </TableCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {filtered.map((r) => {
              const chip = resultChipColor(r.aiResult)
              return (
                <TableRow
                  key={r.id}
                  hover
                  onClick={() => setSelected(r)}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.03)' },
                  }}
                >
                  <TableCell sx={{ color: '#fff' }}>
                    <Typography sx={{ fontWeight: 700 }}>
                      {r.firstName} {r.lastName}
                    </Typography>
                    <Typography sx={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.85rem' }}>
                      DOB: {r.dob} • {r.location}
                    </Typography>
                  </TableCell>

                  <TableCell sx={{ color: 'rgba(255,255,255,0.85)' }}>{r.medicalId}</TableCell>
                  <TableCell sx={{ color: 'rgba(255,255,255,0.85)' }}>{r.cancerType}</TableCell>
                  <TableCell sx={{ color: 'rgba(255,255,255,0.85)' }}>{r.modality}</TableCell>
                  <TableCell sx={{ color: 'rgba(255,255,255,0.85)' }}>{r.scanDate}</TableCell>

                  <TableCell>
                    <Chip
                      label={r.aiResult}
                      sx={{
                        backgroundColor: chip.bg,
                        border: `1px solid ${chip.border}`,
                        color: chip.text,
                        fontWeight: 700,
                      }}
                      size="small"
                    />
                  </TableCell>

                  <TableCell sx={{ color: 'rgba(255,255,255,0.85)' }}>{r.confidence}%</TableCell>

                  <TableCell align="right">
                    <Button
                      size="small"
                      startIcon={<VisibilityIcon />}
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelected(r)
                      }}
                      sx={{ textTransform: 'none', color: '#fff' }}
                    >
                      View
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}

            {filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} sx={{ color: 'rgba(255,255,255,0.65)', py: 5, textAlign: 'center' }}>
                  No matching records found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Drawer
        anchor="right"
        open={Boolean(selected)}
        onClose={() => setSelected(null)}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 440 },
            backgroundColor: '#0f1117',
            color: '#fff',
            borderLeft: '1px solid rgba(255,255,255,0.08)',
          },
        }}
      >
        {selected && (
          <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 1 }}>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: '1.25rem' }}>
                  {selected.firstName} {selected.lastName}
                </Typography>
                <Typography sx={{ color: 'rgba(255,255,255,0.65)' }}>
                  Case ID: {selected.id} • Medical ID: {selected.medicalId}
                </Typography>
              </Box>

              <IconButton onClick={() => setSelected(null)} sx={{ color: '#fff' }}>
                <CloseIcon />
              </IconButton>
            </Box>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            <Section title="Patient Information">
              <InfoRow label="Date of Birth" value={selected.dob} />
              <InfoRow label="Location" value={selected.location} />
            </Section>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            <Section title="Scan Details">
              <InfoRow label="Cancer Type" value={selected.cancerType} />
              <InfoRow label="Modality" value={selected.modality} />
              <InfoRow label="Scan Date" value={selected.scanDate} />
              <InfoRow
                label="Review status"
                value={
                  selected.reviewStatus === 'pending'
                    ? 'Pending'
                    : selected.reviewStatus === 'approved'
                      ? 'Approved'
                      : selected.reviewStatus === 'rejected'
                        ? 'Rejected'
                        : selected.reviewStatus || '—'
                }
              />
            </Section>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            <Section title="AI Diagnostic Output">
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>Classification</Typography>
                <Chip
                  label={selected.aiResult}
                  size="small"
                  sx={{
                    ...(() => {
                      const chip = resultChipColor(selected.aiResult)
                      return {
                        backgroundColor: chip.bg,
                        border: `1px solid ${chip.border}`,
                        color: chip.text,
                        fontWeight: 800,
                      }
                    })(),
                  }}
                />
              </Box>

              <InfoRow label="Confidence" value={`${selected.confidence}%`} />

              <Typography sx={{ mt: 2, color: 'rgba(255,255,255,0.65)', fontSize: '0.92rem' }}>
                Notes
              </Typography>
              <Typography sx={{ color: 'rgba(255,255,255,0.85)', mt: 0.6, lineHeight: 1.6 }}>
                {selected.notes ?? 'No notes available.'}
              </Typography>

              {drawerReviewErr && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {drawerReviewErr}
                </Alert>
              )}
              {drawerReviewMsg && (
                <Alert
                  severity="success"
                  sx={{ mt: 2, backgroundColor: 'rgba(34,197,94,0.18)', color: '#fff' }}
                >
                  {drawerReviewMsg}
                </Alert>
              )}

              <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
                <Button
                  variant="outlined"
                  startIcon={<VisibilityIcon />}
                  sx={{
                    borderColor: 'rgba(255,255,255,0.18)',
                    color: '#fff',
                    textTransform: 'none',
                    borderRadius: 2,
                  }}
                  disabled={!selected.originalImageUrl}
                  onClick={() => {
                    const u = selected.originalImageUrl
                    if (u) window.open(u, '_blank', 'noopener,noreferrer')
                  }}
                >
                  View Scan
                </Button>

                <Button
                  variant="contained"
                  sx={{
                    backgroundColor: '#ff5c5c',
                    textTransform: 'none',
                    borderRadius: 2,
                    '&:hover': { backgroundColor: '#ff3b3b' },
                  }}
                  disabled={!selected.gradCamUrl}
                  onClick={() => {
                    const u = selected.gradCamUrl
                    if (u) window.open(u, '_blank', 'noopener,noreferrer')
                  }}
                >
                  View Localization
                </Button>
              </Stack>
            </Section>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            <Section title="Clinician review">
              {!canActOnSelected && (
                <Typography sx={{ color: 'rgba(255,255,255,0.65)', fontSize: '0.92rem', lineHeight: 1.6 }}>
                  {selected.reviewStatus === 'approved'
                    ? 'This case is already approved. No further action is available here.'
                    : selected.reviewStatus === 'rejected'
                      ? 'This case was rejected. No further action is available here.'
                      : 'This record is not in pending review state (for example demo seed data). Approve and reject are only available when review status is pending and the case exists in the review queue.'}
                </Typography>
              )}
              {canActOnSelected && (
                <>
                  <Typography sx={{ color: 'rgba(255,255,255,0.65)', fontSize: '0.92rem', mb: 2, lineHeight: 1.6 }}>
                    Approve and reject require reviewer name and digital signature.
                  </Typography>
                  <Stack direction="row" spacing={1.5} flexWrap="wrap">
                    <Button
                      variant="contained"
                      disabled={reviewActing}
                      onClick={openApproveModal}
                      sx={{
                        backgroundColor: '#15803d',
                        textTransform: 'none',
                        '&:hover': { backgroundColor: '#166534' },
                      }}
                    >
                      Approve
                    </Button>
                    <Button
                      variant="outlined"
                      disabled={reviewActing}
                      onClick={openRejectModal}
                      sx={{
                        textTransform: 'none',
                        color: '#fca5a5',
                        borderColor: 'rgba(239,68,68,0.5)',
                        '&:hover': { borderColor: '#fca5a5' },
                      }}
                    >
                      Reject
                    </Button>
                  </Stack>
                </>
              )}
            </Section>
          </Box>
        )}
      </Drawer>
    </Box>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Box>
      <Typography sx={{ fontWeight: 900, mb: 1.2, color: '#9bb1ff' }}>{title}</Typography>
      {children}
    </Box>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2, py: 0.8 }}>
      <Typography sx={{ color: 'rgba(255,255,255,0.65)' }}>{label}</Typography>
      <Typography sx={{ color: 'rgba(255,255,255,0.92)', fontWeight: 700 }}>{value}</Typography>
    </Box>
  )
}
