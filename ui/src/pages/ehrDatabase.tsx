import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import SignatureCanvas from 'react-signature-canvas'
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
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Radio,
  useMediaQuery,
  useTheme,
} from '@mui/material'
import {
  absolutizeApiAssetUrl,
  fetchEhrRecords,
  approveReview,
  rejectReview,
  NGROK_HEADERS,
  openImage,
  type EhrRecord,
} from '../api/flareAPI'

type EhrRecordLoose = EhrRecord & Record<string, unknown>
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
  /** Best-effort display name for table/report (never use IDs as primary label here). */
  displayName: string
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

function toAbsoluteUrl(url: string | null | undefined): string | undefined {
  return absolutizeApiAssetUrl(url) ?? undefined
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

function formatPatientNameFromLoose(row: EhrRecordLoose): string {
  const fn = String((row as { firstName?: string }).firstName ?? row.first_name ?? '').trim()
  const ln = String((row as { lastName?: string }).lastName ?? row.last_name ?? '').trim()
  const fromParts = [fn, ln].filter(Boolean).join(' ').trim()
  if (fromParts) return fromParts
  for (const k of ['patientName', 'patient_name', 'name'] as const) {
    const v = row[k]
    if (v != null && String(v).trim()) return String(v).trim()
  }
  return 'Unknown Patient'
}

function formatPatientTableSubline(dob: string, location: string): string {
  const d = (dob || '').trim()
  const loc = (location || '').trim()
  if (d && loc) return `DOB: ${d} • ${loc}`
  if (d) return `DOB: ${d}`
  if (loc) return loc
  return ''
}

function mapEhrToPatientRecord(r: EhrRecord): PatientRecord {
  const loose = r as EhrRecordLoose
  const created = r.createdAt?.split('T')[0] ?? ''
  const overlay = r.segmentation?.overlay_url
  const gradCamRaw = loose.gradcam_url ?? (r as { ct_cam_url?: string }).ct_cam_url ?? overlay ?? undefined
  const mriIn = (loose as { mri_input_url?: string }).mri_input_url
  const st = (r.review_status || '').toLowerCase()
  return {
    id: r.caseId,
    firstName: String(r.firstName ?? loose.first_name ?? '').trim(),
    lastName: String(r.lastName ?? loose.last_name ?? '').trim(),
    displayName: formatPatientNameFromLoose(loose),
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
    originalImageUrl: toAbsoluteUrl(r.input_image_url ?? mriIn),
  }
}

function formatHumanReviewStatus(status: string): string {
  const s = (status || '').toLowerCase()
  if (s === 'pending') return 'Pending'
  if (s === 'approved') return 'Approved'
  if (s === 'rejected') return 'Rejected'
  if (!s) return '—'
  return s.replace(/\b\w/g, (c) => c.toUpperCase())
}

function isReviewCompleteForExport(status: string): boolean {
  const s = (status || '').toLowerCase()
  return s === 'approved' || s === 'rejected'
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

const pulseDotSx = {
  display: 'inline-block',
  width: 8,
  height: 8,
  borderRadius: '50%',
  backgroundColor: '#f59e0b',
  marginRight: 1,
  animation: 'pulse 1.5s ease-in-out infinite',
  '@keyframes pulse': {
    '0%, 100%': { opacity: 0.4 },
    '50%': { opacity: 1 },
  },
}

const SIGNATURE_FALLBACK = 'frontend-reviewer'

function escapeHtml(s: string | number | null | undefined): string {
  if (s == null || s === '') return '—'
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Minimal escapes for a long `data:` or `https:` string inside a double-quoted attribute. */
function escapeHtmlAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
}

function safeHref(url: string | null | undefined): string | null {
  if (!url || typeof url !== 'string') return null
  const t = url.trim()
  if (t.startsWith('/') || t.startsWith('http://') || t.startsWith('https://')) return t
  return null
}

function rowHtml(label: string, value: string): string {
  return `<tr><td class="lab">${escapeHtml(label)}</td><td>${escapeHtml(value)}</td></tr>`
}

function pickRawFieldLoose(
  raw: EhrRecordLoose,
  keys: string[],
  fallback = '—'
): string {
  for (const k of keys) {
    const v = raw[k]
    if (v != null && String(v).trim() !== '' && String(v) !== '—') {
      return String(v)
    }
  }
  return fallback
}

const REPORT_IMAGE_FETCH_INIT: RequestInit = {
  mode: 'cors',
  credentials: 'omit',
  cache: 'no-store',
  headers: NGROK_HEADERS,
}

function responseContentTypeIsImage(res: Response): boolean {
  const raw = (res.headers.get('content-type') || '').split(';')[0].trim().toLowerCase()
  return raw.startsWith('image/')
}

async function tryFetchImageDataUrl(url: string): Promise<string | null> {
  const t = (url || '').trim()
  if (!t) return null
  if (t.toLowerCase().startsWith('data:image/')) return t
  if (!safeHref(t)) return null

  const full = absolutizeApiAssetUrl(t)
  if (!full) return null
  if (full.toLowerCase().startsWith('data:image/')) return full
  if (!full.startsWith('http://') && !full.startsWith('https://')) return null

  try {
    const res = await fetch(full, REPORT_IMAGE_FETCH_INIT)
    if (!res.ok) return null
    if (!responseContentTypeIsImage(res)) return null

    const blob = await res.blob()
    if (blob.size === 0) return null
    if (blob.type && !blob.type.startsWith('image/')) return null

    return await new Promise((resolve) => {
      const fr = new FileReader()
      fr.onload = () => {
        const s = String(fr.result || '')
        if (!s.toLowerCase().startsWith('data:image/')) {
          resolve(null)
          return
        }
        resolve(s)
      }
      fr.onerror = () => resolve(null)
      fr.readAsDataURL(blob)
    })
  } catch {
    return null
  }
}

function signatureBlockHtml(raw: EhrRecordLoose): string {
  const sig = pickRawFieldLoose(
    raw,
    ['signature', 'digitalSignature', 'digital_signature'],
    '—'
  )
  if (sig !== '—' && sig.toLowerCase().startsWith('data:image/')) {
    return `<p class="sigbox"><em>Signature image (captured in workflow)</em></p><p><img class="evimg" src="${escapeHtmlAttr(sig)}" alt="Digital signature" /></p>`
  }
  return ''
}

async function buildVisualRowsEmbedded(raw: EhrRecordLoose): Promise<string> {
  const overlayUrl =
    raw.segmentation && typeof raw.segmentation === 'object'
      ? (raw.segmentation as { overlay_url?: string }).overlay_url
      : undefined
  const pairs: [string, string | null | undefined][] = [
    ['CT / Grad-CAM (gradcam_url)', raw.gradcam_url as string | undefined],
    ['CT cam (ct_cam_url)', (raw as { ct_cam_url?: string }).ct_cam_url],
    ['MRI / segmentation overlay', overlayUrl],
    ['MRI overlay (mri_overlay_url)', (raw as { mri_overlay_url?: string }).mri_overlay_url],
    ['Input / scan image (input_image_url)', raw.input_image_url as string | undefined],
    ['MRI input (mri_input_url)', (raw as { mri_input_url?: string }).mri_input_url],
  ]
  const rows: string[] = []
  for (const [label, u] of pairs) {
    if (!u || !String(u).trim()) continue
    const data = await tryFetchImageDataUrl(String(u))
    const h = safeHref(String(u))
    const normalizedForLink = h
      ? absolutizeApiAssetUrl(String(u).trim()) ?? String(u).trim()
      : ''
    if (data) {
      rows.push(
        `<tr><td class="lab" colspan="2"><div class="vlabel">${escapeHtml(
          label
        )}</div><div class="vimg"><img class="evimg" src="${escapeHtmlAttr(
          data
        )}" alt="${escapeHtml(label)}" /></div></td></tr>`
      )
    } else {
      const fallback = h
        ? `Image could not be embedded. Source: <a href="${escapeHtml(
            normalizedForLink
          )}">${escapeHtml(normalizedForLink)}</a>`
        : '—'
      rows.push(
        `<tr><td class="lab">${escapeHtml(
          label
        )}</td><td class="vfall">${fallback}</td></tr>`
      )
    }
  }
  if (rows.length === 0) {
    return rowHtml('Images', 'No image URLs in this case.')
  }
  return rows.join('')
}

async function buildClinicalReportHtml(
  p: PatientRecord,
  raw: EhrRecordLoose
): Promise<string> {
  const genAt = new Date().toISOString()
  const sig = pickRawFieldLoose(
    raw,
    ['signature', 'digitalSignature', 'digital_signature'],
    ''
  )
  const sigPresent = sig && sig !== '—' && !/^null$/i.test(sig) ? 'Yes' : 'No'
  const pl = raw.pred_label as string | undefined
  const pred = raw.prediction as string | undefined
  const ctProb = raw.ct_prob as number | undefined
  const mriProb = raw.mri_prob as number | undefined
  const fusionScore = raw.fusion_score as number | undefined
  const fusionMode = raw.fusion_mode as string | undefined
  const st = (pickRawFieldLoose(raw, ['review_status'], '') || p.reviewStatus || '')
    .toLowerCase()
    .trim()
  const pending = st === 'pending' || p.reviewStatus === 'pending'
  const pendingNote = pending
    ? '<p class="pending human"><strong>Pending clinician review.</strong> Reviewer and signature fields may be incomplete.</p>'
    : ''

  const patientBlock = [
    rowHtml('Patient name (as shown in app)', p.displayName || '—'),
    rowHtml('Patient ID', String(raw.patient_id ?? '—')),
    rowHtml('Medical ID', p.medicalId || '—'),
    rowHtml('First name (record)', p.firstName || '—'),
    rowHtml('Last name (record)', p.lastName || '—'),
    rowHtml('Date of birth', p.dob?.trim() ? p.dob : '—'),
    rowHtml('Case ID', p.id),
    rowHtml('Hospital ID', String(raw.hospitalId ?? '—')),
    rowHtml('Hospital / facility', String(raw.hospitalName ?? p.location ?? '—')),
  ]

  const studyBlock = [
    rowHtml('Modality', String(p.modality)),
    rowHtml('Cancer type (demo)', String(p.cancerType)),
    rowHtml('Case / scan date (from createdAt)', String(raw.createdAt ?? p.scanDate ?? '—')),
    rowHtml('Report generated (ISO timestamp)', genAt),
  ]

  const aiLines: string[] = [
    rowHtml('result_class (AI / fusion triage)', String(raw.result_class ?? p.aiResult)),
    rowHtml('prediction (if present)', String(pred ?? '—')),
    rowHtml('pred_label (tumor / modality detail)', String(pl ?? '—')),
    rowHtml('confidence (API value)', String(raw.confidence ?? '—')),
    rowHtml('confidence (UI display %)', `${p.confidence}%`),
  ]
  if (typeof ctProb === 'number' && Number.isFinite(ctProb)) {
    aiLines.push(rowHtml('ct_prob (CT abnormality)', String(ctProb)))
  }
  if (typeof mriProb === 'number' && Number.isFinite(mriProb)) {
    aiLines.push(rowHtml('mri_prob (MRI abnormality mass)', String(mriProb)))
  }
  if (typeof fusionScore === 'number' && Number.isFinite(fusionScore)) {
    aiLines.push(rowHtml('fusion_score', String(fusionScore)))
  }
  if (fusionMode) {
    aiLines.push(rowHtml('fusion_mode', String(fusionMode)))
  }

  const reviewBlock = [
    rowHtml('Review status', pickRawFieldLoose(raw, ['review_status'], p.reviewStatus || '—')),
    rowHtml('Reviewer name', pickRawFieldLoose(raw, ['reviewerName', 'reviewer_name'], '—')),
    rowHtml('Reviewer ID', pickRawFieldLoose(raw, ['reviewerId', 'reviewer_id'], '—')),
    rowHtml('Digital signature on file', sigPresent),
    rowHtml('Approved at', pickRawFieldLoose(raw, ['approvedAt', 'approved_at'], '—')),
    rowHtml('Rejected at', pickRawFieldLoose(raw, ['rejectedAt', 'rejected_at'], '—')),
    rowHtml(
      'Rejection / review reason',
      pickRawFieldLoose(raw, ['reject_reason', 'rejectionReason'], '—')
    ),
    rowHtml('Notes', pickRawFieldLoose(raw, ['notes'], p.notes && p.notes !== '—' ? p.notes : '—')),
  ]

  const visualTableBody = await buildVisualRowsEmbedded(raw)
  const sigHtml = signatureBlockHtml(raw)

  const disclaimer =
    'This report was generated by FLARE, a prototype AI decision-support system. It is not a certified medical device, not a legal medical record, and must be reviewed and confirmed by a qualified clinician before any clinical use. FLARE demonstrates HIPAA-aligned prototype safeguards but is not a production EHR system.'

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FLARE Clinical AI Review Report</title>
  <style>
    body { font-family: Georgia, 'Times New Roman', serif; color: #111; background: #fff; margin: 24px; line-height: 1.45; font-size: 11pt; }
    h1 { font-size: 18pt; margin: 0 0 4px; }
    .subtitle { font-size: 12pt; color: #333; margin: 0 0 20px; font-weight: 600; }
    h2 { font-size: 12pt; margin: 18px 0 8px; border-bottom: 1px solid #222; padding-bottom: 4px; }
    .box { border: 1px solid #bbb; padding: 12px 14px; margin-bottom: 12px; background: #fafafa; }
    .ai { background: #e8f2fc; border-color: #6b90b5; }
    .human { background: #eef8ee; border-color: #6a9a6a; }
    .disc { background: #fffbf0; border: 1px solid #c9a227; padding: 14px; margin-top: 18px; font-size: 10pt; line-height: 1.5; }
    .pending { font-size: 10pt; color: #0f5132; margin: 0 0 8px; }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 5px 8px; vertical-align: top; }
    td.lab { width: 40%; font-weight: 600; color: #333; }
    a { color: #0b57d0; word-break: break-all; }
    .note { font-size: 9pt; color: #555; margin-top: 10px; }
    .evimg { max-width: 100%; max-height: 360px; height: auto; object-fit: contain; border: 1px solid #ccc; }
    .vimg { margin-top: 6px; }
    .vlabel { font-size: 10pt; font-weight: 600; margin-bottom: 4px; }
    .vfall { font-size: 9pt; }
    .sigbox { margin: 8px 0 0; }
    @media print {
      body { margin: 12mm; }
      a { color: #000; }
    }
  </style>
</head>
<body>
  <h1>FLARE Clinical AI Review Report</h1>
  <p class="subtitle">Prototype EHR-Style Documentation</p>

  <h2>Patient information</h2>
  <div class="box"><table>${patientBlock.join('')}</table></div>

  <h2>Study information</h2>
  <div class="box"><table>${studyBlock.join('')}</table></div>

  <h2>AI-generated results</h2>
  <div class="box ai"><table>${aiLines.join('')}</table></div>

  <h2>Visual evidence</h2>
  <div class="box"><table>${visualTableBody}</table></div>

  <h2>Clinical review (human decision)</h2>
  <div class="box human">
    ${pendingNote}
    <table>${reviewBlock.join('')}</table>
    ${sigHtml}
  </div>

  <div class="disc">${escapeHtml(disclaimer)}</div>
  <p class="note">The sections above separate automated AI outputs from human review decisions. This document is for demonstration only.</p>
</body>
</html>`
}

function snackbarAlertSx(severity: 'success' | 'error' | 'warning' | 'info') {
  const icon = {
    '& .MuiAlert-icon': { color: '#fff' },
    '& .MuiAlert-action': { color: '#fff' },
  }
  const base = {
    width: '100%',
    alignItems: 'center',
    color: '#fff',
    fontWeight: 600,
    ...icon,
  }
  switch (severity) {
    case 'success':
      return { ...base, bgcolor: '#166534' }
    case 'error':
      return { ...base, bgcolor: '#7f1d1d' }
    case 'warning':
      return { ...base, bgcolor: '#a16207' }
    case 'info':
    default:
      return { ...base, bgcolor: '#1e3a5f' }
  }
}

function openPrintableReport(html: string): boolean {
  const w = window.open('', '_blank')
  if (!w) return false
  w.document.open()
  w.document.write(html)
  w.document.close()
  window.setTimeout(() => {
    try {
      w.focus()
      w.print()
    } catch {
      /* ignore */
    }
  }, 250)
  return true
}

export default function EhrDatabase() {
  const navigate = useNavigate()
  const { user } = useAuth0()
  const muiTheme = useTheme()
  const reportChipNarrow = useMediaQuery(muiTheme.breakpoints.down('md'))
  const reportChipPrefix = reportChipNarrow ? 'Selected:' : 'Selected report case:'
  const sigRef = useRef<SignatureCanvas | null>(null)

  const [records, setRecords] = useState<PatientRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState('')

  const [query, setQuery] = useState('')
  const [cancerFilter, setCancerFilter] = useState<BrainCancerLabel | 'All'>('All')
  const [resultFilter, setResultFilter] = useState<ResultClass | 'All'>('All')
  const [selected, setSelected] = useState<PatientRecord | null>(null)
  /** Single explicit case for printable report (independent of View / drawer). */
  const [exportCaseId, setExportCaseId] = useState<string | null>(null)
  /** When true, the table shows the “For report” column (hidden until user prepares). */
  const [reportSelectMode, setReportSelectMode] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const [signOpen, setSignOpen] = useState(false)
  const [signDialogMode, setSignDialogMode] = useState<'approve' | 'reject'>('approve')
  const [reviewerName, setReviewerName] = useState('')
  const [rejectReason, setRejectReason] = useState('')
  const [signModalError, setSignModalError] = useState('')
  const [reviewActing, setReviewActing] = useState(false)
  const [drawerReviewMsg, setDrawerReviewMsg] = useState('')
  const [drawerReviewErr, setDrawerReviewErr] = useState('')
  const [ehrSourceRows, setEhrSourceRows] = useState<EhrRecord[]>([])
  const [snackbar, setSnackbar] = useState<{
    open: boolean
    message: string
    severity: 'success' | 'error' | 'warning' | 'info'
  }>({ open: false, message: '', severity: 'info' })

  const defaultReviewerName = useMemo(() => {
    if (!user) return SIGNATURE_FALLBACK
    const n =
      user.name?.trim() ||
      user.nickname?.trim() ||
      [user.given_name, user.family_name].filter(Boolean).join(' ').trim()
    return n && n.length > 0 ? n : SIGNATURE_FALLBACK
  }, [user])

  const loadRecords = useCallback(async () => {
    setFetchError('')
    setLoading(true)
    try {
      const { records: rows } = await fetchEhrRecords()
      setEhrSourceRows(rows ?? [])
      const mapped = (rows ?? []).map(mapEhrToPatientRecord)
      setRecords(mapped)
      setSelected((prev) => {
        if (!prev) return null
        return mapped.find((x) => x.id === prev.id) ?? prev
      })
      setExportCaseId((prev) => {
        if (!prev) return null
        return mapped.some((x) => x.id === prev) ? prev : null
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
        r.displayName.toLowerCase().includes(q) ||
        r.medicalId.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)

      const matchesCancer = cancerFilter === 'All' ? true : r.cancerType === cancerFilter
      const matchesResult = resultFilter === 'All' ? true : r.aiResult === resultFilter

      return matchesQuery && matchesCancer && matchesResult
    })
  }, [records, query, cancerFilter, resultFilter])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const aPending = a.reviewStatus === 'pending' ? 0 : 1
      const bPending = b.reviewStatus === 'pending' ? 0 : 1
      return aPending - bPending
    })
  }, [filtered])

  const exportRecord = useMemo(
    () => (exportCaseId ? (records.find((x) => x.id === exportCaseId) ?? null) : null),
    [exportCaseId, records]
  )

  const exportCaseReviewComplete = useMemo(
    () => Boolean(exportRecord && isReviewCompleteForExport(exportRecord.reviewStatus)),
    [exportRecord]
  )

  const canActOnSelected = Boolean(selected?.id && selected.reviewStatus === 'pending')

  const openCaseDetail = useCallback((r: PatientRecord) => {
    setSelected(r)
    setDrawerOpen(true)
  }, [])

  const clearSignPad = useCallback(() => {
    sigRef.current?.clear()
  }, [])

  const closeSignDialog = useCallback(() => {
    if (reviewActing) return
    setSignModalError('')
    setRejectReason('')
    setSignOpen(false)
    clearSignPad()
  }, [reviewActing, clearSignPad])

  useEffect(() => {
    if (!signOpen) {
      clearSignPad()
      return
    }
    const t = window.setTimeout(() => {
      sigRef.current?.clear()
    }, 0)
    return () => window.clearTimeout(t)
  }, [signOpen, signDialogMode, clearSignPad])

  function openApproveModal() {
    if (!selected?.id || selected.reviewStatus !== 'pending') return
    setSignDialogMode('approve')
    setReviewerName(defaultReviewerName)
    setRejectReason('')
    setSignModalError('')
    setSignOpen(true)
  }

  function openRejectModal() {
    if (!selected?.id || selected.reviewStatus !== 'pending') return
    setSignDialogMode('reject')
    setReviewerName(defaultReviewerName)
    setRejectReason('')
    setSignModalError('')
    setSignOpen(true)
  }

  async function confirmSignDialog() {
    if (!selected?.id) return
    const name = reviewerName.trim() || defaultReviewerName
    const sigPad = sigRef.current
    if (!sigPad || sigPad.isEmpty()) {
      setSignModalError('Please sign in the box before continuing.')
      return
    }
    if (signDialogMode === 'reject' && !rejectReason.trim()) {
      setSignModalError('A rejection reason is required.')
      return
    }
    const signature = sigPad.getTrimmedCanvas().toDataURL('image/png')
    setSignModalError('')
    setReviewActing(true)
    setDrawerReviewErr('')
    setDrawerReviewMsg('')
    try {
      if (signDialogMode === 'approve') {
        await approveReview(selected.id, { reviewerName: name, signature })
        setDrawerReviewMsg('Case approved.')
      } else {
        await rejectReview(selected.id, {
          reviewerName: name,
          reason: rejectReason.trim(),
          signature,
        })
        setDrawerReviewMsg('Case rejected.')
      }
      setRejectReason('')
      setSignOpen(false)
      clearSignPad()
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

  const handlePrepareReport = useCallback(() => {
    setReportSelectMode(true)
    setExportCaseId(null)
    setSnackbar({
      open: true,
      message: 'Choose one case to include in the report.',
      severity: 'info',
    })
  }, [])

  const handleCancelReportSelection = useCallback(() => {
    setReportSelectMode(false)
    setExportCaseId(null)
  }, [])

  const handleExportReport = useCallback(async () => {
    if (!exportCaseId || !exportRecord) {
      setSnackbar({
        open: true,
        message:
          reportSelectMode
            ? 'Select exactly one case using the “For report” control, then click Export selected report. Or use “Select for report” in the case details panel (when prepared).'
            : 'Click “Select report case”, choose a case, then click Export selected report.',
        severity: 'warning',
      })
      return
    }
    if (!isReviewCompleteForExport(exportRecord.reviewStatus)) {
      setSnackbar({
        open: true,
        message: 'Approve or reject this case before exporting a clinical review report.',
        severity: 'warning',
      })
      return
    }
    const raw = ehrSourceRows.find((r) => r.caseId === exportCaseId) as EhrRecordLoose | undefined
    if (!raw) {
      setSnackbar({
        open: true,
        message: 'Could not load source data for this case.',
        severity: 'error',
      })
      return
    }
    let html: string
    try {
      html = await buildClinicalReportHtml(exportRecord, raw)
    } catch (e) {
      setSnackbar({
        open: true,
        message: e instanceof Error ? e.message : 'Could not build the report.',
        severity: 'error',
      })
      return
    }
    const ok = openPrintableReport(html)
    if (ok) {
      setSnackbar({
        open: true,
        message: 'Report opened — use your browser Print dialog to save as PDF or paper.',
        severity: 'success',
      })
    } else {
      setSnackbar({
        open: true,
        message: 'Could not open print window. Allow pop-ups and try again.',
        severity: 'error',
      })
    }
  }, [exportCaseId, exportRecord, ehrSourceRows, reportSelectMode])

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
        onClose={closeSignDialog}
        PaperProps={{
          sx: {
            backgroundColor: '#0f1117',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 2,
            minWidth: { xs: '92vw', sm: 420 },
            maxWidth: 480,
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 800, pt: 2.5, px: 3, pb: 1.5 }}>
          {signDialogMode === 'reject' ? 'Reject case' : 'Approve case'}
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 3, px: 3, pb: 1 }}>
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
            placeholder={defaultReviewerName}
            helperText={`Defaults to Auth0 profile, or “${SIGNATURE_FALLBACK}” if empty on submit.`}
            FormHelperTextProps={{ sx: { color: 'rgba(255,255,255,0.45)' } }}
            sx={{ ...fieldSx, mt: 1 }}
          />
          {signDialogMode === 'reject' && (
            <TextField
              label="Rejection reason"
              required
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              fullWidth
              multiline
              minRows={2}
              sx={fieldSx}
            />
          )}
          <Box>
            <Typography sx={{ color: 'rgba(255,255,255,0.75)', mb: 0.75, fontSize: '0.88rem' }}>
              Signature
            </Typography>
            <SignatureCanvas
              ref={sigRef}
              penColor="#ffffff"
              backgroundColor="rgba(255,255,255,0.06)"
              clearOnResize={false}
              canvasProps={{
                width: 400,
                height: 160,
                className: 'ehr-signature-canvas',
                style: {
                  width: '100%',
                  maxWidth: 400,
                  height: 160,
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.2)',
                },
              }}
            />
            <Button
              type="button"
              size="small"
              onClick={clearSignPad}
              disabled={reviewActing}
              sx={{ mt: 1, textTransform: 'none', color: 'rgba(255,255,255,0.75)' }}
            >
              Clear Signature
            </Button>
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={closeSignDialog}
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

      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <Box sx={{ pr: { md: 2 }, pb: 1, minWidth: 0, flex: '1 1 280px' }}>
          <Typography sx={{ fontSize: '1.7rem', fontWeight: 800, letterSpacing: '0.02em' }}>
            EHR Database
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.65)', mt: 0.75, maxWidth: 720, lineHeight: 1.65 }}>
            For a printable report, select one reviewed case. Bulk export is disabled.
          </Typography>
        </Box>

        <Stack
          direction="row"
          spacing={2}
          flexWrap="wrap"
          alignItems="center"
          useFlexGap
          sx={{ justifyContent: 'flex-end', rowGap: 1.5, columnGap: 2, py: 0.5 }}
        >
          <Button
            type="button"
            variant="outlined"
            onClick={handlePrepareReport}
            sx={{
              borderColor: 'rgba(255,255,255,0.18)',
              color: '#fff',
              textTransform: 'none',
              borderRadius: 2,
              fontWeight: 700,
              px: 1.75,
              py: 0.75,
              '&:hover': { borderColor: 'rgba(255,255,255,0.35)' },
            }}
          >
            Select report case
          </Button>
          {reportSelectMode && (
            <Button
              type="button"
              variant="text"
              onClick={handleCancelReportSelection}
              sx={{ color: 'rgba(255,200,200,0.95)', textTransform: 'none', fontWeight: 600, px: 1.5 }}
            >
              Cancel
            </Button>
          )}
          {exportCaseId && exportRecord && (
            <Tooltip title="Only this case will be included. Export is allowed after the case is approved or rejected.">
              <Chip
                size="small"
                label={`${reportChipPrefix} ${exportRecord.displayName} · ${exportRecord.medicalId} · ${formatHumanReviewStatus(exportRecord.reviewStatus)}`}
                sx={{
                  maxWidth: { xs: '100%', md: 480 },
                  fontWeight: 700,
                  color: '#fff',
                  backgroundColor: 'rgba(255,92,92,0.22)',
                  border: '1px solid rgba(255,92,92,0.45)',
                  py: 0.5,
                  height: 'auto',
                  minHeight: 32,
                  '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'normal', lineHeight: 1.3 },
                }}
              />
            </Tooltip>
          )}
          <Tooltip
            title={
              !exportCaseId
                ? 'Select a report case, then choose a row. Export selected report is enabled after selection and review completion.'
                : !exportCaseReviewComplete
                  ? 'This case must be approved or rejected before export.'
                  : 'Opens a print-ready clinical AI review for the selected case only.'
            }
          >
            <span>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                disabled={!exportCaseId || !exportCaseReviewComplete}
                sx={{
                  borderColor: 'rgba(255,255,255,0.18)',
                  color: '#fff',
                  textTransform: 'none',
                  borderRadius: 2,
                  fontWeight: 600,
                  px: 1.75,
                  py: 0.75,
                  '&:hover': { borderColor: 'rgba(255,255,255,0.35)' },
                  '&.Mui-disabled': { color: 'rgba(255,255,255,0.35)' },
                }}
                onClick={() => void handleExportReport()}
              >
                Export selected report
              </Button>
            </span>
          </Tooltip>

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
              {reportSelectMode && (
                <TableCell
                  align="center"
                  sx={{ color: 'rgba(255,255,255,0.75)', fontWeight: 700, width: 72, py: 1.5 }}
                >
                  <Tooltip title="Single-select. Only this row is included in the printable report.">
                    <span>For report</span>
                  </Tooltip>
                </TableCell>
              )}
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
            {sorted.map((r) => {
              const chip = resultChipColor(r.aiResult)
              const sub = formatPatientTableSubline(r.dob, r.location)
              const reportHighlight = reportSelectMode && exportCaseId === r.id
              return (
                <TableRow
                  key={r.id}
                  hover
                  onClick={() => openCaseDetail(r)}
                  selected={reportHighlight}
                  sx={{
                    cursor: 'pointer',
                    position: 'relative',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.03)' },
                    ...(reportHighlight
                      ? {
                          backgroundColor: 'rgba(59,130,246,0.08)',
                          boxShadow: 'inset 3px 0 0 #ff5c5c',
                          '&.Mui-selected': { backgroundColor: 'rgba(59,130,246,0.1)' },
                        }
                      : {}),
                  }}
                >
                  {reportSelectMode && (
                    <TableCell
                      align="center"
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => e.stopPropagation()}
                      sx={{ py: 1, width: 72 }}
                    >
                      <Radio
                        checked={exportCaseId === r.id}
                        onChange={() => {
                          setExportCaseId(r.id)
                        }}
                        onClick={(e) => e.stopPropagation()}
                        value={r.id}
                        name="flare-ehr-export-case"
                        size="small"
                        sx={{
                          p: 0.5,
                          color: 'rgba(255,255,255,0.45)',
                          '&.Mui-checked': { color: '#ff5c5c' },
                        }}
                        inputProps={{ 'aria-label': `Select case ${r.id} for printable report` }}
                      />
                    </TableCell>
                  )}
                  <TableCell sx={{ color: '#fff' }}>
                    <Typography sx={{ fontWeight: 700 }}>{r.displayName}</Typography>
                    {sub ? (
                      <Typography sx={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.85rem' }}>
                        {sub}
                      </Typography>
                    ) : null}
                  </TableCell>

                  <TableCell sx={{ color: '#fff', fontWeight: 700 }}>
                    {r.reviewStatus === 'pending' && (
                      <Tooltip title="Pending review">
                        <Box component="span" sx={pulseDotSx} />
                      </Tooltip>
                    )}
                    {r.medicalId}
                  </TableCell>
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
                        openCaseDetail(r)
                      }}
                      sx={{ textTransform: 'none', color: '#fff' }}
                    >
                      View
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}

            {sorted.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={reportSelectMode ? 9 : 8}
                  sx={{ color: 'rgba(255,255,255,0.65)', py: 5, textAlign: 'center' }}
                >
                  No matching records found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Drawer
        anchor="right"
        open={drawerOpen && Boolean(selected)}
        onClose={() => setDrawerOpen(false)}
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
            <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 1, gap: 1 }}>
              <Box sx={{ minWidth: 0 }}>
                <Typography sx={{ fontWeight: 900, fontSize: '1.25rem' }}>
                  {selected.displayName}
                </Typography>
                <Typography sx={{ color: 'rgba(255,255,255,0.65)' }}>
                  Case ID: {selected.id} • Medical ID: {selected.medicalId}
                </Typography>
                {reportSelectMode && (
                  <>
                    {exportCaseId === selected.id ? (
                      <Chip
                        size="small"
                        sx={{
                          mt: 1.5,
                          fontWeight: 800,
                          color: '#fff',
                          backgroundColor: 'rgba(255,92,92,0.2)',
                          border: '1px solid rgba(255,92,92,0.4)',
                          maxWidth: '100%',
                          height: 'auto',
                          minHeight: 32,
                          py: 0.5,
                          '& .MuiChip-label': { whiteSpace: 'normal', lineHeight: 1.35 },
                        }}
                        label={`${reportChipPrefix} ${selected.displayName} · ${selected.medicalId} · ${formatHumanReviewStatus(selected.reviewStatus)}`}
                      />
                    ) : (
                      <Button
                        type="button"
                        size="small"
                        onClick={() => {
                          setExportCaseId(selected.id)
                          setSnackbar({
                            open: true,
                            message: 'This case is now selected. Click Export selected report when ready (approved or rejected cases only).',
                            severity: 'info',
                          })
                        }}
                        sx={{ mt: 1.25, textTransform: 'none', color: '#ffb4b4', fontWeight: 700 }}
                      >
                        Select for report
                      </Button>
                    )}
                  </>
                )}
              </Box>

              <IconButton onClick={() => setDrawerOpen(false)} sx={{ color: '#fff' }} aria-label="Close details">
                <CloseIcon />
              </IconButton>
            </Box>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            <Section title="Patient Information">
              {selected.dob?.trim() ? <InfoRow label="Date of Birth" value={selected.dob} /> : null}
              <InfoRow label="Location" value={selected.location || '—'} />
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
                    if (u) void openImage(u)
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
                    if (u) void openImage(u)
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
                    Approve and reject require reviewer name, signature, and (for reject) a reason.
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

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
          sx={snackbarAlertSx(snackbar.severity)}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
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
