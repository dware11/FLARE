import { useMemo, useState } from 'react'
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
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search'
import CloseIcon from '@mui/icons-material/Close'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import DownloadIcon from '@mui/icons-material/Download'
import VisibilityIcon from '@mui/icons-material/Visibility'

type CancerType = 'Brain' | 'Breast'
type ResultClass = 'Normal' | 'Benign' | 'Malignant'
type ScanModality = 'MRI' | 'CT' | 'Mammography' | 'Ultrasound'

type PatientRecord = {
  id: string
  firstName: string
  lastName: string
  dob: string
  medicalId: string
  location: string
  cancerType: CancerType
  modality: ScanModality
  scanDate: string
  aiResult: ResultClass
  confidence: number // 0-100
  notes?: string
  // In a real app these would be URLs returned by backend (S3, database)
  gradCamUrl?: string
  originalImageUrl?: string
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

export default function EhrDatabase() {
  // Mock records (replace with fetch from backend later)
  const records: PatientRecord[] = useMemo(
    () => [
      {
        id: 'case-1001',
        firstName: 'Ava',
        lastName: 'Johnson',
        dob: '1991-05-18',
        medicalId: 'HOU-23910',
        location: 'Houston, TX',
        cancerType: 'Breast',
        modality: 'Mammography',
        scanDate: '2026-01-05',
        aiResult: 'Benign',
        confidence: 86,
        notes: 'Follow-up recommended in 6 months.',
      },
      {
        id: 'case-1002',
        firstName: 'Noah',
        lastName: 'Williams',
        dob: '1983-11-02',
        medicalId: 'HOU-77102',
        location: 'Houston, TX',
        cancerType: 'Brain',
        modality: 'MRI',
        scanDate: '2026-01-06',
        aiResult: 'Malignant',
        confidence: 92,
        notes: 'High confidence. Consider specialist review.',
      },
      {
        id: 'case-1003',
        firstName: 'Mia',
        lastName: 'Brown',
        dob: '2001-03-29',
        medicalId: 'HOU-90871',
        location: 'Houston, TX',
        cancerType: 'Breast',
        modality: 'Ultrasound',
        scanDate: '2026-01-07',
        aiResult: 'Normal',
        confidence: 78,
        notes: 'No suspicious findings detected.',
      },
      {
        id: 'case-1004',
        firstName: 'Ethan',
        lastName: 'Davis',
        dob: '1976-09-14',
        medicalId: 'HOU-55019',
        location: 'Houston, TX',
        cancerType: 'Brain',
        modality: 'CT',
        scanDate: '2026-01-07',
        aiResult: 'Benign',
        confidence: 73,
        notes: 'Low-risk features. Monitor symptoms.',
      },
    ],
    []
  )

  const [query, setQuery] = useState('')
  const [cancerFilter, setCancerFilter] = useState<CancerType | 'All'>('All')
  const [resultFilter, setResultFilter] = useState<ResultClass | 'All'>('All')
  const [selected, setSelected] = useState<PatientRecord | null>(null)

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
      {/* Page Title */}
      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2, mb: 3 }}>
        <Box>
          <Typography sx={{ fontSize: '1.7rem', fontWeight: 800, letterSpacing: '0.02em' }}>
            EHR Database
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.65)', mt: 0.5 }}>
            Patient scan history and AI diagnostic results
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
            onClick={() => alert('Mock action: route to Cancer Detection page to create a new scan')}
          >
            New Scan
          </Button>
        </Stack>
      </Box>

      {/* Filters */}
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
              onChange={(e) => setCancerFilter(e.target.value as any)}
              sx={{
                '& .MuiInputBase-root': { color: '#fff', borderRadius: 2 },
                '& label': { color: 'rgba(255,255,255,0.65)' },
                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
              }}
            >
              <MenuItem value="All">All</MenuItem>
              <MenuItem value="Brain">Brain</MenuItem>
              <MenuItem value="Breast">Breast</MenuItem>
            </TextField>

            <TextField
              select
              label="AI Result"
              value={resultFilter}
              onChange={(e) => setResultFilter(e.target.value as any)}
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

      {/* Table */}
      <TableContainer
        component={Paper}
        sx={{
          backgroundColor: 'rgba(0,0,0,0.30)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <Table>
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

      {/* Details Drawer */}
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

            {/* Patient Info */}
            <Section title="Patient Information">
              <InfoRow label="Date of Birth" value={selected.dob} />
              <InfoRow label="Location" value={selected.location} />
            </Section>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            {/* Scan Info */}
            <Section title="Scan Details">
              <InfoRow label="Cancer Type" value={selected.cancerType} />
              <InfoRow label="Modality" value={selected.modality} />
              <InfoRow label="Scan Date" value={selected.scanDate} />
            </Section>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.08)', my: 2 }} />

            {/* AI Result */}
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
                  onClick={() => alert('Mock: open original scan image')}
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
                  onClick={() => alert('Mock: open Grad-CAM / segmentation overlay')}
                >
                  View Localization
                </Button>
              </Stack>
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