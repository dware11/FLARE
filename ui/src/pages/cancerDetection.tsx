import { useCallback, useEffect, useMemo, useState } from "react";
import type { InputHTMLAttributes } from "react";
import {
  Box,
  Typography,
  TextField,
  MenuItem,
  Button,
  Card,
  CardContent,
  Divider,
  Alert,
  Chip,
  CircularProgress,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import {
  predictScan,
  predictMriBraTSFolder,
  API_BASE,
  type CancerScanResult,
} from "../api/flareAPI";
import type { CancerType, ResultClass } from "../api/flareAPI";
import {
  MRI_SEQUENCE_KEYS,
  MRI_SEQUENCE_LABELS,
  analyzePatientFolder,
  mriBraTSFolderComplete,
} from "../utils/scanFolderModality";

const HOSPITALS = [
  { id: "H001", name: "Houston Methodist Hospital" },
  { id: "H002", name: "Memorial Hermann - Texas Medical Center" },
  { id: "H003", name: "Baylor St. Luke's Medical Center" },
  { id: "H004", name: "Ben Taub Hospital" },
  { id: "H005", name: "Texas Children's Hospital" },
];

function hospitalName(id: string): string {
  return HOSPITALS.find((h) => h.id === id)?.name ?? id;
}

function predictionTheme(p: ResultClass) {
  switch (p) {
    case "Malignant":
      return {
        bg: "rgba(239,68,68,0.15)",
        border: "rgba(239,68,68,0.3)",
        text: "#fca5a5",
        bar: "#ef4444",
      };
    case "Benign":
      return {
        bg: "rgba(59,130,246,0.15)",
        border: "rgba(59,130,246,0.3)",
        text: "#93c5fd",
        bar: "#3b82f6",
      };
    default:
      return {
        bg: "rgba(34,197,94,0.15)",
        border: "rgba(34,197,94,0.3)",
        text: "#86efac",
        bar: "#22c55e",
      };
  }
}

const fieldSx = {
  "& .MuiInputBase-root": { color: "#fff", borderRadius: 2 },
  "& label": { color: "rgba(255,255,255,0.65)" },
  "& fieldset": { borderColor: "rgba(255,255,255,0.12)" },
};

const cardSx = {
  backgroundColor: "rgba(0,0,0,0.30)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 3,
  boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
  color: "#fff",
};

const placeholderBoxSx = {
  minHeight: 200,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(255,255,255,0.04)",
  border: "1px dashed rgba(255,255,255,0.12)",
  borderRadius: 2,
  px: 2,
  textAlign: "center" as const,
};

/** NIfTI volumes are binary — no <img> preview; server extracts one slice for BRISC. */
function isNiftiFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".nii.gz") || n.endsWith(".nii");
}

/** NIfTI volumes are binary — no <img> preview; server extracts one slice for BRISC. */
function isNiftiFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".nii.gz") || n.endsWith(".nii");
}

export default function CancerDetection() {
  const [hospitalId, setHospitalId] = useState("H001");
  const [cancerType, setCancerType] = useState<CancerType | "">("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [medicalId, setMedicalId] = useState("");
  const [dob, setDob] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  /** Brain only: single slice/volume vs BraTS-style patient folder. */
  const [brainUploadMode, setBrainUploadMode] = useState<"single" | "folder">("single");
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [folderInputKey, setFolderInputKey] = useState(0);

  const [loading, setLoading] = useState(false);
  /** Legacy `/predict` shape vs newer `/api/mri/predict` classification + segmentation JSON. */
  const [scanResult, setScanResult] = useState<CancerScanResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (brainUploadMode === "folder") {
      setImagePreviewUrl(null);
      return;
    }
    // Only raster scans get a blob URL; NIfTI would produce a broken <img> src
    if (!file || isNiftiFile(file)) {
      setImagePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file, brainUploadMode]);

  const folderAnalysis = useMemo(() => analyzePatientFolder(folderFiles), [folderFiles]);

  const canUpload = useMemo(() => {
    return Boolean(cancerType) && firstName && lastName && medicalId && dob && hospitalId;
  }, [cancerType, firstName, lastName, medicalId, dob, hospitalId]);

  const canSubmit = useMemo(() => {
    if (!canUpload) return false;
    if (cancerType !== "brain") return Boolean(file);
    if (brainUploadMode === "single") return Boolean(file);
    if (folderFiles.length === 0) return false;
    if (folderAnalysis.kind === "mri_brats" && mriBraTSFolderComplete(folderAnalysis.mriSequences)) {
      return true;
    }
    // CT + unknown: do not submit until CT / spec is wired (see flareAPI comment block)
    return false;
  }, [canUpload, cancerType, brainUploadMode, file, folderFiles.length, folderAnalysis]);

  const scanDateToday = useMemo(() => new Date().toISOString().split("T")[0], []);

  const onFileChosen = useCallback((f: File | null) => {
    setFile(f);
    setFolderFiles([]);
    setFolderInputKey((k) => k + 1);
    setScanResult(null);
    setError("");
  }, []);

  const onFolderChosen = useCallback((list: FileList | File[] | null) => {
    const arr = list ? Array.from(list) : [];
    setFolderFiles(arr);
    setFile(null);
    setImagePreviewUrl(null);
    setScanResult(null);
    setError("");
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (cancerType === "brain" && brainUploadMode === "folder") {
        const all = Array.from(e.dataTransfer.files || []);
        if (all.length > 0) onFolderChosen(all);
        return;
      }
      const f = e.dataTransfer.files?.[0];
      if (f && /\.(png|jpe?g|nii(\.gz)?)$/i.test(f.name)) onFileChosen(f);
    },
    [cancerType, brainUploadMode, onFileChosen, onFolderChosen]
  );

  async function onRun() {
    setError("");
    setScanResult(null);
    if (!cancerType) return setError("Please select a cancer type.");
    if (cancerType === "brain" && brainUploadMode === "folder") {
      if (!mriBraTSFolderComplete(folderAnalysis.mriSequences)) {
        return setError("Select a patient folder with all four BraTS sequences (t1n, t1c, t2w, t2f).");
      }
    } else if (!file) {
      return setError("Please upload a scan.");
    }

    setLoading(true);
    try {
      if (cancerType === "brain" && brainUploadMode === "folder") {
        const m = folderAnalysis.mriSequences;
        const folderOutcome = await predictMriBraTSFolder({
          sequences: {
            t1n: m.t1n!,
            t1c: m.t1c!,
            t2w: m.t2w!,
            t2f: m.t2f!,
          },
          hospitalId,
          firstName,
          lastName,
          dob,
          medicalId,
        });
        setScanResult(folderOutcome);
      } else {
        const pred = await predictScan({
          cancerType,
          file: file!,
          hospitalId,
          firstName,
          lastName,
          dob,
          medicalId,
        });
        setScanResult({ kind: "legacy", pred });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const predLegacy = scanResult?.kind === "legacy" ? scanResult.pred : null;
  const mriV2 = scanResult?.kind === "mri_api_v2" ? scanResult : null;

  const showGeoWarning = Boolean(
    (predLegacy && (predLegacy.prediction === "Malignant" || predLegacy.prediction === "Benign")) ||
      (mriV2 && mriV2.classification.label.toLowerCase() === "abnormal")
  );

  const theme = predLegacy
    ? predictionTheme(predLegacy.prediction)
    : mriV2
      ? predictionTheme(mriV2.classification.label.toLowerCase() === "abnormal" ? "Malignant" : "Normal")
      : predictionTheme("Normal");

  return (
    <Box
      sx={{
        width: "100vw",
        minHeight: "100vh",
        px: { xs: 2, md: 6 },
        py: 4,
        color: "#fff",
        background: "radial-gradient(circle at bottom right, #1b2335 0%, #0b0f19 60%)",
      }}
    >
      <Typography sx={{ fontSize: "1.7rem", fontWeight: 800, mb: 1, letterSpacing: "0.02em" }}>
        AI Cancer Detection
      </Typography>
      <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 3 }}>
        Enter patient data, select cancer type, then upload a scan to run the appropriate model.
      </Typography>

      <Card sx={cardSx}>
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
              gap: 2,
            }}
          >
            <TextField select label="Hospital" value={hospitalId} onChange={(e) => setHospitalId(e.target.value)} sx={fieldSx}>
              {HOSPITALS.map((h) => (
                <MenuItem key={h.id} value={h.id}>
                  {h.name}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              select
              label="Cancer Type"
              value={cancerType}
              onChange={(e) => {
                setCancerType(e.target.value as CancerType);
                setFile(null);
                setFolderFiles([]);
                setFolderInputKey((k) => k + 1);
                setBrainUploadMode("single");
                setScanResult(null);
              }}
              SelectProps={{
                renderValue: (v) => (v === "brain" ? "Brain" : v === "breast" ? "Breast" : ""),
              }}
              sx={fieldSx}
            >
              <MenuItem value="brain">Brain</MenuItem>
              <MenuItem value="breast">
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  Breast
                  <Chip
                    label="Coming Soon"
                    size="small"
                    sx={{
                      height: 22,
                      fontSize: "0.7rem",
                      backgroundColor: "rgba(255,92,92,0.15)",
                      border: "1px solid rgba(255,92,92,0.35)",
                      color: "#ff5c5c",
                      fontWeight: 700,
                    }}
                  />
                </Box>
              </MenuItem>
            </TextField>

            <TextField label="First Name" value={firstName} onChange={(e) => setFirstName(e.target.value)} sx={fieldSx} />
            <TextField label="Last Name" value={lastName} onChange={(e) => setLastName(e.target.value)} sx={fieldSx} />
            <TextField label="Medical ID" value={medicalId} onChange={(e) => setMedicalId(e.target.value)} sx={fieldSx} />
            <TextField
              label="Date of Birth"
              type="date"
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={fieldSx}
            />
          </Box>

          <Divider sx={{ borderColor: "rgba(255,255,255,0.08)", my: 3 }} />

          {canUpload ? (
            <Box>
              <Typography sx={{ mb: 1.2, color: "rgba(255,255,255,0.8)", fontWeight: 700 }}>
                Upload {cancerType === "brain" ? "Brain (MRI/CT)" : "Breast (Mammography/Ultrasound)"} Scan
              </Typography>

              {cancerType === "brain" && (
                <ToggleButtonGroup
                  exclusive
                  value={brainUploadMode}
                  onChange={(_, v) => {
                    if (v == null) return;
                    setBrainUploadMode(v);
                    setFile(null);
                    setFolderFiles([]);
                    setFolderInputKey((k) => k + 1);
                    setScanResult(null);
                    setError("");
                  }}
                  sx={{
                    mb: 2,
                    "& .MuiToggleButton-root": {
                      color: "rgba(255,255,255,0.75)",
                      textTransform: "none",
                      borderColor: "rgba(255,255,255,0.2)",
                    },
                    "& .MuiToggleButton-root.Mui-selected": {
                      backgroundColor: "rgba(255,92,92,0.22)",
                      color: "#fff",
                    },
                  }}
                >
                  <ToggleButton value="single">Single scan</ToggleButton>
                  <ToggleButton value="folder">Patient folder</ToggleButton>
                </ToggleButtonGroup>
              )}

              <Box
                component="label"
                htmlFor={cancerType === "brain" && brainUploadMode === "folder" ? "flare-folder-upload" : "flare-scan-upload"}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
                onDrop={onDrop}
                sx={{
                  display: "block",
                  cursor: "pointer",
                  border: "1px dashed rgba(255,255,255,0.15)",
                  borderRadius: 2,
                  py: 4,
                  px: 2,
                  textAlign: "center",
                  backgroundColor: "rgba(255,255,255,0.03)",
                  transition: "background-color 0.2s",
                  "&:hover": { backgroundColor: "rgba(255,255,255,0.05)" },
                }}
              >
                <Typography sx={{ color: "rgba(255,255,255,0.75)" }}>
                  {cancerType === "brain" && brainUploadMode === "folder"
                    ? "Click to choose a folder or drag files here"
                    : "Click to upload or drag and drop"}
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.85rem", mt: 0.5 }}>
                  {cancerType === "brain" && brainUploadMode === "folder"
                    ? "BraTS-style: *-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz ( *-seg.nii.gz ignored )"
                    : "JPG/PNG or NIfTI (.nii / .nii.gz) for brain — volume → one slice on server"}
                </Typography>
                {cancerType === "brain" && brainUploadMode === "folder" && folderFiles.length > 0 && (
                  <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.9rem" }}>
                    {folderFiles.length} file{folderFiles.length !== 1 ? "s" : ""} in folder
                  </Typography>
                )}
                {!(cancerType === "brain" && brainUploadMode === "folder") && file && (
                  <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.9rem" }}>
                    {file.name}
                  </Typography>
                )}
                {cancerType === "brain" && brainUploadMode === "folder" ? (
                  <input
                    key={folderInputKey}
                    id="flare-folder-upload"
                    type="file"
                    // webkitdirectory omits accept in many browsers — rely on client-side detection
                    {...({ webkitdirectory: "", directory: "" } as InputHTMLAttributes<HTMLInputElement>)}
                    multiple
                    style={{ display: "none" }}
                    onChange={(e) => onFolderChosen(e.target.files)}
                  />
                ) : (
                  <input
                    id="flare-scan-upload"
                    type="file"
                    accept=".jpg,.jpeg,.png,.nii,.nii.gz"
                    style={{ display: "none" }}
                    onChange={(e) => onFileChosen(e.target.files?.[0] ?? null)}
                  />
                )}
              </Box>

              {cancerType === "brain" && brainUploadMode === "folder" && folderFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1.5 }}>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", fontWeight: 600 }}>Detected modality</Typography>
                    <Chip
                      label={
                        folderAnalysis.kind === "mri_brats"
                          ? "MRI (BraTS-style)"
                          : folderAnalysis.kind === "ct"
                            ? "CT (preview)"
                            : "Unknown"
                      }
                      size="small"
                      sx={{
                        fontWeight: 700,
                        backgroundColor:
                          folderAnalysis.kind === "mri_brats"
                            ? "rgba(155,177,255,0.2)"
                            : folderAnalysis.kind === "ct"
                              ? "rgba(251,191,36,0.15)"
                              : "rgba(255,255,255,0.08)",
                        color: folderAnalysis.kind === "mri_brats" ? "#9bb1ff" : "#fff",
                        border: "1px solid rgba(255,255,255,0.12)",
                      }}
                    />
                  </Box>
                  {folderAnalysis.ignoredSegCount > 0 && (
                    <Typography sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.8rem", mb: 1 }}>
                      Ignored {folderAnalysis.ignoredSegCount} label file(s) (*-seg.nii.gz)
                    </Typography>
                  )}

                  {folderAnalysis.kind === "mri_brats" && (
                    <Box sx={{ ...cardSx, p: 2, maxWidth: 480, textAlign: "left" }}>
                      <Typography sx={{ color: "#9bb1ff", fontWeight: 700, mb: 1 }}>Required MRI sequences</Typography>
                      {MRI_SEQUENCE_KEYS.map((key) => {
                        const ok = folderAnalysis.mriSequences[key] != null;
                        const label = MRI_SEQUENCE_LABELS[key];
                        return (
                          <Typography
                            key={key}
                            sx={{
                              fontSize: "0.9rem",
                              color: ok ? "rgba(134,239,172,0.95)" : "rgba(252,165,165,0.95)",
                              mb: 0.5,
                            }}
                          >
                            {ok ? "✓" : "○"} {label}{" "}
                            <Typography component="span" sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem" }}>
                              (*-{key}.nii.gz)
                            </Typography>
                            {ok ? "" : " — missing"}
                          </Typography>
                        );
                      })}
                      <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.8rem", mt: 1.5 }}>
                        Submit sends <code style={{ color: "rgba(155,177,255,0.9)" }}>POST /api/mri/predict</code> with{" "}
                        <code style={{ color: "rgba(155,177,255,0.9)" }}>modality=brain_brats</code> and multipart fields{" "}
                        <code style={{ color: "rgba(155,177,255,0.9)" }}>t1n, t1c, t2w, t2f</code> plus patient fields.
                        Relative image URLs in the response are resolved against{" "}
                        <code style={{ color: "rgba(155,177,255,0.9)" }}>{API_BASE}</code>.
                      </Typography>
                    </Box>
                  )}

                  {folderAnalysis.kind === "ct" && (
                    <Box sx={{ ...cardSx, p: 2, maxWidth: 480, textAlign: "left" }}>
                      <Typography sx={{ color: "#fbbf24", fontWeight: 700, mb: 1 }}>CT-style files detected</Typography>
                      <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", mb: 1 }}>
                        Names matching <code>*-ct.nii.gz</code> (contract TBD by CT team).
                      </Typography>
                      {folderAnalysis.ctCandidates.map((f) => (
                        <Typography key={`${f.name}-${f.size}`} sx={{ fontSize: "0.85rem", wordBreak: "break-all" }}>
                          • {f.name} ({(f.size / 1024 / 1024).toFixed(1)} MB)
                        </Typography>
                      ))}
                      {/*
                        FUTURE: when CT checkpoint + /api/ct/predict are ready, enable Submit and call
                        predictCtPatientFolder(...) from flareAPI.ts (see commented stub there).
                      */}
                      <Alert severity="warning" sx={{ mt: 2, backgroundColor: "rgba(251,191,36,0.08)", color: "#fff" }}>
                        CT folder inference is not enabled yet. Endpoint and filename rules will ship with the CT model.
                      </Alert>
                    </Box>
                  )}

                  {folderAnalysis.kind === "unknown" && (
                    <Alert severity="info" sx={{ backgroundColor: "rgba(255,255,255,0.06)", color: "#fff" }}>
                      No BraTS MRI pattern (<code>*-t1n.nii.gz</code>) and no CT pattern (<code>*-ct.nii.gz</code>) found.
                      Add the expected NIfTI names or use single-scan mode.
                    </Alert>
                  )}
                </Box>
              )}

              {cancerType === "brain" && brainUploadMode === "single" && file && isNiftiFile(file) && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Uploaded Scan</Typography>
                  <Box
                    sx={{
                      ...cardSx,
                      maxWidth: 400,
                      p: 2,
                      textAlign: "left",
                    }}
                  >
                    <Typography sx={{ color: "#9bb1ff", fontWeight: 700, mb: 1 }}>NIfTI Volume</Typography>
                    <Typography sx={{ fontSize: "0.9rem", wordBreak: "break-all", mb: 0.5 }}>{file.name}</Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", mb: 1.5 }}>
                      {(file.size / 1024 / 1024).toFixed(1)} MB
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.85rem" }}>
                      Best axial slice will be auto-extracted on Delta
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.85rem", mt: 0.5 }}>
                      BRISC classification + segmentation pipeline
                    </Typography>
                  </Box>
                </Box>
              )}

              {imagePreviewUrl && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Uploaded Scan</Typography>
                  <Box
                    component="img"
                    src={imagePreviewUrl}
                    alt="Uploaded scan preview"
                    sx={{
                      maxWidth: 400,
                      width: "100%",
                      borderRadius: 2,
                      border: "1px solid rgba(255,255,255,0.12)",
                      display: "block",
                    }}
                  />
                </Box>
              )}

              {loading && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 3, justifyContent: "center" }}>
                  <CircularProgress size={28} sx={{ color: "#ff5c5c" }} />
                  <Typography sx={{ color: "rgba(255,255,255,0.85)" }}>
                    Running AI inference on Delta...
                  </Typography>
                </Box>
              )}

              <Button
                variant="contained"
                fullWidth
                disabled={!canSubmit || loading}
                onClick={onRun}
                sx={{
                  mt: 3,
                  backgroundColor: "#ff5c5c",
                  textTransform: "none",
                  borderRadius: 2,
                  py: 1.4,
                  "&:hover": { backgroundColor: "#ff3b3b" },
                }}
              >
                Run AI Scan
              </Button>
            </Box>
          ) : (
            <Alert severity="info" sx={{ backgroundColor: "rgba(255,255,255,0.05)", color: "#fff" }}>
              Fill out patient information and select a cancer type to enable upload.
            </Alert>
          )}

          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {mriV2 && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", mb: 2, color: "#fff" }}>
              Results — MRI (/api/mri/predict)
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                gap: 3,
                alignItems: "start",
              }}
            >
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <Typography sx={{ color: "#9bb1ff", fontWeight: 700, fontSize: "0.95rem" }}>
                  Primary read (MRI + tumor blend)
                </Typography>
                {mriV2.segmentation.overlay_url ? (
                  <Box
                    component="img"
                    src={mriV2.segmentation.overlay_url}
                    alt="Fusion overlay"
                    sx={{
                      maxWidth: "100%",
                      width: "100%",
                      borderRadius: 2,
                      border: "1px solid rgba(255,255,255,0.12)",
                      display: "block",
                    }}
                  />
                ) : mriV2.segmentation.original_url ? (
                  <Box sx={{ position: "relative" }}>
                    {mriV2.classification.label.toLowerCase() === "normal" && (
                      <Alert
                        severity="success"
                        sx={{
                          mb: 1,
                          backgroundColor: "rgba(34,197,94,0.15)",
                          color: "#86efac",
                          border: "1px solid rgba(34,197,94,0.35)",
                          "& .MuiAlert-icon": { color: "#22c55e" },
                        }}
                      >
                        No tumor detected
                      </Alert>
                    )}
                    <Box
                      component="img"
                      src={mriV2.segmentation.original_url}
                      alt="Original MRI"
                      sx={{
                        maxWidth: "100%",
                        width: "100%",
                        borderRadius: 2,
                        border: "1px solid rgba(255,255,255,0.12)",
                        display: "block",
                      }}
                    />
                  </Box>
                ) : (
                  <Box sx={placeholderBoxSx}>
                    <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>No image URL in response</Typography>
                  </Box>
                )}

                {mriV2.segmentation.mask_url && (
                  <Box sx={{ mt: 1 }}>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.75, fontSize: "0.85rem" }}>
                      Tumor Mask
                    </Typography>
                    <Box
                      component="img"
                      src={mriV2.segmentation.mask_url}
                      alt="Tumor mask"
                      sx={{
                        maxWidth: 200,
                        width: "100%",
                        borderRadius: 1,
                        border: "1px solid rgba(255,255,255,0.12)",
                        display: "block",
                      }}
                    />
                  </Box>
                )}
              </Box>

              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {showGeoWarning && (
                  <Alert
                    severity="warning"
                    sx={{
                      backgroundColor: "rgba(234,179,8,0.12)",
                      color: "#fff",
                      border: "1px solid rgba(234,179,8,0.35)",
                      "& .MuiAlert-icon": { color: "#eab308" },
                    }}
                  >
                    Case submitted to Geo Tracker — pending radiologist review
                  </Alert>
                )}

                <Box
                  sx={{
                    p: 2.5,
                    borderRadius: 2,
                    backgroundColor: theme.bg,
                    border: `1px solid ${theme.border}`,
                    textAlign: "center",
                  }}
                >
                  <Typography
                    sx={{
                      fontSize: "2rem",
                      fontWeight: 900,
                      color:
                        mriV2.classification.label.toLowerCase() === "abnormal" ? "#f87171" : "#86efac",
                    }}
                  >
                    {mriV2.classification.label}
                  </Typography>
                </Box>

                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>Confidence</Typography>
                  <Typography sx={{ fontSize: "1.75rem", fontWeight: 800, color: theme.text, mb: 1 }}>
                    {(mriV2.classification.confidence * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, Math.max(0, mriV2.classification.confidence * 100))}
                    sx={{
                      height: 10,
                      borderRadius: 1,
                      backgroundColor: "rgba(255,255,255,0.08)",
                      "& .MuiLinearProgress-bar": {
                        backgroundColor: theme.bar,
                        borderRadius: 1,
                      },
                    }}
                  />
                </Box>

                {(mriV2.segmentation.tumor_pixel_count != null || mriV2.classification.tumor_px != null) && (
                  <Typography sx={{ color: "rgba(255,255,255,0.8)", fontSize: "0.95rem" }}>
                    Affected area:{" "}
                    <b style={{ color: "#fca5a5" }}>
                      {(mriV2.segmentation.tumor_pixel_count ?? mriV2.classification.tumor_px)?.toLocaleString()} px
                    </b>
                  </Typography>
                )}

                <Card sx={{ ...cardSx, boxShadow: "none" }}>
                  <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                    <Typography sx={{ fontWeight: 700, mb: 1, color: "#9bb1ff", fontSize: "0.95rem" }}>
                      Patient summary
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.88rem", lineHeight: 1.7 }}>
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Name:</b> {firstName} {lastName}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Medical ID:</b> {medicalId}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Hospital:</b> {hospitalName(hospitalId)}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>DOB:</b> {dob}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Scan date:</b> {scanDateToday}
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {predLegacy && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", mb: 2, color: "#fff" }}>
              Results
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                gap: 3,
                alignItems: "start",
              }}
            >
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Original Scan</Typography>
                  {imagePreviewUrl ? (
                    <Box
                      component="img"
                      src={imagePreviewUrl}
                      alt="Original"
                      sx={{
                        maxWidth: 400,
                        width: "100%",
                        borderRadius: 2,
                        border: "1px solid rgba(255,255,255,0.12)",
                        display: "block",
                      }}
                    />
                  ) : (
                    <Box sx={placeholderBoxSx}>
                      <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>No scan loaded</Typography>
                    </Box>
                  )}
                </Box>

                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Segmentation Overlay</Typography>
                  {predLegacy.localization_url ? (
                    <Box
                      component="img"
                      src={predLegacy.localization_url}
                      alt="Segmentation"
                      sx={{
                        maxWidth: 400,
                        width: "100%",
                        borderRadius: 2,
                        border: "1px solid rgba(255,255,255,0.12)",
                        display: "block",
                      }}
                    />
                  ) : (
                    <Box sx={placeholderBoxSx}>
                      <Typography sx={{ color: "rgba(255,255,255,0.55)" }}>
                        Segmentation overlay not available for this result
                      </Typography>
                    </Box>
                  )}
                </Box>

                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Grad-CAM</Typography>
                  <Box sx={placeholderBoxSx}>
                    <Typography sx={{ color: "rgba(255,255,255,0.55)" }}>
                      Grad-CAM visualization — coming soon
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem", mt: 1 }}>
                      Will highlight regions most influential to the AI decision
                    </Typography>
                  </Box>
                </Box>
              </Box>

              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {showGeoWarning && (
                  <Alert
                    severity="warning"
                    sx={{
                      backgroundColor: "rgba(234,179,8,0.12)",
                      color: "#fff",
                      border: "1px solid rgba(234,179,8,0.35)",
                      "& .MuiAlert-icon": { color: "#eab308" },
                    }}
                  >
                    Case submitted to Geo Tracker — pending radiologist review
                  </Alert>
                )}

                <Box
                  sx={{
                    p: 2.5,
                    borderRadius: 2,
                    backgroundColor: theme.bg,
                    border: `1px solid ${theme.border}`,
                    textAlign: "center",
                  }}
                >
                  <Typography sx={{ fontSize: "2rem", fontWeight: 900, color: theme.text }}>
                    {predLegacy.prediction}
                  </Typography>
                </Box>

                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>Model Confidence</Typography>
                  <Typography sx={{ fontSize: "1.75rem", fontWeight: 800, color: theme.text, mb: 1 }}>
                    {(predLegacy.confidence * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, Math.max(0, predLegacy.confidence * 100))}
                    sx={{
                      height: 10,
                      borderRadius: 1,
                      backgroundColor: "rgba(255,255,255,0.08)",
                      "& .MuiLinearProgress-bar": {
                        backgroundColor: theme.bar,
                        borderRadius: 1,
                      },
                    }}
                  />
                </Box>

                {predLegacy.probabilities && predLegacy.probabilities.length > 0 && (
                  <Box>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1, fontWeight: 700 }}>
                      Class probabilities
                    </Typography>
                    {predLegacy.probabilities.map((row) => (
                      <Box key={row.label} sx={{ mb: 1.2 }}>
                        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.3 }}>
                          <Typography sx={{ color: "rgba(255,255,255,0.85)", fontSize: "0.85rem", textTransform: "capitalize" }}>
                            {row.label.replace(/_/g, " ")}
                          </Typography>
                          <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.85rem" }}>
                            {(row.value * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={Math.min(100, Math.max(0, row.value * 100))}
                          sx={{
                            height: 4,
                            borderRadius: 1,
                            backgroundColor: "rgba(255,255,255,0.06)",
                            "& .MuiLinearProgress-bar": {
                              backgroundColor: theme.bar,
                              opacity: 0.85,
                            },
                          }}
                        />
                      </Box>
                    ))}
                  </Box>
                )}

                <Card sx={{ ...cardSx, boxShadow: "none" }}>
                  <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                    <Typography sx={{ fontWeight: 700, mb: 1, color: "#9bb1ff", fontSize: "0.95rem" }}>
                      Patient summary
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.88rem", lineHeight: 1.7 }}>
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Name:</b> {firstName} {lastName}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Medical ID:</b> {medicalId}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Hospital:</b> {hospitalName(hospitalId)}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>DOB:</b> {dob}
                      <br />
                      <b style={{ color: "rgba(255,255,255,0.5)" }}>Scan date:</b> {scanDateToday}
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
