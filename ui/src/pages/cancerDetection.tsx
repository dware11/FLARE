import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { InputHTMLAttributes } from "react";
import { useNavigate } from "react-router-dom";
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
  Snackbar,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
} from "@mui/material";
import {
  predictScan,
  predictMriBraTSFolder,
  predictCtFile,
  predictFusion,
  API_BASE,
  absolutizeApiAssetUrl,
  type CancerScanResult,
  type CancerType,
  type ResultClass,
} from "../api/flareAPI";
import {
  MRI_SEQUENCE_KEYS,
  MRI_SEQUENCE_LABELS,
  analyzePatientFolder,
  mriBraTSFolderComplete,
} from "../utils/scanFolderModality";
import { useCancerInference } from "../context/CancerInferenceContext";
import { useNgrokImage } from "../hooks/useNgrokImage";

const HOSPITALS = [
  { id: "H001", name: "Houston Methodist Hospital" },
  { id: "H002", name: "Memorial Hermann - Texas Medical Center" },
  { id: "H003", name: "Baylor St. Luke's Medical Center" },
  { id: "H004", name: "Ben Taub Hospital" },
  { id: "H005", name: "Texas Children's Hospital" },
];

const isValidMedicalId = (id: string) => /^P\d{4}$/.test(id);

const ERR_DOB_FUTURE = "Date of birth cannot be in the future.";
const ERR_DOB_MINOR =
  "FLARE prototype currently supports adult patients only. Patient must be at least 18 years old.";

function toYmdLocal(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Latest date of birth (inclusive) for age >= 18: current local date minus 18 years. */
function getAdultDobMaxYmd(d = new Date()): string {
  const t = new Date(d.getTime());
  t.setFullYear(t.getFullYear() - 18);
  return toYmdLocal(t);
}

function getTodayYmdLocal(d = new Date()): string {
  return toYmdLocal(d);
}

/** Display YYYY-MM-DD as MM/DD/YYYY for patient-facing helper text (adult cutoff). */
function formatYmdAsUs(ymd: string): string {
  const m = ymd.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return ymd;
  return `${m[2]}/${m[3]}/${m[1]}`;
}

const ERR_DOB_INVALID = "Please enter a valid date of birth.";

type DobValidation = { valid: true } | { valid: false; message: string };

function validateAdultDob(dob: string): DobValidation {
  const t = dob.trim();
  if (!t) return { valid: false, message: "" };
  if (!/^\d{4}-\d{2}-\d{2}$/.test(t)) return { valid: false, message: ERR_DOB_INVALID };
  const todayY = getTodayYmdLocal();
  if (t > todayY) return { valid: false, message: ERR_DOB_FUTURE };
  const maxAdult = getAdultDobMaxYmd();
  if (t > maxAdult) return { valid: false, message: ERR_DOB_MINOR };
  return { valid: true };
}

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
  "& .MuiOutlinedInput-root": {
    color: "#ffffff",
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 2,
    "& fieldset": { borderColor: "rgba(255,255,255,0.2)" },
    "&:hover fieldset": { borderColor: "rgba(255,255,255,0.4)" },
    "&.Mui-focused fieldset": { borderColor: "#ff5c5c" },
  },
  "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.65)" },
  "& .MuiInputLabel-root.Mui-focused": { color: "#ff5c5c" },
  "& input": { color: "#ffffff" },
  "& .MuiSelect-icon": { color: "rgba(255,255,255,0.65)" },
};

const dobFieldSx = {
  ...fieldSx,
  "& input[type='date']": {
    color: "#ffffff",
    colorScheme: "dark",
  },
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

const resultImgSx = {
  width: "100%",
  minHeight: 220,
  maxHeight: 320,
  objectFit: "cover" as const,
  borderRadius: 2,
  border: "1px solid rgba(255,255,255,0.12)",
  display: "block",
};

function formatMmSs(totalSec: number): string {
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** NIfTI volumes are binary — no <img> preview; server extracts one slice for BRISC. */
function isNiftiFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".nii.gz") || n.endsWith(".nii");
}

function isNpzFile(f: File | null): boolean {
  if (!f) return false;
  return f.name.toLowerCase().endsWith(".npz");
}

/** CT "Patient folder" mode: one NPZ or ZIP of DICOM (stable patient-level path in UI; not browser directory upload). */
function isCtPatientVolumeFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".npz") || n.endsWith(".zip");
}

function isCtSingleSliceFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png");
}

function isFusionCtUploadFile(f: File | null): boolean {
  if (!f) return false;
  const n = f.name.toLowerCase();
  return n.endsWith(".npz") || n.endsWith(".zip") || n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg");
}

type BrainPipeline = "mri" | "ct" | "fusion";

type CtUploadMode = "single" | "patient";

function fusionModeLabel(mode: string): string {
  switch (mode) {
    case "ct_mri":
      return "CT + MRI Fusion";
    case "ct_only":
      return "CT Only";
    case "mri_only":
      return "MRI Only";
    default:
      return mode;
  }
}

const FLARE_LAST_RESULT_KEY = "flare_last_result";
const FLARE_FORM_KEY = "flare_form_state";
const FLARE_AI_NOTICE_KEY = "flare_ai_notice_dismissed";
const REVIEW_DISCLAIMER_TEXT =
  "AI-generated result. This output must be reviewed and confirmed by a qualified clinician.";

const pairImgSx = {
  width: "100%",
  maxWidth: 300,
  borderRadius: 2,
  objectFit: "cover" as const,
  border: "1px solid rgba(255,255,255,0.12)",
  display: "block",
};

const MRI_SEG_OVERLAY_LABEL = "MRI Segmentation Overlay";
const MRI_RAW_MASK_LABEL = "Raw Tumor Mask";

/**
 * For legacy /predict, localization_url is overlay | mask (then gradcam). No explicit field — infer from path.
 * BRISC: *_overlay.png = blend; *_mask.png without overlay = raw mask.
 */
function mriPathLooksLikeMaskOnly(url: string | undefined): boolean {
  if (url == null || url === "") return false;
  const p = url.toLowerCase();
  if (p.includes("overlay")) return false;
  if (p.includes("gradcam") || p.includes("/static/cam/") || p.includes("/cam/")) return false;
  if (p.includes("_mask") || p.includes("mask.png")) return true;
  return false;
}

/** Fusion: MRI localization panel is primary; CT Grad-CAM is supporting. */
const fusionMriOvlBoxSx = { flex: "2 1 200px" as const, minWidth: 0, maxWidth: 440 };
const fusionMriOvlImgSx = {
  width: "100%",
  maxWidth: 440,
  borderRadius: 2,
  objectFit: "cover" as const,
  border: "1px solid rgba(255,255,255,0.12)",
  display: "block",
};

const fusionCtCamBoxSx = { flex: "1 1 140px" as const, minWidth: 0, maxWidth: 240 };
const fusionCtCamImgSx = {
  width: "100%",
  maxWidth: 240,
  borderRadius: 2,
  objectFit: "cover" as const,
  border: "1px solid rgba(255,255,255,0.12)",
  display: "block",
};

const fusionMriInBoxSx = { flex: "1 1 160px" as const, minWidth: 0, maxWidth: 300 };
const fusionMriInImgSx = {
  width: "100%",
  maxWidth: 300,
  borderRadius: 2,
  objectFit: "cover" as const,
  border: "1px solid rgba(255,255,255,0.12)",
  display: "block",
};

export default function CancerDetection() {
  const navigate = useNavigate();
  const { stored, patchStored, clearStored } = useCancerInference();
  const hydratedRef = useRef(false);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyzeSecondsRef = useRef(0);
  const [analyzeSeconds, setAnalyzeSeconds] = useState(0);
  const [completedSeconds, setCompletedSeconds] = useState<number | null>(null);
  const [inferSnackOpen, setInferSnackOpen] = useState(false);
  const [inferSnackMsg, setInferSnackMsg] = useState("");

  const startAnalyzeTimer = useCallback(() => {
    analyzeSecondsRef.current = 0;
    setAnalyzeSeconds(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      analyzeSecondsRef.current += 1;
      setAnalyzeSeconds(analyzeSecondsRef.current);
    }, 1000);
  }, []);

  const stopAnalyzeTimer = useCallback((): number => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    return analyzeSecondsRef.current;
  }, []);

  useEffect(() => {
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, []);

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
  const [brainPipeline, setBrainPipeline] = useState<BrainPipeline>("mri");
  const [ctUploadMode, setCtUploadMode] = useState<CtUploadMode>("patient");
  const [ctFileInputKey, setCtFileInputKey] = useState(0);
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [folderInputKey, setFolderInputKey] = useState(0);
  const [fusionCtFile, setFusionCtFile] = useState<File | null>(null);
  const [fusionMriFile, setFusionMriFile] = useState<File | null>(null);
  const [fusionInputKey, setFusionInputKey] = useState(0);

  const [loading, setLoading] = useState(false);
  /** Legacy `/predict` shape vs newer `/api/mri/predict` classification + segmentation JSON. */
  const [scanResult, setScanResult] = useState<CancerScanResult | null>(null);
  const [ctScanResult, setCtScanResult] = useState<Record<string, unknown> | null>(null);
  const [fusionScanResult, setFusionScanResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [aiNoticeOpen, setAiNoticeOpen] = useState(false);
  const [dontShowAiNotice, setDontShowAiNotice] = useState(false);

  useEffect(() => {
    if (!loading) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [loading]);

  useEffect(() => {
    setScanResult(stored.scanResult);
    setCtScanResult(stored.ctScanResult);
    setFusionScanResult(stored.fusionScanResult);
    setCompletedSeconds(stored.completedSeconds);
  }, [stored]);

  useEffect(() => {
    if (hydratedRef.current) return;
    hydratedRef.current = true;
    try {
      sessionStorage.removeItem(FLARE_LAST_RESULT_KEY);
      sessionStorage.removeItem(FLARE_FORM_KEY);
    } catch { /* ignore */ }
    try {
      localStorage.removeItem(FLARE_LAST_RESULT_KEY);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    try {
      const dismissed = localStorage.getItem(FLARE_AI_NOTICE_KEY);
      if (dismissed !== "true") {
        setAiNoticeOpen(true);
      }
    } catch {
      setAiNoticeOpen(true);
    }
  }, []);

  useEffect(() => {
    if (cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder") {
      setImagePreviewUrl(null);
      return;
    }
    // Only raster scans get a blob URL; NIfTI / NPZ / ZIP would produce a broken <img> src
    if (!file || isNiftiFile(file) || isNpzFile(file) || file.name.toLowerCase().endsWith(".zip")) {
      setImagePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file, brainUploadMode, brainPipeline, cancerType]);

  const folderAnalysis = useMemo(() => analyzePatientFolder(folderFiles), [folderFiles]);

  const dobValidation = useMemo(() => validateAdultDob(dob), [dob]);

  /** Latest allowed DOB for age ≥ 18; recomputed each render so cutoff matches “today”. */
  const adultDobCutoffYmd = getAdultDobMaxYmd();
  const dobGuidanceText = `Adult patient only: select a date on or before ${formatYmdAsUs(adultDobCutoffYmd)}.`;
  const dobShowError = Boolean(dob) && !dobValidation.valid;
  const dobHelperText =
    dobShowError && dobValidation.valid === false && dobValidation.message
      ? dobValidation.message
      : dobGuidanceText;

  const canUpload = useMemo(() => {
    return (
      Boolean(cancerType) &&
      firstName &&
      lastName &&
      medicalId &&
      isValidMedicalId(medicalId) &&
      dob &&
      dobValidation.valid &&
      hospitalId
    );
  }, [cancerType, firstName, lastName, medicalId, dob, dobValidation.valid, hospitalId]);

  const canSubmit = useMemo(() => {
    if (!canUpload) return false;
    if (cancerType !== "brain") return Boolean(file);
    if (brainPipeline === "ct") {
      if (ctUploadMode === "single") return Boolean(file && isCtSingleSliceFile(file));
      return Boolean(file && isCtPatientVolumeFile(file));
    }
    if (brainPipeline === "fusion") return Boolean(fusionCtFile && fusionMriFile);
    if (brainUploadMode === "single") return Boolean(file);
    if (folderFiles.length === 0) return false;
    if (folderAnalysis.kind === "mri_brats" && mriBraTSFolderComplete(folderAnalysis.mriSequences)) {
      return true;
    }
    return false;
  }, [
    canUpload,
    cancerType,
    brainPipeline,
    brainUploadMode,
    file,
    folderFiles.length,
    folderAnalysis,
    fusionCtFile,
    fusionMriFile,
    ctUploadMode,
  ]);

  const scanDateToday = useMemo(() => new Date().toISOString().split("T")[0], []);

  const LOCKED_MSG = "Inference is currently running. Please wait for completion.";

  const onFileChosen = useCallback(
    (f: File | null) => {
      if (loading) return;
      if (cancerType === "brain" && brainPipeline === "ct" && f) {
        if (ctUploadMode === "single" && !isCtSingleSliceFile(f)) {
          setError("Please upload a CT image slice as JPG or PNG.");
          return;
        }
        if (ctUploadMode === "patient" && !isCtPatientVolumeFile(f)) {
          setError("Please upload a CT patient-level NPZ volume or zipped DICOM study.");
          return;
        }
      }
      setFile(f);
      setFolderFiles([]);
      setFolderInputKey((k) => k + 1);
      setScanResult(null);
      setCtScanResult(null);
      setFusionScanResult(null);
      setError("");
    },
    [loading, cancerType, brainPipeline, ctUploadMode]
  );

  const onFolderChosen = useCallback((list: FileList | File[] | null) => {
    if (loading) return;
    const arr = list ? Array.from(list) : [];
    setFolderFiles(arr);
    setFile(null);
    setImagePreviewUrl(null);
    setScanResult(null);
    setCtScanResult(null);
    setFusionScanResult(null);
    setError("");
  }, [loading]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (loading) return;
      if (cancerType === "brain" && brainPipeline === "ct") {
        const f = e.dataTransfer.files?.[0] ?? null;
        if (!f) return;
        if (ctUploadMode === "single") {
          if (isCtSingleSliceFile(f)) onFileChosen(f);
          else setError("Please upload a CT image slice as JPG or PNG.");
        } else {
          if (isCtPatientVolumeFile(f)) onFileChosen(f);
          else setError("Please upload a CT patient-level NPZ volume or zipped DICOM study.");
        }
        return;
      }
      if (cancerType === "brain" && brainUploadMode === "folder" && brainPipeline === "mri") {
        const all = Array.from(e.dataTransfer.files || []);
        if (all.length > 0) onFolderChosen(all);
        return;
      }
      const f = e.dataTransfer.files?.[0];
      const pattern = /\.(png|jpe?g|nii(\.gz)?|npz|zip)$/i;
      if (f && pattern.test(f.name)) onFileChosen(f);
    },
    [loading, cancerType, brainUploadMode, brainPipeline, ctUploadMode, onFileChosen, onFolderChosen]
  );

  const onDropFusionCt = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (loading) return;
    const f = e.dataTransfer.files?.[0];
    if (f && isFusionCtUploadFile(f)) {
      setFusionCtFile(f);
      setCtScanResult(null);
      setFusionScanResult(null);
      setError("");
    }
  }, [loading]);

  const onDropFusionMri = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (loading) return;
    const f = e.dataTransfer.files?.[0];
    if (f && /\.(png|jpe?g|nii(\.gz)?|npz)$/i.test(f.name)) {
      setFusionMriFile(f);
      setCtScanResult(null);
      setFusionScanResult(null);
      setError("");
    }
  }, [loading]);

  const resetPageState = useCallback(() => {
    clearStored();
    setScanResult(null);
    setCtScanResult(null);
    setFusionScanResult(null);
    setCompletedSeconds(null);
    setError("");
    setHospitalId("H001");
    setCancerType("");
    setFirstName("");
    setLastName("");
    setMedicalId("");
    setDob("");
    setFile(null);
    setFolderFiles([]);
    setFolderInputKey((k) => k + 1);
    setBrainPipeline("mri");
    setBrainUploadMode("single");
    setCtUploadMode("patient");
    setCtFileInputKey((k) => k + 1);
    setFusionCtFile(null);
    setFusionMriFile(null);
    setFusionInputKey((k) => k + 1);
    try {
      sessionStorage.removeItem(FLARE_LAST_RESULT_KEY);
      sessionStorage.removeItem(FLARE_FORM_KEY);
    } catch { /* ignore */ }
  }, [clearStored]);

  useEffect(() => {
    const onRefreshApp = () => {
      if (loading) {
        setError(LOCKED_MSG);
        return;
      }
      resetPageState();
    };
    window.addEventListener("flare:refresh-app", onRefreshApp);
    return () => window.removeEventListener("flare:refresh-app", onRefreshApp);
  }, [loading, resetPageState, LOCKED_MSG]);

  async function onRun() {
    setError("");
    try {
      sessionStorage.removeItem(FLARE_LAST_RESULT_KEY);
      sessionStorage.removeItem(FLARE_FORM_KEY);
    } catch {
      /* ignore */
    }
    clearStored();
    setScanResult(null);
    setCtScanResult(null);
    setFusionScanResult(null);
    setCompletedSeconds(null);
    if (!cancerType) return setError("Please select a cancer type.");
    const dcheck = validateAdultDob(dob);
    if (!dcheck.valid) {
      return setError(
        dcheck.message && dcheck.message.length > 0
          ? dcheck.message
          : "Please enter a valid date of birth."
      );
    }

    if (cancerType === "brain" && brainPipeline === "fusion") {
      if (!fusionCtFile || !fusionMriFile) {
        return setError("Upload both CT (.npz/.zip/.jpg/.jpeg/.png) and MRI scans.");
      }
      if (!isFusionCtUploadFile(fusionCtFile)) {
        return setError("CT file must be .npz, .zip, .jpg, .jpeg, or .png.");
      }
      setLoading(true);
      startAnalyzeTimer();
      try {
        const r = await predictFusion(fusionCtFile, fusionMriFile, medicalId, hospitalId);
        setFusionScanResult(r);
        const elapsed = stopAnalyzeTimer();
        setCompletedSeconds(elapsed);
        patchStored({
          fusionScanResult: r,
          scanResult: null,
          ctScanResult: null,
          completedSeconds: elapsed,
        });
        setInferSnackMsg(`Inference complete — model finished in ${elapsed}s`);
        setInferSnackOpen(true);
      } catch (e: unknown) {
        stopAnalyzeTimer();
        setError(e instanceof Error ? e.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
      return;
    }

    if (cancerType === "brain" && brainPipeline === "ct") {
      if (ctUploadMode === "single" && (!file || !isCtSingleSliceFile(file))) {
        return setError("Please upload a CT image slice as JPG or PNG.");
      }
      if (ctUploadMode === "patient" && (!file || !isCtPatientVolumeFile(file))) {
        return setError("Please upload a CT patient-level NPZ volume or zipped DICOM study.");
      }
      const ctFile = file;
      if (!ctFile) return;
      setLoading(true);
      startAnalyzeTimer();
      try {
        const r = await predictCtFile(ctFile, medicalId, hospitalId, firstName, lastName, dob);
        setCtScanResult(r);
        const elapsed = stopAnalyzeTimer();
        setCompletedSeconds(elapsed);
        patchStored({
          ctScanResult: r,
          scanResult: null,
          fusionScanResult: null,
          completedSeconds: elapsed,
        });
        setInferSnackMsg(`Inference complete — model finished in ${elapsed}s`);
        setInferSnackOpen(true);
      } catch (e: unknown) {
        stopAnalyzeTimer();
        setError(e instanceof Error ? e.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
      return;
    }

    if (cancerType === "brain" && brainUploadMode === "folder") {
      if (!mriBraTSFolderComplete(folderAnalysis.mriSequences)) {
        return setError("Select a patient folder with all four BraTS sequences (t1n, t1c, t2w, t2f).");
      }
    } else if (!file) {
      return setError("Please upload a scan.");
    }

    setLoading(true);
    startAnalyzeTimer();
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
        const elapsed = stopAnalyzeTimer();
        setCompletedSeconds(elapsed);
        patchStored({
          scanResult: folderOutcome,
          ctScanResult: null,
          fusionScanResult: null,
          completedSeconds: elapsed,
        });
        setInferSnackMsg(`Inference complete — model finished in ${elapsed}s`);
        setInferSnackOpen(true);
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
        const legacy: CancerScanResult = { kind: "legacy", pred };
        setScanResult(legacy);
        const elapsed = stopAnalyzeTimer();
        setCompletedSeconds(elapsed);
        patchStored({
          scanResult: legacy,
          ctScanResult: null,
          fusionScanResult: null,
          completedSeconds: elapsed,
        });
        setInferSnackMsg(`Inference complete — model finished in ${elapsed}s`);
        setInferSnackOpen(true);
      }
    } catch (e: unknown) {
      stopAnalyzeTimer();
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const predLegacy = scanResult?.kind === "legacy" ? scanResult.pred : null;
  const mriV2 = scanResult?.kind === "mri_api_v2" ? scanResult : null;
  const hasAnyResult = Boolean(mriV2 || predLegacy || ctScanResult || fusionScanResult);

  const reviewCaseId = useMemo(() => {
    const fromFusion = fusionScanResult?.caseId;
    if (typeof fromFusion === "string" && fromFusion.trim() !== "") return fromFusion;

    const fromCt = ctScanResult?.caseId;
    if (typeof fromCt === "string" && fromCt.trim() !== "") return fromCt;

    const fromMri = mriV2?.caseId;
    if (typeof fromMri === "string" && fromMri.trim() !== "") return fromMri;

    const fromLegacy = predLegacy?.caseId;
    if (typeof fromLegacy === "string" && fromLegacy.trim() !== "") return fromLegacy;

    return null;
  }, [fusionScanResult, ctScanResult, mriV2, predLegacy]);

  const reviewRequired = useMemo(() => {
    if (typeof fusionScanResult?.review_required === "boolean") return fusionScanResult.review_required;
    if (typeof ctScanResult?.review_required === "boolean") return ctScanResult.review_required;
    if (typeof mriV2?.review_required === "boolean") return mriV2.review_required;
    if (typeof predLegacy?.review_required === "boolean") return predLegacy.review_required;
    return Boolean(
      (predLegacy && (predLegacy.prediction === "Malignant" || predLegacy.prediction === "Benign")) ||
      (mriV2 && mriV2.classification.label.toLowerCase() === "abnormal")
    );
  }, [fusionScanResult, ctScanResult, mriV2, predLegacy]);

  const onDismissAiNotice = useCallback(() => {
    try {
      if (dontShowAiNotice) {
        localStorage.setItem(FLARE_AI_NOTICE_KEY, "true");
      }
    } catch {
      // ignore storage failures; just continue
    }
    setAiNoticeOpen(false);
  }, [dontShowAiNotice]);

  const showGeoWarning = Boolean(
    (predLegacy && (predLegacy.prediction === "Malignant" || predLegacy.prediction === "Benign")) ||
      (mriV2 && mriV2.classification.label.toLowerCase() === "abnormal")
  );

  const theme = predLegacy
    ? predictionTheme(predLegacy.prediction)
    : mriV2
      ? predictionTheme(mriV2.classification.label.toLowerCase() === "abnormal" ? "Malignant" : "Normal")
      : predictionTheme("Normal");

  const mriOrigPath = mriV2 ? mriV2.input_image_url || mriV2.segmentation.original_url || undefined : undefined;
  const mriOvlPath =
    mriV2?.segmentation?.overlay_url ?? mriV2?.segmentation?.mask_url ?? undefined;
  const mriMaskPath = mriV2?.segmentation?.mask_url ?? undefined;
  const mriHasSeparateMaskPanel = Boolean(
    mriV2?.segmentation?.overlay_url && mriV2?.segmentation?.mask_url
  );
  const mriV2SecondColumnLabel = useMemo(() => {
    if (!mriV2?.segmentation) return MRI_SEG_OVERLAY_LABEL;
    if (mriV2.segmentation.overlay_url) return MRI_SEG_OVERLAY_LABEL;
    if (mriV2.segmentation.mask_url) return MRI_RAW_MASK_LABEL;
    return MRI_SEG_OVERLAY_LABEL;
  }, [mriV2]);

  const mriOrigSrc = useNgrokImage(mriOrigPath);
  const mriOvlSrc = useNgrokImage(mriOvlPath);
  const mriMaskSrc = useNgrokImage(mriMaskPath);

  const legacyLocPath = predLegacy?.localization_url ?? undefined;
  const legacySecondColumnLabel = useMemo(
    () =>
      legacyLocPath == null
        ? MRI_SEG_OVERLAY_LABEL
        : mriPathLooksLikeMaskOnly(legacyLocPath)
          ? MRI_RAW_MASK_LABEL
          : MRI_SEG_OVERLAY_LABEL,
    [legacyLocPath]
  );
  const legacyLocSrc = useNgrokImage(legacyLocPath);

  const ctCamPath = ctScanResult ? (ctScanResult.cam_url as string | undefined) : undefined;
  const ctGradCamSrc = useNgrokImage(ctCamPath);

  const fusionCtCamPath = useMemo(() => {
    if (!fusionScanResult) return undefined;
    const top = fusionScanResult.ct_cam_url as string | undefined;
    if (top) return top;
    const ctD = fusionScanResult.ct_details as Record<string, unknown> | undefined;
    return typeof ctD?.cam_url === "string" && ctD.cam_url ? (ctD.cam_url as string) : undefined;
  }, [fusionScanResult]);

  const fusionMriOvlPath = useMemo(() => {
    if (!fusionScanResult) return undefined;
    const top = fusionScanResult.mri_overlay_url as string | undefined;
    if (top) return top;
    const m = fusionScanResult.mri_details as Record<string, unknown> | undefined;
    const seg = m?.segmentation as Record<string, unknown> | undefined;
    if (typeof seg?.overlay_url === "string" && seg.overlay_url) return seg.overlay_url as string;
    if (typeof seg?.mask_url === "string" && seg.mask_url) return seg.mask_url as string;
    return undefined;
  }, [fusionScanResult]);

  const fusionMriColumnLabel = useMemo(() => {
    if (!fusionScanResult) return MRI_SEG_OVERLAY_LABEL;
    if (fusionScanResult.mri_overlay_url) return MRI_SEG_OVERLAY_LABEL;
    const m = fusionScanResult.mri_details as Record<string, unknown> | undefined;
    const seg = m?.segmentation as Record<string, unknown> | undefined;
    if (typeof seg?.overlay_url === "string" && seg.overlay_url) return MRI_SEG_OVERLAY_LABEL;
    if (typeof seg?.mask_url === "string" && seg.mask_url) return MRI_RAW_MASK_LABEL;
    return MRI_SEG_OVERLAY_LABEL;
  }, [fusionScanResult]);

  const fusionMriInPath = fusionScanResult
    ? (fusionScanResult.mri_input_url as string | undefined)
    : undefined;

  const fusionMriClassLine = useMemo(() => {
    if (!fusionScanResult) return null;
    const m = fusionScanResult.mri_details as Record<string, unknown> | undefined;
    if (!m) return null;
    const pl = m.pred_label ?? m.result_class;
    const c = m.confidence;
    const confNum = typeof c === "number" ? c : c != null ? Number(c) : NaN;
    const predStr = pl != null && String(pl).length > 0 ? String(pl) : "—";
    const confStr = Number.isFinite(confNum) ? `${(confNum * 100).toFixed(1)}%` : "—";
    return { predStr, confStr };
  }, [fusionScanResult]);

  /** Final fused abnormality: API is_abnormal > score vs threshold > result_class / pred_label (not MRI class alone). */
  const fusionDecisionAbnormal = useMemo(() => {
    if (!fusionScanResult) return false;
    if (typeof fusionScanResult.is_abnormal === "boolean") {
      return fusionScanResult.is_abnormal;
    }
    const t = Number(fusionScanResult.threshold ?? 0.5);
    const s = Number(fusionScanResult.fusion_score ?? 0);
    if (Number.isFinite(s) && Number.isFinite(t)) {
      return s >= t;
    }
    const ehr = String(fusionScanResult.result_class ?? "");
    if (ehr === "Malignant" || ehr === "Abnormal") return true;
    if (ehr === "Normal" || ehr === "Benign") return false;
    return String(fusionScanResult.pred_label ?? "") === "Abnormal";
  }, [fusionScanResult]);

  const fusionStatusCopy = fusionDecisionAbnormal
    ? "Abnormal / Review Required"
    : "Normal / Not Flagged";

  const fusionFinalTheme = predictionTheme(fusionDecisionAbnormal ? "Malignant" : "Normal");

  const fusionCtGradCamSrc = useNgrokImage(fusionCtCamPath);
  const fusionMriOrigSrc = useNgrokImage(fusionMriInPath);
  const fusionMriSegSrc = useNgrokImage(fusionMriOvlPath);

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

      <Dialog
        open={aiNoticeOpen}
        onClose={onDismissAiNotice}
        PaperProps={{
          sx: {
            backgroundColor: "#0f1117",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 2,
            minWidth: { xs: 300, sm: 500 },
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 800 }}>AI Assistance Notice</DialogTitle>
        <DialogContent>
          <Typography sx={{ color: "rgba(255,255,255,0.8)", lineHeight: 1.65 }}>
            FLARE uses AI to analyze uploaded medical images. Results are decision-support only
            and must be reviewed and confirmed by a qualified clinician before clinical use.
          </Typography>
          <FormControlLabel
            sx={{ mt: 2, color: "rgba(255,255,255,0.8)" }}
            control={
              <Checkbox
                checked={dontShowAiNotice}
                onChange={(e) => setDontShowAiNotice(e.target.checked)}
                sx={{ color: "rgba(255,255,255,0.7)" }}
              />
            }
            label="Don’t show this again"
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            variant="contained"
            onClick={onDismissAiNotice}
            sx={{
              backgroundColor: "#ff5c5c",
              textTransform: "none",
              "&:hover": { backgroundColor: "#ff3b3b" },
            }}
          >
            Continue
          </Button>
        </DialogActions>
      </Dialog>

      <Card sx={cardSx}>
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
              gap: 2,
            }}
          >
            <TextField select label="Hospital" disabled={loading} value={hospitalId} onChange={(e) => setHospitalId(e.target.value)} sx={fieldSx}>
              {HOSPITALS.map((h) => (
                <MenuItem key={h.id} value={h.id}>
                  {h.name}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              select
              label="Cancer Type"
              disabled={loading}
              value={cancerType}
              onChange={(e) => {
                if (loading) { setError(LOCKED_MSG); return; }
                setCancerType(e.target.value as CancerType);
                setFile(null);
                setFolderFiles([]);
                setFolderInputKey((k) => k + 1);
                setBrainUploadMode("single");
                setBrainPipeline("mri");
                setCtUploadMode("patient");
                setCtFileInputKey((k) => k + 1);
                setFusionCtFile(null);
                setFusionMriFile(null);
                setFusionInputKey((k) => k + 1);
                setScanResult(null);
                setCtScanResult(null);
                setFusionScanResult(null);
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

            <TextField label="First Name" disabled={loading} value={firstName} onChange={(e) => setFirstName(e.target.value)} sx={fieldSx} />
            <TextField label="Last Name" disabled={loading} value={lastName} onChange={(e) => setLastName(e.target.value)} sx={fieldSx} />
            <TextField
              label="Medical ID"
              disabled={loading}
              value={medicalId}
              onChange={(e) => setMedicalId(e.target.value)}
              placeholder="P0001"
              error={medicalId.length > 0 && !isValidMedicalId(medicalId)}
              helperText={
                medicalId.length > 0 && !isValidMedicalId(medicalId)
                  ? "Format: P followed by 4 digits"
                  : undefined
              }
              sx={fieldSx}
            />
            <TextField
              label="Date of Birth"
              type="date"
              disabled={loading}
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              InputLabelProps={{ shrink: true }}
              error={dobShowError}
              helperText={dobHelperText}
              FormHelperTextProps={{
                sx: {
                  minHeight: 40,
                  lineHeight: 1.45,
                  ...(dobShowError
                    ? {}
                    : { color: "rgba(255,255,255,0.58)" }),
                },
              }}
              inputProps={{ max: adultDobCutoffYmd }}
              sx={dobFieldSx}
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
                  disabled={loading}
                  value={brainPipeline}
                  onChange={(_, v) => {
                    if (v == null) return;
                    if (loading) { setError(LOCKED_MSG); return; }
                    setBrainPipeline(v);
                    if (v === "ct") {
                      setCtUploadMode("patient");
                      setCtFileInputKey((k) => k + 1);
                    }
                    setFile(null);
                    setFolderFiles([]);
                    setFolderInputKey((k) => k + 1);
                    setFusionCtFile(null);
                    setFusionMriFile(null);
                    setFusionInputKey((k) => k + 1);
                    setScanResult(null);
                    setCtScanResult(null);
                    setFusionScanResult(null);
                    setError("");
                  }}
                  sx={{
                    mb: 2,
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 0.5,
                    pointerEvents: loading ? "none" : "auto",
                    opacity: loading ? 0.85 : 1,
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
                  <ToggleButton value="mri">Brain MRI</ToggleButton>
                  <ToggleButton value="ct">Brain CT</ToggleButton>
                  <ToggleButton value="fusion">CT + MRI Fusion</ToggleButton>
                </ToggleButtonGroup>
              )}

              {cancerType === "brain" && brainPipeline === "ct" && (
                <ToggleButtonGroup
                  exclusive
                  disabled={loading}
                  value={ctUploadMode}
                  onChange={(_, v: CtUploadMode | null) => {
                    if (v == null) return;
                    if (loading) { setError(LOCKED_MSG); return; }
                    setCtUploadMode(v);
                    setFile(null);
                    setFolderFiles([]);
                    setFolderInputKey((k) => k + 1);
                    setCtFileInputKey((k) => k + 1);
                    setCtScanResult(null);
                    setError("");
                  }}
                  sx={{
                    mb: 2,
                    pointerEvents: loading ? "none" : "auto",
                    opacity: loading ? 0.85 : 1,
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
                  <ToggleButton value="patient">Patient folder</ToggleButton>
                </ToggleButtonGroup>
              )}

              {cancerType === "brain" && brainPipeline === "mri" && (
                <ToggleButtonGroup
                  exclusive
                  disabled={loading}
                  value={brainUploadMode}
                  onChange={(_, v) => {
                    if (v == null) return;
                    if (loading) { setError(LOCKED_MSG); return; }
                    setBrainUploadMode(v);
                    setFile(null);
                    setFolderFiles([]);
                    setFolderInputKey((k) => k + 1);
                    setScanResult(null);
                    setCtScanResult(null);
                    setFusionScanResult(null);
                    setError("");
                  }}
                  sx={{
                    mb: 2,
                    pointerEvents: loading ? "none" : "auto",
                    opacity: loading ? 0.85 : 1,
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

              {cancerType === "brain" && brainPipeline === "fusion" ? (
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                    gap: 2,
                  }}
                >
                  <Box
                    component="label"
                    htmlFor="flare-fusion-ct-upload"
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onDrop={onDropFusionCt}
                    sx={{
                      display: "block",
                      cursor: loading ? "not-allowed" : "pointer",
                      pointerEvents: loading ? "none" : "auto",
                      border: "1px dashed rgba(251,191,36,0.35)",
                      borderRadius: 2,
                      py: 3,
                      px: 2,
                      textAlign: "center",
                      backgroundColor: "rgba(255,255,255,0.03)",
                      "&:hover": { backgroundColor: "rgba(255,255,255,0.05)" },
                    }}
                  >
                    <Typography sx={{ color: "#fbbf24", fontWeight: 700, mb: 0.5 }}>CT input</Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.9rem" }}>
                      NPZ / ZIP preferred for patient-level CT; JPG / PNG supported
                    </Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem", mt: 0.75, maxWidth: 400, mx: "auto" }}>
                      For best CT sequence inference, use a preprocessed NPZ or zipped DICOM study.
                    </Typography>
                    {fusionCtFile && (
                      <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.85rem", wordBreak: "break-all" }}>
                        {fusionCtFile.name}
                      </Typography>
                    )}
                    <input
                      key={`ct-${fusionInputKey}`}
                      id="flare-fusion-ct-upload"
                      type="file"
                      accept=".npz,.zip,.jpg,.jpeg,.png"
                      disabled={loading}
                      style={{ display: "none" }}
                      onChange={(e) => {
                        const f = e.target.files?.[0] ?? null;
                        if (f && isFusionCtUploadFile(f)) {
                          setFusionCtFile(f);
                          setCtScanResult(null);
                          setFusionScanResult(null);
                          setError("");
                        }
                      }}
                    />
                  </Box>
                  <Box
                    component="label"
                    htmlFor="flare-fusion-mri-upload"
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onDrop={onDropFusionMri}
                    sx={{
                      display: "block",
                      cursor: loading ? "not-allowed" : "pointer",
                      pointerEvents: loading ? "none" : "auto",
                      border: "1px dashed rgba(155,177,255,0.4)",
                      borderRadius: 2,
                      py: 3,
                      px: 2,
                      textAlign: "center",
                      backgroundColor: "rgba(255,255,255,0.03)",
                      "&:hover": { backgroundColor: "rgba(255,255,255,0.05)" },
                    }}
                  >
                    <Typography sx={{ color: "#9bb1ff", fontWeight: 700, mb: 0.5 }}>MRI scan</Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.9rem" }}>
                      JPG / PNG / NIfTI / NPZ
                    </Typography>
                    {fusionMriFile && (
                      <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.85rem", wordBreak: "break-all" }}>
                        {fusionMriFile.name}
                      </Typography>
                    )}
                    <input
                      key={`mri-${fusionInputKey}`}
                      id="flare-fusion-mri-upload"
                      type="file"
                      accept=".jpg,.jpeg,.png,.nii,.nii.gz,.npz"
                      disabled={loading}
                      style={{ display: "none" }}
                      onChange={(e) => {
                        const f = e.target.files?.[0] ?? null;
                        if (f && /\.(png|jpe?g|nii(\.gz)?|npz)$/i.test(f.name)) {
                          setFusionMriFile(f);
                          setCtScanResult(null);
                          setFusionScanResult(null);
                          setError("");
                        }
                      }}
                    />
                  </Box>
                </Box>
              ) : (
                <Box
                  component="label"
                  htmlFor={
                    cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder"
                      ? "flare-folder-upload"
                      : "flare-scan-upload"
                  }
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={onDrop}
                  sx={{
                    display: "block",
                    cursor: loading ? "not-allowed" : "pointer",
                    pointerEvents: loading ? "none" : "auto",
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
                    {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder"
                      ? "Click to choose a folder or drag files here"
                      : cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "single"
                        ? "Click to upload a CT image slice"
                        : cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "patient"
                          ? "Click to upload a CT patient folder or volume"
                      : "Click to upload or drag and drop"}
                  </Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.85rem", mt: 0.5 }}>
                    {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder"
                      ? "BraTS-style: *-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz ( *-seg.nii.gz ignored )"
                      : cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "single"
                        ? "JPG / PNG — single-slice testing path"
                        : cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "patient"
                          ? ".npz volume or .zip DICOM study — recommended for k=21 sequence inference"
                        : cancerType === "brain"
                          ? "JPG/PNG, NIfTI (.nii / .nii.gz), or NPZ — routed to appropriate pipeline on server"
                          : "JPG/PNG or NIfTI (.nii / .nii.gz)"}
                  </Typography>
                  {cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "single" && (
                    <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem", mt: 0.75, maxWidth: 520, mx: "auto" }}>
                      Single-slice CT is supported for quick testing, but patient-level volume input is preferred for sequence inference.
                    </Typography>
                  )}
                  {cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "patient" && (
                    <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: "0.8rem", mt: 0.75, maxWidth: 520, mx: "auto" }}>
                      Use a preprocessed NPZ volume or zipped DICOM series for patient-level CT inference.
                    </Typography>
                  )}
                  {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder" && folderFiles.length > 0 && (
                    <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.9rem" }}>
                      {folderFiles.length} file{folderFiles.length !== 1 ? "s" : ""} in folder
                    </Typography>
                  )}
                  {!(
                    cancerType === "brain" &&
                    brainPipeline === "mri" &&
                    brainUploadMode === "folder"
                  ) &&
                    file && (
                      <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.9rem" }}>
                        {file.name}
                      </Typography>
                    )}
                  {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder" ? (
                    <input
                      key={folderInputKey}
                      id="flare-folder-upload"
                      type="file"
                      disabled={loading}
                      {...({ webkitdirectory: "", directory: "" } as InputHTMLAttributes<HTMLInputElement>)}
                      multiple
                      style={{ display: "none" }}
                      onChange={(e) => onFolderChosen(e.target.files)}
                    />
                  ) : (
                    <input
                      key={
                        cancerType === "brain" && brainPipeline === "ct"
                          ? `ct-${ctFileInputKey}-${ctUploadMode}`
                          : "flare-scan-upload"
                      }
                      id="flare-scan-upload"
                      type="file"
                      disabled={loading}
                      accept={
                        cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "single"
                          ? ".jpg,.jpeg,.png"
                          : cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "patient"
                            ? ".npz,.zip"
                            : ".jpg,.jpeg,.png,.nii,.nii.gz,.npz"
                      }
                      style={{ display: "none" }}
                      onChange={(e) => onFileChosen(e.target.files?.[0] ?? null)}
                    />
                  )}
                </Box>
              )}

              {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "folder" && folderFiles.length > 0 && (
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

              {cancerType === "brain" && brainPipeline === "mri" && brainUploadMode === "single" && file && isNiftiFile(file) && (
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

              {cancerType === "brain" && brainPipeline === "ct" && file && isCtPatientVolumeFile(file) && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Uploaded CT patient study</Typography>
                  <Box sx={{ ...cardSx, maxWidth: 400, p: 2, textAlign: "left" }}>
                    <Typography sx={{ color: "#fbbf24", fontWeight: 700, mb: 1 }}>
                      {file.name.toLowerCase().endsWith(".zip") ? "CT DICOM Study (ZIP)" : "CT NPZ volume"}
                    </Typography>
                    <Typography sx={{ fontSize: "0.9rem", wordBreak: "break-all", mb: 0.5 }}>{file.name}</Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem" }}>
                      {(file.size / 1024 / 1024).toFixed(1)} MB — inference via{" "}
                      <code style={{ color: "rgba(251,191,36,0.95)" }}>POST /api/ct/predict</code>
                      {file.name.toLowerCase().endsWith(".zip") && " (DICOM → preprocess → model)"}
                    </Typography>
                  </Box>
                </Box>
              )}

              {cancerType === "brain" && brainPipeline === "ct" && file && isCtSingleSliceFile(file) && !imagePreviewUrl && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>Uploaded CT image slice</Typography>
                  <Box sx={{ ...cardSx, maxWidth: 400, p: 2, textAlign: "left" }}>
                    <Typography sx={{ fontSize: "0.9rem", wordBreak: "break-all" }}>{file.name}</Typography>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", mt: 0.5 }}>
                      {(file.size / 1024 / 1024).toFixed(1)} MB — <code style={{ color: "rgba(251,191,36,0.95)" }}>POST /api/ct/predict</code>
                    </Typography>
                  </Box>
                </Box>
              )}

              {imagePreviewUrl && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1 }}>
                    {cancerType === "brain" && brainPipeline === "ct" && ctUploadMode === "single"
                      ? "Uploaded CT image slice"
                      : "Uploaded Scan"}
                  </Typography>
                  <Box
                    component="img"
                    src={imagePreviewUrl}
                    alt="Uploaded scan preview"
                    sx={{ ...resultImgSx, maxWidth: 400 }}
                  />
                </Box>
              )}

              {loading && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 3, justifyContent: "center" }}>
                  <CircularProgress size={28} sx={{ color: "#ff5c5c" }} />
                  <Typography sx={{ color: "rgba(255,255,255,0.85)" }}>
                    Analyzing... {formatMmSs(analyzeSeconds)}
                  </Typography>
                </Box>
              )}

              <Box sx={{ display: "flex", gap: 2, mt: 3 }}>
                <Button
                  variant="contained"
                  fullWidth
                  disabled={!canSubmit || loading}
                  onClick={onRun}
                  sx={{
                    backgroundColor: "#ff5c5c",
                    textTransform: "none",
                    borderRadius: 2,
                    py: 1.4,
                    "&:hover": { backgroundColor: "#ff3b3b" },
                  }}
                >
                  Run AI Scan
                </Button>
                <Button
                  variant="outlined"
                  disabled={loading}
                  onClick={resetPageState}
                  sx={{
                    minWidth: 100,
                    textTransform: "none",
                    borderRadius: 2,
                    py: 1.4,
                    color: "rgba(255,255,255,0.7)",
                    borderColor: "rgba(255,255,255,0.2)",
                    "&:hover": { borderColor: "rgba(255,255,255,0.5)", color: "#fff" },
                  }}
                >
                  Clear
                </Button>
              </Box>
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
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", mb: 1, color: "#fff" }}>
              Results — MRI (/api/mri/predict)
            </Typography>
            {completedSeconds != null && (
              <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.9rem", mb: 2 }}>
                Completed in {completedSeconds}s
              </Typography>
            )}
            <Alert
              severity="info"
              sx={{ mb: 2, backgroundColor: "rgba(59,130,246,0.12)", color: "#e5efff" }}
            >
              {REVIEW_DISCLAIMER_TEXT}
            </Alert>
            {mriV2.classification.label.toLowerCase() === "normal" &&
              (mriV2.segmentation.original_url || mriV2.input_image_url) && (
                <Alert
                  severity="success"
                  sx={{
                    mb: 2,
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
              sx={{
                display: "flex",
                flexDirection: { xs: "column", sm: "row" },
                gap: 2,
                mb: 3,
                flexWrap: "wrap",
              }}
            >
              <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                {mriOrigPath ? (
                  mriOrigSrc ? (
                    <Box component="img" src={mriOrigSrc} alt="Original MRI" sx={pairImgSx} />
                  ) : (
                    <Box
                      sx={{
                        ...placeholderBoxSx,
                        minHeight: 200,
                        maxWidth: 300,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                    </Box>
                  )
                ) : (
                  <Box sx={{ ...placeholderBoxSx, minHeight: 200, maxWidth: 300 }}>
                    <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>No original image URL</Typography>
                  </Box>
                )}
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                  Original Scan
                </Typography>
              </Box>
              <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                {mriOvlPath ? (
                  <>
                    {mriOvlSrc ? (
                      <Box component="img" src={mriOvlSrc} alt={mriV2SecondColumnLabel} sx={pairImgSx} />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 300,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      {mriV2SecondColumnLabel}
                    </Typography>
                  </>
                ) : (
                  <>
                    <Box sx={{ ...placeholderBoxSx, minHeight: 200, maxWidth: 300 }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>
                        MRI segmentation overlay is not available for this run.
                      </Typography>
                    </Box>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      {MRI_SEG_OVERLAY_LABEL}
                    </Typography>
                  </>
                )}
              </Box>
            </Box>

            {mriHasSeparateMaskPanel && mriMaskPath && (
              <Box sx={{ mb: 2 }}>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.75, fontSize: "0.85rem" }}>
                  Tumor Mask
                </Typography>
                {mriMaskSrc ? (
                  <Box
                    component="img"
                    src={mriMaskSrc}
                    alt="Tumor mask"
                    sx={{ maxWidth: 200, borderRadius: 1, border: "1px solid rgba(255,255,255,0.12)" }}
                  />
                ) : (
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      minHeight: 80,
                      maxWidth: 200,
                    }}
                  >
                    <CircularProgress size={28} sx={{ color: "#ff5c5c" }} />
                  </Box>
                )}
              </Box>
            )}

            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 560 }}>
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
                    color: mriV2.classification.label.toLowerCase() === "abnormal" ? "#f87171" : "#86efac",
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
          </CardContent>
        </Card>
      )}

      {predLegacy && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", mb: 1, color: "#fff" }}>
              Results
            </Typography>
            {completedSeconds != null && (
              <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.9rem", mb: 2 }}>
                Completed in {completedSeconds}s
              </Typography>
            )}
            <Alert
              severity="info"
              sx={{ mb: 2, backgroundColor: "rgba(59,130,246,0.12)", color: "#e5efff" }}
            >
              {REVIEW_DISCLAIMER_TEXT}
            </Alert>
            <Box
              sx={{
                display: "flex",
                flexDirection: { xs: "column", sm: "row" },
                gap: 2,
                mb: 3,
                flexWrap: "wrap",
              }}
            >
              <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                {imagePreviewUrl ? (
                  <Box component="img" src={imagePreviewUrl} alt="Original" sx={pairImgSx} />
                ) : (
                  <Box sx={{ ...placeholderBoxSx, minHeight: 200, maxWidth: 300 }}>
                    <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>No scan loaded</Typography>
                  </Box>
                )}
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                  Original Scan
                </Typography>
              </Box>
              <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                {legacyLocPath ? (
                  <>
                    {legacyLocSrc ? (
                      <Box component="img" src={legacyLocSrc} alt={legacySecondColumnLabel} sx={pairImgSx} />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 300,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      {legacySecondColumnLabel}
                    </Typography>
                  </>
                ) : (
                  <>
                    <Box sx={{ ...placeholderBoxSx, minHeight: 200, maxWidth: 300 }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.55)" }}>
                        Segmentation processing...
                      </Typography>
                    </Box>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      {MRI_SEG_OVERLAY_LABEL}
                    </Typography>
                  </>
                )}
              </Box>
            </Box>

            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 560 }}>
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
          </CardContent>
        </Card>
      )}

      {ctScanResult && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", mb: 1, color: "#fff" }}>
              Results — CT (/api/ct/predict)
            </Typography>
            {completedSeconds != null && (
              <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.9rem", mb: 2 }}>
                Completed in {completedSeconds}s
              </Typography>
            )}
            <Alert
              severity="info"
              sx={{ mb: 2, backgroundColor: "rgba(59,130,246,0.12)", color: "#e5efff" }}
            >
              {REVIEW_DISCLAIMER_TEXT}
            </Alert>
            <Box
              sx={{
                display: "flex",
                flexDirection: { xs: "column", sm: "row" },
                justifyContent: imagePreviewUrl ? "flex-start" : "center",
                gap: 2,
                mb: 3,
                flexWrap: "wrap",
              }}
            >
              {imagePreviewUrl && (
                <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                  <Box component="img" src={imagePreviewUrl} alt="CT input" sx={pairImgSx} />
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                    Original Scan
                  </Typography>
                </Box>
              )}
              <Box sx={{ flex: "1 1 140px", maxWidth: 300 }}>
                {ctCamPath ? (
                  <>
                    {ctGradCamSrc ? (
                      <Box component="img" src={ctGradCamSrc} alt="CT Grad-CAM Attention Map" sx={pairImgSx} />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 300,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      CT Grad-CAM Attention Map
                    </Typography>
                  </>
                ) : (
                  <>
                    <Box sx={{ ...placeholderBoxSx, minHeight: 200, maxWidth: 300 }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.5)" }}>
                        CT Grad-CAM is not available for this run.
                      </Typography>
                    </Box>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      CT Grad-CAM Attention Map
                    </Typography>
                  </>
                )}
              </Box>
            </Box>

            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 560 }}>
              <Box
                sx={{
                  p: 2.5,
                  borderRadius: 2,
                  backgroundColor: predictionTheme(
                    String(ctScanResult.result_class) === "Abnormal" ? "Malignant" : "Normal"
                  ).bg,
                  border: `1px solid ${predictionTheme(String(ctScanResult.result_class) === "Abnormal" ? "Malignant" : "Normal").border}`,
                  textAlign: "center",
                }}
              >
                <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", mb: 0.5 }}>
                  Prediction
                </Typography>
                <Typography
                  sx={{
                    fontSize: "1.75rem",
                    fontWeight: 900,
                    color: predictionTheme(String(ctScanResult.result_class) === "Abnormal" ? "Malignant" : "Normal").text,
                    textTransform: "capitalize",
                  }}
                >
                  {String(ctScanResult.pred_label ?? "")}
                </Typography>
              </Box>
              <Box>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>Confidence</Typography>
                <Typography
                  sx={{
                    fontSize: "1.5rem",
                    fontWeight: 800,
                    color: predictionTheme(String(ctScanResult.result_class) === "Abnormal" ? "Malignant" : "Normal").text,
                    mb: 1,
                  }}
                >
                  {(Number(ctScanResult.confidence ?? 0) * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0, Number(ctScanResult.confidence ?? 0) * 100))}
                  sx={{
                    height: 10,
                    borderRadius: 1,
                    backgroundColor: "rgba(255,255,255,0.08)",
                    "& .MuiLinearProgress-bar": {
                      backgroundColor: predictionTheme(String(ctScanResult.result_class) === "Abnormal" ? "Malignant" : "Normal").bar,
                      borderRadius: 1,
                    },
                  }}
                />
              </Box>
              <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.85rem" }}>
                p_normal: {Number(ctScanResult.p_normal ?? 0).toFixed(4)} · p_abnormal:{" "}
                {Number(ctScanResult.p_abnormal ?? 0).toFixed(4)}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {fusionScanResult && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.2rem", color: "#fff", mb: 1 }}>
              Results — Fusion (/api/fusion/predict)
            </Typography>
            {completedSeconds != null && (
              <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.9rem", mb: 2 }}>
                Completed in {completedSeconds}s
              </Typography>
            )}
            <Alert
              severity="info"
              sx={{ mb: 2, backgroundColor: "rgba(59,130,246,0.12)", color: "#e5efff" }}
            >
              {REVIEW_DISCLAIMER_TEXT}
            </Alert>

            <Box sx={{ mb: 2, display: "flex", flexDirection: "column", gap: 0.75 }}>
              {fusionScanResult.ct_prob != null && (
                <Typography sx={{ color: "rgba(255,255,255,0.8)", fontSize: "0.95rem" }}>
                  CT abnormality (p_abnormal):{" "}
                  <strong>{(Number(fusionScanResult.ct_prob) * 100).toFixed(1)}%</strong>
                </Typography>
              )}
              {fusionMriClassLine && (
                <Typography sx={{ color: "rgba(255,255,255,0.8)", fontSize: "0.95rem" }}>
                  MRI predicted class: <strong>{fusionMriClassLine.predStr}</strong> · confidence{" "}
                  <strong>{fusionMriClassLine.confStr}</strong>
                </Typography>
              )}
            </Box>

            {(absolutizeApiAssetUrl(fusionMriOvlPath) ||
              absolutizeApiAssetUrl(fusionCtCamPath) ||
              absolutizeApiAssetUrl(fusionMriInPath)) && (
              <Box
                sx={{
                  display: "flex",
                  flexDirection: { xs: "column", sm: "row" },
                  alignItems: { sm: "flex-start" },
                  gap: 2,
                  mb: 3,
                  flexWrap: "wrap",
                }}
              >
                {absolutizeApiAssetUrl(fusionMriOvlPath) && (
                  <Box sx={fusionMriOvlBoxSx}>
                    {fusionMriSegSrc ? (
                      <Box
                        component="img"
                        src={fusionMriSegSrc}
                        alt={fusionMriColumnLabel}
                        sx={fusionMriOvlImgSx}
                      />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 440,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      {fusionMriColumnLabel}
                    </Typography>
                  </Box>
                )}
                {absolutizeApiAssetUrl(fusionCtCamPath) && (
                  <Box sx={fusionCtCamBoxSx}>
                    {fusionCtGradCamSrc ? (
                      <Box
                        component="img"
                        src={fusionCtGradCamSrc}
                        alt="CT Grad-CAM Attention Map"
                        sx={fusionCtCamImgSx}
                      />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 240,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      CT Grad-CAM Attention Map
                    </Typography>
                  </Box>
                )}
                {absolutizeApiAssetUrl(fusionMriInPath) && (
                  <Box sx={fusionMriInBoxSx}>
                    {fusionMriOrigSrc ? (
                      <Box
                        component="img"
                        src={fusionMriOrigSrc}
                        alt="MRI input"
                        sx={fusionMriInImgSx}
                      />
                    ) : (
                      <Box
                        sx={{
                          ...placeholderBoxSx,
                          minHeight: 200,
                          maxWidth: 300,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <CircularProgress size={32} sx={{ color: "#ff5c5c" }} />
                      </Box>
                    )}
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 1, fontWeight: 600, fontSize: "0.9rem" }}>
                      MRI input (soft-tissue reference)
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
              <Box sx={{ textAlign: "center", py: 1 }}>
                <Chip
                  size="small"
                  label={fusionModeLabel(String(fusionScanResult.fusion_mode ?? ""))}
                  sx={{
                    mb: 1.5,
                    fontWeight: 700,
                    backgroundColor: "rgba(155,177,255,0.2)",
                    color: "#e2e8f0",
                    border: "1px solid rgba(255,255,255,0.12)",
                  }}
                />
                <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem", mb: 0.5, fontWeight: 600 }}>
                  Fusion score
                </Typography>
                <Typography
                  sx={{
                    fontSize: { xs: "2.75rem", sm: "3.25rem" },
                    fontWeight: 900,
                    lineHeight: 1.1,
                    color: fusionFinalTheme.text,
                  }}
                >
                  {(Number(fusionScanResult.fusion_score ?? 0) * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0, Number(fusionScanResult.fusion_score ?? 0) * 100))}
                  sx={{
                    mt: 1.5,
                    maxWidth: 480,
                    mx: "auto",
                    height: 12,
                    borderRadius: 1,
                    backgroundColor: "rgba(255,255,255,0.08)",
                    "& .MuiLinearProgress-bar": {
                      backgroundColor: fusionFinalTheme.bar,
                      borderRadius: 1,
                    },
                  }}
                />
              </Box>

              <Box
                sx={{
                  p: 2.5,
                  borderRadius: 2,
                  backgroundColor: fusionFinalTheme.bg,
                  border: `1px solid ${fusionFinalTheme.border}`,
                  textAlign: "center",
                }}
              >
                <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", mb: 0.5 }}>
                  Fused screening status
                </Typography>
                <Typography
                  sx={{
                    fontSize: "1.25rem",
                    fontWeight: 900,
                    lineHeight: 1.35,
                    color: fusionFinalTheme.text,
                  }}
                >
                  {fusionStatusCopy}
                </Typography>
              </Box>

              <Box>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5, fontWeight: 600 }}>
                  CT abnormality (p_abnormal): {(Number(fusionScanResult.ct_prob ?? 0) * 100).toFixed(0)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0, Number(fusionScanResult.ct_prob ?? 0) * 100))}
                  sx={{
                    height: 10,
                    borderRadius: 1,
                    backgroundColor: "rgba(255,255,255,0.08)",
                    "& .MuiLinearProgress-bar": { backgroundColor: "#fbbf24", borderRadius: 1 },
                  }}
                />
                <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.8rem", mt: 0.5 }}>
                  raw {Number(fusionScanResult.ct_prob ?? 0).toFixed(4)}
                </Typography>
              </Box>
              <Box>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5, fontWeight: 600 }}>
                  MRI model score (mri_prob): {(Number(fusionScanResult.mri_prob ?? 0) * 100).toFixed(0)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, Math.max(0, Number(fusionScanResult.mri_prob ?? 0) * 100))}
                  sx={{
                    height: 10,
                    borderRadius: 1,
                    backgroundColor: "rgba(255,255,255,0.08)",
                    "& .MuiLinearProgress-bar": { backgroundColor: "#9bb1ff", borderRadius: 1 },
                  }}
                />
                <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.8rem", mt: 0.5 }}>
                  raw {Number(fusionScanResult.mri_prob ?? 0).toFixed(4)}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {hasAnyResult && (
        <Card sx={{ ...cardSx, mt: 3 }}>
          <CardContent sx={{ p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.1rem", color: "#fff", mb: 1 }}>
              Clinical review
            </Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 2, lineHeight: 1.6 }}>
              <strong>Approve this case in the EHR Database</strong> after you have reviewed the AI output. Formal
              approve/reject is recorded there; approved abnormal cases may appear in aggregate research views (for
              example Geo Tracker).
            </Typography>

            {!reviewCaseId && (
              <Alert severity="info" sx={{ mb: 2, backgroundColor: "rgba(255,255,255,0.08)", color: "#fff" }}>
                No case ID was returned for this run, so nothing was queued for EHR review yet.
              </Alert>
            )}
            {reviewCaseId && !reviewRequired && (
              <Alert severity="info" sx={{ mb: 2, backgroundColor: "rgba(255,255,255,0.08)", color: "#fff" }}>
                This result is not on the pending abnormal review path. Open EHR if a record exists for follow-up.
              </Alert>
            )}
            {reviewCaseId && reviewRequired && (
              <Alert severity="info" sx={{ mb: 2, backgroundColor: "rgba(255,255,255,0.08)", color: "#fff" }}>
                Case ID <strong>{reviewCaseId}</strong> — use EHR Database to approve or reject.
              </Alert>
            )}

            <Button
              variant="contained"
              onClick={() => navigate("/ehr-database")}
              sx={{
                width: "fit-content",
                backgroundColor: "#ff5c5c",
                textTransform: "none",
                "&:hover": { backgroundColor: "#ff3b3b" },
              }}
            >
              Open EHR Database
            </Button>
          </CardContent>
        </Card>
      )}

      <Snackbar
        open={inferSnackOpen}
        autoHideDuration={4000}
        onClose={() => setInferSnackOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        <Alert
          onClose={() => setInferSnackOpen(false)}
          severity="success"
          variant="filled"
          sx={{
            width: "100%",
            backgroundColor: "rgba(34,197,94,0.92)",
            color: "#0b0f19",
            fontWeight: 600,
          }}
        >
          {inferSnackMsg}
        </Alert>
      </Snackbar>
    </Box>
  );
}
