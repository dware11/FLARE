import { useCallback, useEffect, useMemo, useState } from "react";
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
} from "@mui/material";
import { predictScan } from "../api/flareAPI";
import type { CancerType, PredictResponse, ResultClass } from "../api/flareAPI";

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

export default function CancerDetection() {
  const [hospitalId, setHospitalId] = useState("H001");
  const [cancerType, setCancerType] = useState<CancerType | "">("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [medicalId, setMedicalId] = useState("");
  const [dob, setDob] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setImagePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const canUpload = useMemo(() => {
    return Boolean(cancerType) && firstName && lastName && medicalId && dob && hospitalId;
  }, [cancerType, firstName, lastName, medicalId, dob, hospitalId]);

  const canSubmit = canUpload && file;

  const scanDateToday = useMemo(() => new Date().toISOString().split("T")[0], []);

  const onFileChosen = useCallback((f: File | null) => {
    setFile(f);
    setPred(null);
    setError("");
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const f = e.dataTransfer.files?.[0];
      if (f && /\.(png|jpe?g)$/i.test(f.name)) onFileChosen(f);
    },
    [onFileChosen]
  );

  async function onRun() {
    setError("");
    setPred(null);
    if (!cancerType) return setError("Please select a cancer type.");
    if (!file) return setError("Please upload an image.");

    setLoading(true);
    try {
      const result = await predictScan({
        cancerType,
        file,
        hospitalId,
        firstName,
        lastName,
        dob,
        medicalId,
      });
      setPred(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const showGeoWarning = pred && (pred.prediction === "Malignant" || pred.prediction === "Benign");
  const theme = pred ? predictionTheme(pred.prediction) : predictionTheme("Normal");

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
                setPred(null);
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

              <Box
                component="label"
                htmlFor="flare-scan-upload"
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
                  Click to upload or drag and drop
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.85rem", mt: 0.5 }}>
                  PNG or JPG
                </Typography>
                {file && (
                  <Typography sx={{ color: "#ff5c5c", mt: 1.5, fontWeight: 600, fontSize: "0.9rem" }}>
                    {file.name}
                  </Typography>
                )}
                <input
                  id="flare-scan-upload"
                  type="file"
                  accept=".png,.jpg,.jpeg"
                  style={{ display: "none" }}
                  onChange={(e) => onFileChosen(e.target.files?.[0] ?? null)}
                />
              </Box>

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

      {pred && (
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
              {/* LEFT — images */}
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
                  {pred.localization_url ? (
                    <Box
                      component="img"
                      src={pred.localization_url}
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

              {/* RIGHT — AI + summary */}
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
                    {pred.prediction}
                  </Typography>
                </Box>

                <Box>
                  <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>Model Confidence</Typography>
                  <Typography sx={{ fontSize: "1.75rem", fontWeight: 800, color: theme.text, mb: 1 }}>
                    {(pred.confidence * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, Math.max(0, pred.confidence * 100))}
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

                {pred.probabilities && pred.probabilities.length > 0 && (
                  <Box>
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1, fontWeight: 700 }}>
                      Class probabilities
                    </Typography>
                    {pred.probabilities.map((row) => (
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
