import { useMemo, useState } from "react";
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
} from "@mui/material";
import { predictScan } from "../api/flareAPI";
import type { CancerType, PredictResponse } from "../api/flareAPI";

const HOSPITALS = [
  { id: "H001", name: "Houston Methodist Hospital" },
  { id: "H002", name: "Memorial Hermann - Texas Medical Center" },
  { id: "H003", name: "Baylor St. Luke's Medical Center" },
  { id: "H004", name: "Ben Taub Hospital" },
  { id: "H005", name: "Texas Children's Hospital" },
];

export default function CancerDetection() {
  const [hospitalId, setHospitalId] = useState("H001");
  const [cancerType, setCancerType] = useState<CancerType | "">("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [medicalId, setMedicalId] = useState("");
  const [dob, setDob] = useState("");

  const [file, setFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");

  const canUpload = useMemo(() => {
    return Boolean(cancerType) && firstName && lastName && medicalId && dob && hospitalId;
  }, [cancerType, firstName, lastName, medicalId, dob, hospitalId]);

  const canSubmit = canUpload && file;

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

  const showGeoAlert =
    pred && (pred.prediction === "Malignant" || pred.prediction === "Benign");

  return (
    <Box sx={{ px: { xs: 3, md: 10 }, py: 5 }}>
      <Typography sx={{ color: "#fff", fontSize: "1.7rem", fontWeight: 800, mb: 1 }}>
        AI Cancer Detection
      </Typography>
      <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 3 }}>
        Enter patient data, select cancer type, then upload a scan to run the appropriate model.
      </Typography>

      <Card
        sx={{
          backgroundColor: "rgba(0,0,0,0.30)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 3,
          color: "#fff",
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
              gap: 2,
            }}
          >
            <TextField
              select
              label="Hospital"
              value={hospitalId}
              onChange={(e) => setHospitalId(e.target.value)}
              sx={fieldSx}
            >
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
                renderValue: (v) =>
                  v === "brain" ? "Brain" : v === "breast" ? "Breast" : "",
              }}
              sx={fieldSx}
            >
              <MenuItem value="brain">Brain</MenuItem>
              <MenuItem value="breast">
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: "100%" }}>
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

            <TextField
              label="First Name"
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
              sx={fieldSx}
            />
            <TextField
              label="Last Name"
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
              sx={fieldSx}
            />

            <TextField
              label="Medical ID"
              value={medicalId}
              onChange={(e) => setMedicalId(e.target.value)}
              sx={fieldSx}
            />
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

              <input
                type="file"
                accept=".png,.jpg,.jpeg"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                style={{ color: "white" }}
              />

              <Box sx={{ mt: 3 }}>
                <Button
                  variant="contained"
                  disabled={!canSubmit || loading}
                  onClick={onRun}
                  sx={{
                    backgroundColor: "#ff5c5c",
                    textTransform: "none",
                    borderRadius: 2,
                    px: 4,
                    py: 1.2,
                    "&:hover": { backgroundColor: "#ff3b3b" },
                  }}
                >
                  {loading ? "Running..." : "Run AI Scan"}
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

          {pred && (
            <Box sx={{ mt: 4 }}>
              <Typography sx={{ fontWeight: 900, color: "#9bb1ff", mb: 1 }}>
                AI Result
              </Typography>
              <Typography sx={{ color: "rgba(255,255,255,0.85)" }}>
                <b>Prediction:</b> {pred.prediction}
              </Typography>
              <Typography sx={{ color: "rgba(255,255,255,0.85)" }}>
                <b>Confidence:</b> {(pred.confidence * 100).toFixed(1)}%
              </Typography>

              {showGeoAlert && (
                <Alert
                  severity="info"
                  sx={{
                    mt: 2,
                    backgroundColor: "rgba(255,92,92,0.12)",
                    color: "#fff",
                    border: "1px solid rgba(255,92,92,0.35)",
                  }}
                >
                  Abnormal result submitted to Geo Tracker for review.
                </Alert>
              )}

              {pred.localization_url && (
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ color: "rgba(255,255,255,0.7)", mb: 1 }}>
                    Localization (Grad-CAM / Segmentation)
                  </Typography>
                  <Box
                    component="img"
                    src={pred.localization_url}
                    alt="Localization"
                    sx={{ width: "100%", maxWidth: 520, borderRadius: 2, border: "1px solid rgba(255,255,255,0.12)" }}
                  />
                </Box>
              )}
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

const fieldSx = {
  "& .MuiInputBase-root": { color: "#fff", borderRadius: 2 },
  "& label": { color: "rgba(255,255,255,0.65)" },
  "& fieldset": { borderColor: "rgba(255,255,255,0.12)" },
};
