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
} from "@mui/material";
import { predictScan, saveCase } from "../api/flareAPI";
import type { CancerType, PredictResponse } from "../api/flareAPI";

export default function CancerDetection() {
  const [location, setLocation] = useState("Houston");
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
    return Boolean(cancerType) && firstName && lastName && medicalId && dob && location;
  }, [cancerType, firstName, lastName, medicalId, dob, location]);

  const canSubmit = canUpload && file;

  async function onRun() {
    setError("");
    setPred(null);

    if (!cancerType) return setError("Please select a cancer type.");
    if (!file) return setError("Please upload an image.");

    setLoading(true);
    try {
      // 1) run correct model based on cancerType
      const result = await predictScan({ cancerType, file });
      setPred(result);

      // 2) save patient data + AI result for EHR
      await saveCase({
        first_name: firstName,
        last_name: lastName,
        dob,
        medical_id: medicalId,
        location,
        cancer_type: cancerType,
        prediction: result.prediction,
        confidence: result.confidence,
        localization_url: result.localization_url ?? null,
      });
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

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
              label="Location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              sx={fieldSx}
            >
              <MenuItem value="Houston">Houston</MenuItem>
            </TextField>

            <TextField
              select
              label="Cancer Type"
              value={cancerType}
              onChange={(e) => {
                setCancerType(e.target.value as CancerType);
                setFile(null);   // reset upload when switching type
                setPred(null);   // reset results
              }}
              sx={fieldSx}
            >
              <MenuItem value="brain">Brain</MenuItem>
              <MenuItem value="breast">Breast</MenuItem>
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

          {/* Upload appears only after cancerType + patient info are filled */}
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