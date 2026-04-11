import { Fragment, useCallback, useEffect, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from "@mui/material";
import {
  fetchGeoSummary,
  fetchPendingReviews,
  approveReview,
  rejectReview,
} from "../api/flareAPI";
import type { GeoSummary, ReviewCase, HospitalSummary } from "../api/flareAPI";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const cardSx = {
  backgroundColor: "rgba(0,0,0,0.30)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 3,
  boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
  color: "#fff",
};

function markerColor(h: HospitalSummary): string {
  if (h.approvedAbnormalCount >= 5) return "#ef4444";
  if (h.approvedAbnormalCount >= 2) return "#f97316";
  return "#22c55e";
}

function severityLabel(h: HospitalSummary): string {
  if (h.approvedAbnormalCount >= 5) return "High";
  if (h.approvedAbnormalCount >= 2) return "Medium";
  return "Low";
}

function severityChipColor(h: HospitalSummary) {
  if (h.approvedAbnormalCount >= 5)
    return { bg: "rgba(239,68,68,0.18)", border: "rgba(239,68,68,0.45)", text: "#fca5a5" };
  if (h.approvedAbnormalCount >= 2)
    return { bg: "rgba(249,115,22,0.18)", border: "rgba(249,115,22,0.45)", text: "#fdba74" };
  return { bg: "rgba(34,197,94,0.18)", border: "rgba(34,197,94,0.45)", text: "#86efac" };
}

function trendLabel(t: HospitalSummary["trend"]): string {
  if (t === "up") return "Trending up";
  if (t === "attention") return "Needs attention";
  return "Stable";
}

export default function GeoTracker() {
  const [geo, setGeo] = useState<GeoSummary | null>(null);
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actingId, setActingId] = useState<string | null>(null);

  const [signOpen, setSignOpen] = useState(false);
  const [approveCaseId, setApproveCaseId] = useState<string | null>(null);
  const [reviewerName, setReviewerName] = useState("");
  const [digitalSignature, setDigitalSignature] = useState("");
  const [signModalError, setSignModalError] = useState("");

  const loadAll = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      const [g, r] = await Promise.all([fetchGeoSummary(), fetchPendingReviews()]);
      setGeo(g);
      setCases(r.cases ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load Geo Tracker data.");
      setGeo(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  function openApproveModal(caseId: string) {
    setApproveCaseId(caseId);
    setReviewerName("");
    setDigitalSignature("");
    setSignModalError("");
    setSignOpen(true);
  }

  async function confirmApprove() {
    if (!approveCaseId) return;
    const name = reviewerName.trim();
    const sig = digitalSignature.trim();
    if (!name || !sig) {
      setSignModalError("Reviewer name and digital signature are required.");
      return;
    }
    setSignModalError("");
    setActingId(approveCaseId);
    setError("");
    try {
      await approveReview(approveCaseId, { reviewerName: name, signature: sig });
      setSignOpen(false);
      setApproveCaseId(null);
      await loadAll();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Approve failed.");
    } finally {
      setActingId(null);
    }
  }

  async function onReject(caseId: string) {
    setActingId(caseId);
    try {
      await rejectReview(caseId);
      await loadAll();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Reject failed.");
    } finally {
      setActingId(null);
    }
  }

  const fieldSx = {
    "& .MuiInputBase-root": { color: "#fff", borderRadius: 2 },
    "& label": { color: "rgba(255,255,255,0.65)" },
    "& fieldset": { borderColor: "rgba(255,255,255,0.12)" },
  };

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
      {/* ── Approve signature dialog ── */}
      <Dialog
        open={signOpen}
        onClose={() => !actingId && setSignOpen(false)}
        PaperProps={{
          sx: {
            backgroundColor: "#0f1117",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 2,
            minWidth: 360,
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 800 }}>Sign off on this case</DialogTitle>
        <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
          {signModalError && (
            <Alert severity="error" sx={{ backgroundColor: "rgba(239,68,68,0.15)", color: "#fff" }}>
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
            disabled={Boolean(actingId)}
            sx={{ color: "rgba(255,255,255,0.7)" }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={Boolean(actingId)}
            onClick={() => void confirmApprove()}
            sx={{
              backgroundColor: "#ff5c5c",
              textTransform: "none",
              "&:hover": { backgroundColor: "#ff3b3b" },
            }}
          >
            Confirm Approval
          </Button>
        </DialogActions>
      </Dialog>

      {/* ── Header ── */}
      <Box sx={{ mb: 5 }}>
        <Typography sx={{ fontSize: "1.7rem", fontWeight: 800, letterSpacing: "0.02em" }}>
          Geo Tracker
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.6)", mt: 0.5 }}>
          Hospital-level case monitoring and radiologist review queue for the Houston metro area.
        </Typography>
      </Box>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
          <CircularProgress sx={{ color: "#ff5c5c" }} />
        </Box>
      )}

      {!loading && error && (
        <Alert severity="error" sx={{ mb: 4, backgroundColor: "rgba(239,68,68,0.12)", color: "#fff" }}>
          {error}
        </Alert>
      )}

      {!loading && geo && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {/* ── Map ── */}
          <Box
            sx={{
              borderRadius: 3,
              overflow: "hidden",
              height: 440,
              width: "100%",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <MapContainer
              center={[29.7064, -95.3978]}
              zoom={13}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {geo.hospitals.map((h) => {
                const col = markerColor(h);
                return (
                  <Fragment key={h.hospitalId}>
                    <CircleMarker
                      center={[h.latitude, h.longitude]}
                      radius={18}
                      pathOptions={{
                        color: col,
                        fillColor: col,
                        fillOpacity: 0.25,
                        weight: 0,
                      }}
                    />
                    <CircleMarker
                      center={[h.latitude, h.longitude]}
                      radius={10}
                      pathOptions={{
                        color: "#fff",
                        weight: 2,
                        fillColor: col,
                        fillOpacity: 0.95,
                      }}
                    >
                      <Popup>
                        <div style={{ minWidth: 180 }}>
                          <strong style={{ fontSize: "0.95rem" }}>{h.name}</strong>
                          <hr style={{ border: "none", borderTop: "1px solid #ddd", margin: "6px 0" }} />
                          <div style={{ fontSize: "0.85rem", lineHeight: 1.6 }}>
                            Approved abnormal: <b>{h.approvedAbnormalCount}</b>
                            <br />
                            Pending review: <b>{h.pendingCount}</b>
                            <br />
                            Trend: <b>{trendLabel(h.trend)}</b>
                          </div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  </Fragment>
                );
              })}
            </MapContainer>
          </Box>

          {/* ── Summary cards ── */}
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
              gap: 3,
            }}
          >
            <Card sx={cardSx}>
              <CardContent sx={{ p: 3 }}>
                <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem", mb: 1 }}>
                  Total Pending
                </Typography>
                <Typography sx={{ fontSize: "2.25rem", fontWeight: 900 }}>
                  {geo.totals.pending}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={cardSx}>
              <CardContent sx={{ p: 3 }}>
                <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem", mb: 1 }}>
                  Total Approved Abnormal
                </Typography>
                <Typography sx={{ fontSize: "2.25rem", fontWeight: 900 }}>
                  {geo.totals.approvedAbnormal}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          {/* ── Hospital cards ── */}
          <Box>
            <Typography sx={{ fontWeight: 800, mb: 2, fontSize: "1.15rem" }}>
              Hospitals
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" },
                gap: 3,
              }}
            >
              {geo.hospitals.map((h) => {
                const chipC = severityChipColor(h);
                return (
                  <Card key={h.hospitalId} sx={{ ...cardSx, height: "100%" }}>
                    <CardContent sx={{ p: 3 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1.5 }}>
                        <Box
                          sx={{
                            width: 12,
                            height: 12,
                            borderRadius: "50%",
                            backgroundColor: markerColor(h),
                            flexShrink: 0,
                          }}
                        />
                        <Chip
                          label={severityLabel(h)}
                          size="small"
                          sx={{
                            fontWeight: 700,
                            fontSize: "0.75rem",
                            backgroundColor: chipC.bg,
                            border: `1px solid ${chipC.border}`,
                            color: chipC.text,
                          }}
                        />
                      </Box>
                      <Typography sx={{ fontWeight: 800, mb: 1.5, fontSize: "1rem" }}>
                        {h.name}
                      </Typography>
                      <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.9rem", lineHeight: 1.7 }}>
                        Pending: {h.pendingCount}
                        <br />
                        Approved abnormal: {h.approvedAbnormalCount}
                      </Typography>
                      <Chip
                        label={trendLabel(h.trend)}
                        size="small"
                        sx={{
                          mt: 2,
                          backgroundColor: "rgba(255,255,255,0.06)",
                          border: "1px solid rgba(255,255,255,0.12)",
                          color: "rgba(255,255,255,0.7)",
                          fontWeight: 600,
                          fontSize: "0.75rem",
                        }}
                      />
                    </CardContent>
                  </Card>
                );
              })}
            </Box>
          </Box>

          {/* ── Pending cases table ── */}
          <Box>
            <Typography sx={{ fontWeight: 800, mb: 2, fontSize: "1.15rem" }}>
              Pending Cases
            </Typography>
            <TableContainer
              component={Paper}
              sx={{
                backgroundColor: "rgba(0,0,0,0.30)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 3,
                overflow: "hidden",
              }}
            >
              <Table>
                <TableHead>
                  <TableRow sx={{ backgroundColor: "rgba(255,255,255,0.03)" }}>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Patient ID</TableCell>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Hospital</TableCell>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Prediction</TableCell>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Confidence</TableCell>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Time</TableCell>
                    <TableCell align="right" sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>
                      Actions
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {cases.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} sx={{ color: "rgba(255,255,255,0.65)", py: 5, textAlign: "center" }}>
                        No pending cases.
                      </TableCell>
                    </TableRow>
                  )}
                  {cases.map((c) => (
                    <TableRow key={c.caseId} hover sx={{ "&:hover": { backgroundColor: "rgba(255,255,255,0.03)" } }}>
                      <TableCell sx={{ color: "#fff" }}>{c.patient_id}</TableCell>
                      <TableCell sx={{ color: "rgba(255,255,255,0.85)" }}>{c.hospitalName}</TableCell>
                      <TableCell sx={{ color: "rgba(255,255,255,0.85)" }}>{c.result_class}</TableCell>
                      <TableCell sx={{ color: "rgba(255,255,255,0.85)" }}>
                        {(c.confidence * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell sx={{ color: "rgba(255,255,255,0.85)" }}>{c.createdAt}</TableCell>
                      <TableCell align="right">
                        <Button
                          size="small"
                          disabled={actingId === c.caseId}
                          onClick={() => openApproveModal(c.caseId)}
                          sx={{
                            mr: 1,
                            textTransform: "none",
                            color: "#86efac",
                            borderColor: "rgba(34,197,94,0.5)",
                            "&:hover": { borderColor: "#86efac" },
                          }}
                          variant="outlined"
                        >
                          Approve
                        </Button>
                        <Button
                          size="small"
                          disabled={actingId === c.caseId}
                          onClick={() => void onReject(c.caseId)}
                          sx={{
                            textTransform: "none",
                            color: "#fca5a5",
                            borderColor: "rgba(239,68,68,0.5)",
                            "&:hover": { borderColor: "#fca5a5" },
                          }}
                          variant="outlined"
                        >
                          Reject
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </Box>
      )}
    </Box>
  );
}
