import { useCallback, useEffect, useState } from "react";
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
} from "@mui/material";
import {
  fetchGeoSummary,
  fetchPendingReviews,
  approveReview,
  rejectReview,
} from "../api/flareAPI";
import type { GeoSummary, ReviewCase, HospitalSummary } from "../api/flareAPI";

function severityLabel(hex: string): string {
  const m: Record<string, string> = {
    "#e11d48": "High",
    "#f97316": "Medium",
    "#eab308": "Pending",
    "#94a3b8": "Clear",
  };
  return m[hex.toLowerCase()] ?? "—";
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

  const loadAll = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      const [g, r] = await Promise.all([fetchGeoSummary(), fetchPendingReviews()]);
      setGeo(g);
      setCases(r.cases ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load Geo Tracker data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  async function onApprove(caseId: string) {
    setActingId(caseId);
    try {
      await approveReview(caseId);
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
      <Box sx={{ mb: 3 }}>
        <Typography sx={{ fontSize: "1.7rem", fontWeight: 800, letterSpacing: "0.02em" }}>
          Geo Tracker
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.65)", mt: 0.5 }}>
          Hospital-level abnormal case load and pending review queue
        </Typography>
      </Box>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
          <CircularProgress sx={{ color: "#ff5c5c" }} />
        </Box>
      )}

      {!loading && error && (
        <Alert severity="error" sx={{ mb: 2, backgroundColor: "rgba(239,68,68,0.12)", color: "#fff" }}>
          {error}
        </Alert>
      )}

      {!loading && geo && (
        <>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
              gap: 2,
              mb: 3,
            }}
          >
            <Card
              sx={{
                backgroundColor: "rgba(0,0,0,0.30)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 3,
                boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
              }}
            >
              <CardContent>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>Total Pending</Typography>
                <Typography sx={{ fontSize: "2rem", fontWeight: 800, color: "#fff" }}>
                  {geo.totals.pending}
                </Typography>
              </CardContent>
            </Card>
            <Card
              sx={{
                backgroundColor: "rgba(0,0,0,0.30)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 3,
                boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
              }}
            >
              <CardContent>
                <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 0.5 }}>
                  Total Approved Abnormal
                </Typography>
                <Typography sx={{ fontSize: "2rem", fontWeight: 800, color: "#fff" }}>
                  {geo.totals.approvedAbnormal}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" },
              gap: 2,
              mb: 4,
            }}
          >
            {geo.hospitals.map((h) => (
              <Card
                key={h.hospitalId}
                sx={{
                  height: "100%",
                  backgroundColor: "rgba(0,0,0,0.30)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 3,
                  boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
                }}
              >
                <CardContent>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                    <Box
                      sx={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        backgroundColor: h.severityColor,
                      }}
                    />
                    <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.8rem" }}>
                      {severityLabel(h.severityColor)}
                    </Typography>
                  </Box>
                  <Typography sx={{ fontWeight: 800, color: "#fff", mb: 1 }}>{h.name}</Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.9rem" }}>
                    Pending: {h.pendingCount} · Approved abnormal: {h.approvedAbnormalCount}
                  </Typography>
                  <Chip
                    label={trendLabel(h.trend)}
                    size="small"
                    sx={{
                      mt: 1.5,
                      backgroundColor: "rgba(255,92,92,0.15)",
                      border: "1px solid rgba(255,92,92,0.35)",
                      color: "#ff5c5c",
                      fontWeight: 700,
                    }}
                  />
                </CardContent>
              </Card>
            ))}
          </Box>

          <Typography sx={{ fontWeight: 800, mb: 1.5, color: "#fff" }}>Pending cases</Typography>
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
                        onClick={() => void onApprove(c.caseId)}
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
        </>
      )}
    </Box>
  );
}
