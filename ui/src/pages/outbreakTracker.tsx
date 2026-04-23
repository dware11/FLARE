import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import ReactECharts from "echarts-for-react";
import { fetchOutbreakStatus } from "../api/flareAPI";
import type { OutbreakStatus } from "../api/flareAPI";

const cardSx = {
  backgroundColor: "rgba(0,0,0,0.30)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 3,
  boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
  color: "#fff",
};

/** Labels aligned with backend `HOSPITAL_REGISTRY` (demo). */
const HOSPITAL_LABELS: Record<string, string> = {
  H001: "Houston Methodist Hospital",
  H002: "Memorial Hermann - Texas Medical Center",
  H003: "Baylor St. Luke's Medical Center",
  H004: "Ben Taub Hospital",
  H005: "Texas Children's Hospital",
};

export default function OutbreakTracker() {
  const [outbreak, setOutbreak] = useState<OutbreakStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      setOutbreak(await fetchOutbreakStatus());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load outbreak data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const onRefreshApp = () => {
      void load();
    };
    window.addEventListener("flare:refresh-app", onRefreshApp);
    return () => window.removeEventListener("flare:refresh-app", onRefreshApp);
  }, [load]);

  const totalAbnormal = outbreak?.total_approved_abnormal ?? 0;
  const detectionRate = outbreak
    ? `${outbreak.percent_of_baseline.toFixed(1)}% of expected baseline`
    : "—";

  const hospitalCountRows = useMemo(() => {
    if (!outbreak?.hospital_counts) return [];
    return Object.entries(outbreak.hospital_counts)
      .map(([id, count]) => ({
        id,
        name: HOSPITAL_LABELS[id] ?? id,
        count: count ?? 0,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [outbreak?.hospital_counts]);

  const months = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const trendData = [1, 2, 3, 4, 6, 9, 8, 7, 6, 5, totalAbnormal || 3];

  const chartOption = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: "rgba(30,30,30,0.95)",
      borderColor: "rgba(255,255,255,0.1)",
      borderWidth: 1,
      textStyle: { color: "#fff" },
    },
    grid: { top: 48, left: 56, right: 24, bottom: 40 },
    xAxis: {
      type: "category" as const,
      boundaryGap: false,
      data: months,
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
      axisLabel: { color: "rgba(255,255,255,0.55)" },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value" as const,
      axisLine: { show: false },
      axisLabel: { color: "rgba(255,255,255,0.55)" },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)", type: "dashed" as const } },
    },
    series: [
      {
        name: "Abnormal detection trend",
        type: "line",
        smooth: true,
        data: trendData,
        symbol: "circle",
        symbolSize: 8,
        lineStyle: { width: 3, color: "#ff6b7a" },
        itemStyle: { color: "#ff6b7a" },
        areaStyle: {
          color: {
            type: "linear" as const,
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(255,107,122,0.25)" },
              { offset: 1, color: "rgba(255,107,122,0.02)" },
            ],
          },
        },
      },
    ],
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
      <Box sx={{ mb: 5 }}>
        <Typography sx={{ fontSize: "1.7rem", fontWeight: 800, letterSpacing: "0.02em" }}>
          Outbreak Analytics (Prototype)
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.6)", mt: 1, maxWidth: 720, lineHeight: 1.6 }}>
          <strong>Metro-wide prototype / illustrative.</strong> Top-line metrics and per-hospital counts come from the
          live API; the trend chart uses placeholder months for demo layout only.
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

      {!loading && outbreak && (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
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
                  Confirmed Abnormal Detections
                </Typography>
                <Typography sx={{ fontSize: "2.5rem", fontWeight: 900, color: "#ff6b7a" }}>
                  {totalAbnormal}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={cardSx}>
              <CardContent sx={{ p: 3 }}>
                <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem", mb: 1 }}>
                  Detection Rate
                </Typography>
                <Typography sx={{ fontSize: "1.5rem", fontWeight: 800, color: "#fff" }}>
                  {detectionRate}
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.45)", fontSize: "0.85rem", mt: 0.5 }}>
                  Status: {outbreak.outbreak_level}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          <Card sx={{ ...cardSx, p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.1rem", mb: 0.5 }}>
              Approved abnormal by site (API)
            </Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.85rem", mb: 2 }}>
              Counts from <code style={{ color: "rgba(255,255,255,0.7)" }}>hospital_counts</code> in outbreak status
            </Typography>
            <TableContainer
              component={Paper}
              sx={{
                backgroundColor: "rgba(0,0,0,0.25)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 2,
              }}
            >
              <Table size="small">
                <TableHead>
                  <TableRow sx={{ backgroundColor: "rgba(255,255,255,0.04)" }}>
                    <TableCell sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>Hospital</TableCell>
                    <TableCell align="right" sx={{ color: "rgba(255,255,255,0.75)", fontWeight: 700 }}>
                      Approved abnormal
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {hospitalCountRows.map((row) => (
                    <TableRow key={row.id} hover sx={{ "&:hover": { backgroundColor: "rgba(255,255,255,0.03)" } }}>
                      <TableCell sx={{ color: "#fff" }}>{row.name}</TableCell>
                      <TableCell align="right" sx={{ color: "rgba(255,255,255,0.85)" }}>
                        {row.count}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Card>

          <Card sx={{ ...cardSx, p: 3 }}>
            <Typography sx={{ fontWeight: 800, fontSize: "1.1rem", mb: 0.5 }}>
              Abnormal Detection Trend
            </Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.85rem", mb: 2 }}>
              Illustrative monthly series (prototype only — not a real time series from the server)
            </Typography>
            <ReactECharts option={chartOption} style={{ height: 340, width: "100%" }} />
          </Card>
        </Box>
      )}
    </Box>
  );
}
