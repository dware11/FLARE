import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  TextField,
  MenuItem,
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

/** Feb–Jul monthly points (illustrative demo) keyed by HOSPITAL_REGISTRY id */
const ILLUSTRATIVE_TRENDS: Record<string, number[]> = {
  H001: [1, 2, 3, 5, 7, 9],
  H002: [1, 2, 3, 5, 6, 8],
  H003: [0, 1, 2, 3, 4, 5],
  H004: [0, 0, 1, 1, 2, 3],
  H005: [0, 0, 1, 1, 2, 3],
};

const TREND_MONTHS = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"];

const HOSPITALS: { id: string; name: string }[] = [
  { id: "H001", name: "Houston Methodist Hospital" },
  { id: "H002", name: "Memorial Hermann - Texas Medical Center" },
  { id: "H003", name: "Baylor St. Luke's Medical Center" },
  { id: "H004", name: "Ben Taub Hospital" },
  { id: "H005", name: "Texas Children's Hospital" },
];

const fieldSx = {
  "& .MuiInputBase-root": { color: "#fff", borderRadius: 2 },
  "& label": { color: "rgba(255,255,255,0.65)" },
  "& fieldset": { borderColor: "rgba(255,255,255,0.12)" },
};

export default function OutbreakTracker() {
  const [outbreak, setOutbreak] = useState<OutbreakStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedHospital, setSelectedHospital] = useState("H001");

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

  const chartOption = useMemo(() => {
    const series = ILLUSTRATIVE_TRENDS[selectedHospital] ?? ILLUSTRATIVE_TRENDS.H001;
    const hName = HOSPITALS.find((h) => h.id === selectedHospital)?.name ?? "Hospital";
    return {
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
        data: TREND_MONTHS,
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
          name: hName,
          type: "line",
          smooth: true,
          data: series,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 3, color: "#ff6b7a" },
          itemStyle: { color: "#ff6b7a" },
          areaStyle: {
            color: {
              type: "linear" as const,
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(255,107,122,0.25)" },
                { offset: 1, color: "rgba(255,107,122,0.02)" },
              ],
            },
          },
        },
      ],
    };
  }, [selectedHospital]);

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
          Houston Outbreak Analytics
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.6)", mt: 1, maxWidth: 720, lineHeight: 1.6 }}>
          Regional brain abnormality detection trends across Houston metro hospitals. Counts reflect
          clinician-approved AI detections.
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
            <Typography sx={{ fontWeight: 800, fontSize: "1.1rem", mb: 2 }}>
              Abnormal detection trend
            </Typography>
            <TextField
              select
              fullWidth
              label="Hospital"
              value={selectedHospital}
              onChange={(e) => setSelectedHospital(e.target.value)}
              sx={{ ...fieldSx, mb: 2, maxWidth: 480 }}
            >
              {HOSPITALS.map((h) => (
                <MenuItem key={h.id} value={h.id}>
                  {h.name}
                </MenuItem>
              ))}
            </TextField>
            <ReactECharts
              key={selectedHospital}
              option={chartOption}
              style={{ height: 340, width: "100%" }}
            />
            <Typography
              sx={{
                color: "rgba(255,255,255,0.45)",
                fontSize: "0.8rem",
                mt: 1.5,
                textAlign: "center",
              }}
            >
              Illustrative monthly series — prototype only
            </Typography>
          </Card>
        </Box>
      )}
    </Box>
  );
}
