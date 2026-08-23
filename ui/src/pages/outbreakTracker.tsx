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
import { alpha } from "@mui/material/styles";
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

const HOSPITAL_MONTHLY_DATA: Record<string, number[]> = {
  H001: [42, 48, 55, 62, 71, 89],
  H002: [38, 42, 49, 53, 61, 78],
  H003: [28, 30, 34, 38, 42, 51],
  H004: [18, 20, 24, 26, 28, 32],
  H005: [16, 16, 16, 16, 16, 14],
};

const HOSPITAL_LINE_COLORS: Record<string, string> = {
  H001: "#2563EB",
  H002: "#059669",
  H003: "#DB2777",
  H004: "#EA580C",
  H005: "#7C3AED",
};

/** Sum of all 5 hospitals per month; chart label is "Houston Regional Trend" (id stays HOUSTON_OVERALL). */
const HOUSTON_OVERALL_DATA = HOSPITAL_MONTHLY_DATA.H001.map(
  (_, i) =>
    HOSPITAL_MONTHLY_DATA.H001[i] +
    HOSPITAL_MONTHLY_DATA.H002[i] +
    HOSPITAL_MONTHLY_DATA.H003[i] +
    HOSPITAL_MONTHLY_DATA.H004[i] +
    HOSPITAL_MONTHLY_DATA.H005[i]
);

const HOUSTON_OVERALL = "HOUSTON_OVERALL";

function getLastSixMonthLabels(now: Date = new Date()): string[] {
  const fmt = new Intl.DateTimeFormat(undefined, { month: "short" });
  const labels: string[] = [];
  for (let i = 5; i >= 0; i -= 1) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    labels.push(fmt.format(d));
  }
  return labels;
}

const TREND_MONTHS = getLastSixMonthLabels();

const HOSPITAL_IDS = ["H001", "H002", "H003", "H004", "H005"] as const;

const HOSPITALS: { id: string; name: string }[] = [
  { id: HOUSTON_OVERALL, name: "Houston Regional Trend" },
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

const getHospitalStatus = (data: number[]) => {
  const baseline = data[0];
  const latest = data[data.length - 1];
  if (baseline <= 0) {
    if (latest <= 0) return { label: "Normal", color: "#16a34a" };
    return { label: "Elevated", color: "#f59e0b" };
  }
  const pctChange = ((latest - baseline) / baseline) * 100;
  if (pctChange >= 41) return { label: "Critical", color: "#dc2626" };
  if (pctChange >= 16) return { label: "Elevated", color: "#f59e0b" };
  return { label: "Normal", color: "#16a34a" };
};

export default function OutbreakTracker() {
  const [outbreak, setOutbreak] = useState<OutbreakStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedHospital, setSelectedHospital] = useState(HOUSTON_OVERALL);

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

  const trendData = useMemo(() => {
    if (selectedHospital === HOUSTON_OVERALL) return HOUSTON_OVERALL_DATA;
    return HOSPITAL_MONTHLY_DATA[selectedHospital] ?? HOSPITAL_MONTHLY_DATA.H001;
  }, [selectedHospital]);

  const statusBadge = useMemo(() => getHospitalStatus(trendData), [trendData]);

  const detectionSummaryLine = useMemo(() => {
    const data = trendData;
    const latestValue = data[data.length - 1];
    const baselineValue = data[0];
    const firstMonth = TREND_MONTHS[0] ?? "baseline";
    if (baselineValue <= 0) {
      return `${latestValue} cases`;
    }
    const pctChange = (((latestValue - baselineValue) / baselineValue) * 100).toFixed(0);
    const sign = Number(pctChange) >= 0 ? "above" : "below";
    return `${latestValue} cases — ${Math.abs(Number(pctChange))}% ${sign} ${firstMonth} baseline`;
  }, [trendData]);

  const chartOption = useMemo(() => {
    const commonTooltip = {
      trigger: "axis" as const,
      backgroundColor: "rgba(30,30,30,0.95)",
      borderColor: "rgba(255,255,255,0.1)",
      borderWidth: 1,
      textStyle: { color: "#fff" },
    };
    const commonXAxis = {
      type: "category" as const,
      boundaryGap: false,
      data: TREND_MONTHS,
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
      axisLabel: { color: "rgba(255,255,255,0.55)" },
      splitLine: { show: false },
    };
    const commonYAxis = {
      type: "value" as const,
      axisLine: { show: false },
      axisLabel: { color: "rgba(255,255,255,0.55)" },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)", type: "dashed" as const } },
    };

    if (selectedHospital === HOUSTON_OVERALL) {
      const legendData = HOSPITAL_IDS.map(
        (id) => HOSPITALS.find((h) => h.id === id)!.name
      ).concat("Houston Regional Trend");
      const series = [
        ...HOSPITAL_IDS.map((id) => {
          const c = HOSPITAL_LINE_COLORS[id];
          return {
            name: HOSPITALS.find((h) => h.id === id)!.name,
            type: "line" as const,
            smooth: true,
            data: HOSPITAL_MONTHLY_DATA[id],
            symbol: "circle",
            symbolSize: 6,
            lineStyle: { width: 1.5, color: c },
            itemStyle: { color: c },
            z: 1,
          };
        }),
        {
          name: "Houston Regional Trend",
          type: "line" as const,
          smooth: true,
          data: HOUSTON_OVERALL_DATA,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 4, color: "#F59E0B" },
          itemStyle: { color: "#F59E0B" },
          z: 2,
        },
      ];
      return {
        backgroundColor: "transparent",
        tooltip: commonTooltip,
        legend: {
          type: "scroll" as const,
          orient: "vertical" as const,
          right: 24,
          top: "middle" as const,
          data: legendData,
          textStyle: {
            color: "rgba(255,255,255,0.88)",
            fontSize: 11,
            lineHeight: 16,
          },
          itemWidth: 18,
          itemGap: 10,
          itemHeight: 10,
          width: 260,
          pageIconColor: "rgba(255,255,255,0.75)",
          pageTextStyle: { color: "rgba(255,255,255,0.6)" },
          pageButtonItemGap: 6,
        },
        grid: { top: 48, left: 48, right: 300, bottom: 40, containLabel: true },
        xAxis: commonXAxis,
        yAxis: commonYAxis,
        series,
      };
    }

    const series = HOSPITAL_MONTHLY_DATA[selectedHospital] ?? HOSPITAL_MONTHLY_DATA.H001;
    const hName = HOSPITALS.find((h) => h.id === selectedHospital)?.name ?? "Hospital";
    return {
      backgroundColor: "transparent",
      tooltip: commonTooltip,
      grid: { top: 48, left: 56, right: 24, bottom: 40 },
      xAxis: commonXAxis,
      yAxis: commonYAxis,
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
            <Box
              sx={{
                mt: 2,
                p: 2.5,
                borderRadius: 2,
                border: `1px solid ${alpha(statusBadge.color, 0.4)}`,
                backgroundColor: alpha(statusBadge.color, 0.08),
              }}
            >
              <Typography sx={{ color: "rgba(255,255,255,0.9)", fontSize: "0.95rem", lineHeight: 1.65 }}>
                <strong>Detection Rate:</strong> {detectionSummaryLine}
              </Typography>
              <Typography
                sx={{ color: statusBadge.color, fontWeight: 800, fontSize: "1.05rem", mt: 1 }}
              >
                Status: {statusBadge.label}
              </Typography>
            </Box>
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
