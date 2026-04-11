import ReactECharts from "echarts-for-react";
import { Box, Chip, Typography, Divider } from "@mui/material";
import LocationOnOutlinedIcon from "@mui/icons-material/LocationOnOutlined";
import { useMemo, useState } from "react";

type DemoLens = "pattern" | "activity" | "trend";

export default function OutbreakAnalyticsSection() {
  const [lens, setLens] = useState<DemoLens>("pattern");

  const months = useMemo(
    () => ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    []
  );

  const series1 = useMemo(() => [1400, 1800, 2400, 3200, 4100, 5200, 4700, 4300, 4000, 3700, 2100], []);
  const series2 = useMemo(() => [2200, 2500, 2800, 3100, 4500, 5600, 6400, 5200, 4600, 4300, 3900], []);

  const chartOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(30,30,30,0.95)",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        textStyle: { color: "#fff" },
      },
      legend: {
        top: 16,
        left: 16,
        itemWidth: 14,
        itemHeight: 8,
        textStyle: { color: "rgba(255,255,255,0.75)" },
        data: ["Trend signal (A)", "Trend signal (B)"],
      },
      grid: { top: 72, left: 48, right: 28, bottom: 44 },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: months,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "rgba(255,255,255,0.55)" },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        axisLine: { show: false },
        axisLabel: { color: "rgba(255,255,255,0.55)" },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)", type: "dashed" } },
      },
      series: [
        {
          name: "Trend signal (A)",
          type: "line",
          smooth: true,
          data: series1,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 3, color: "#a78bfa" },
          itemStyle: { color: "#a78bfa" },
          emphasis: { focus: "series" },
        },
        {
          name: "Trend signal (B)",
          type: "line",
          smooth: true,
          data: series2,
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 3, color: "#ff6b7a" },
          itemStyle: { color: "#ff6b7a" },
          emphasis: { focus: "series" },
        },
      ],
    }),
    [months, series1, series2]
  );

  const gaugeValue = 54;
  const casesInArea = 407;
  /** Illustrative only — not a clinical or outbreak classification. */
  const activityLevelLabel = "Moderate (illustrative)";

  const gaugeOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      series: [
        {
          type: "gauge",
          startAngle: 180,
          endAngle: 0,
          center: ["50%", "58%"],
          radius: "85%",
          min: 0,
          max: 100,
          splitNumber: 0,
          axisLine: {
            lineStyle: {
              width: 18,
              color: [
                [gaugeValue / 100, "#ff5c5c"],
                [1, "rgba(255,255,255,0.10)"],
              ],
            },
          },
          pointer: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          detail: {
            show: true,
            offsetCenter: [0, 10],
            formatter: "{value}%",
            color: "#ff5c5c",
            fontSize: 34,
            fontWeight: 700,
          },
          title: { show: false },
          data: [{ value: gaugeValue }],
        },
      ],
    }),
    [gaugeValue]
  );

  return (
    <Box sx={{ mt: 6 }}>
      <Typography sx={{ fontWeight: 800, fontSize: "1.35rem", mb: 1, color: "#fff" }}>
        Outbreak Analytics (Prototype)
      </Typography>
      <Typography sx={{ color: "rgba(255,255,255,0.65)", mb: 1.5, lineHeight: 1.5 }}>
        Exploratory views of volume and time-based signals using placeholder data. This is not outbreak detection or
        etiology classification.
      </Typography>
      <Typography sx={{ color: "rgba(255,255,255,0.5)", mb: 3, fontSize: "0.9rem", lineHeight: 1.55 }}>
        This section is designed for future epidemiological expansion across cancer types and modalities.
      </Typography>

      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, color: "rgba(255,255,255,0.85)" }}>
          <Typography sx={{ fontSize: "1.1rem", fontWeight: 600 }}>Houston, TX</Typography>
          <LocationOnOutlinedIcon />
        </Box>
      </Box>

      <Divider sx={{ borderColor: "rgba(255,92,92,0.55)", mb: 4 }} />

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", lg: "1.2fr 0.8fr" },
          gap: 4,
          alignItems: "start",
        }}
      >
        <Box
          sx={{
            borderRadius: "14px",
            border: "1px solid rgba(255,255,255,0.10)",
            backgroundColor: "rgba(0,0,0,0.35)",
            overflow: "hidden",
            boxShadow: "0 16px 40px rgba(0,0,0,0.45)",
          }}
        >
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", p: 2.5 }}>
            <Box>
              <Typography sx={{ fontWeight: 700, fontSize: "1.05rem" }}>Trend chart (prototype)</Typography>
              <Typography sx={{ color: "rgba(255,255,255,0.70)", fontSize: "0.95rem" }}>
                Illustrative time series — not a confirmed outbreak curve
              </Typography>
            </Box>
            <Chip
              label="Example chart"
              size="small"
              sx={{
                backgroundColor: "rgba(255,255,255,0.08)",
                color: "rgba(255,255,255,0.75)",
              }}
            />
          </Box>
          <Box sx={{ height: 420, px: 1.5, pb: 2 }}>
            <ReactECharts option={chartOption} style={{ height: "100%", width: "100%" }} />
          </Box>
        </Box>

        <Box
          sx={{
            borderRadius: "18px",
            border: "2px dashed rgba(255,255,255,0.25)",
            p: 3,
            backgroundColor: "rgba(0,0,0,0.25)",
          }}
        >
          <Typography sx={{ color: "#ff5c5c", fontSize: "2rem", fontWeight: 800, lineHeight: 1 }}>
            {activityLevelLabel}
          </Typography>
          <Typography sx={{ color: "rgba(255,255,255,0.85)", mb: 0.5, fontWeight: 600 }}>Activity level</Typography>
          <Typography sx={{ color: "rgba(255,255,255,0.5)", mb: 2, fontSize: "0.82rem", lineHeight: 1.4 }}>
            Demo gauge only — does not represent a validated regional risk score
          </Typography>
          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, alignItems: "center" }}>
            <Box>
              <ReactECharts option={gaugeOption} style={{ height: 200, width: "100%" }} />
              <Typography sx={{ color: "rgba(255,255,255,0.75)", textAlign: "center", mt: -1, fontSize: "0.85rem" }}>
                vs. illustrative baseline
              </Typography>
            </Box>
            <Box sx={{ textAlign: "center" }}>
              <Typography sx={{ color: "#ff5c5c", fontSize: "2.5rem", fontWeight: 800 }}>{casesInArea}</Typography>
              <Typography sx={{ color: "rgba(255,255,255,0.75)", fontSize: "0.9rem" }}>
                illustrative case count
                <br />
                (not live registry data)
              </Typography>
            </Box>
          </Box>
          <Box
            sx={{
              mt: 3,
              borderRadius: 2,
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,0.10)",
              backgroundColor: "rgba(255,255,255,0.04)",
            }}
          >
            <Box
              component="img"
              src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/OpenStreetMap_Logo.svg/512px-OpenStreetMap_Logo.svg.png"
              alt="Decorative map placeholder"
              sx={{ width: "100%", height: 180, objectFit: "cover", opacity: 0.75 }}
            />
          </Box>
          <Typography sx={{ color: "rgba(255,255,255,0.75)", textAlign: "center", mt: 1.5, fontSize: "0.9rem" }}>
            Map placeholder — hospital map and counts are shown above
          </Typography>
        </Box>
      </Box>

      <Box sx={{ mt: 4, textAlign: "center" }}>
        <Typography sx={{ color: "#ff5c5c", fontWeight: 800, fontSize: "1.35rem", mb: 0.5 }}>
          Neutral grouping (demo)
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: "0.85rem", mb: 1.5, maxWidth: 520, mx: "auto" }}>
          Choose a lens label for future filters — pattern indicator, activity level, or trend signal (UI only; same
          placeholder data today).
        </Typography>
        <Box sx={{ display: "inline-flex", flexWrap: "wrap", gap: 1, justifyContent: "center", alignItems: "center" }}>
          <Chip
            label="Pattern indicator"
            onClick={() => setLens("pattern")}
            variant={lens === "pattern" ? "filled" : "outlined"}
            sx={{
              color: "#fff",
              borderColor: "rgba(255,255,255,0.25)",
              backgroundColor: lens === "pattern" ? "rgba(255,92,92,0.18)" : "transparent",
            }}
          />
          <Chip
            label="Activity level"
            onClick={() => setLens("activity")}
            variant={lens === "activity" ? "filled" : "outlined"}
            sx={{
              color: "#fff",
              borderColor: "rgba(255,255,255,0.25)",
              backgroundColor: lens === "activity" ? "rgba(255,92,92,0.18)" : "transparent",
            }}
          />
          <Chip
            label="Trend signal"
            onClick={() => setLens("trend")}
            variant={lens === "trend" ? "filled" : "outlined"}
            sx={{
              color: "#fff",
              borderColor: "rgba(255,255,255,0.25)",
              backgroundColor: lens === "trend" ? "rgba(255,92,92,0.18)" : "transparent",
            }}
          />
        </Box>
      </Box>
    </Box>
  );
}
