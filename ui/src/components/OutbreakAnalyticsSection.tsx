import ReactECharts from "echarts-for-react";
import { Box, Typography } from "@mui/material";
import { useMemo } from "react";
import type { OutbreakStatus } from "../api/flareAPI";

type Props = {
  outbreak: OutbreakStatus | null;
};

export default function OutbreakAnalyticsSection({ outbreak }: Props) {
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
        top: 12,
        left: 12,
        itemWidth: 14,
        itemHeight: 8,
        textStyle: { color: "rgba(255,255,255,0.75)" },
        data: ["Trend signal (A)", "Trend signal (B)"],
      },
      grid: { top: 64, left: 48, right: 24, bottom: 40 },
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

  const gaugePct = outbreak
    ? Math.min(100, Math.max(0, outbreak.percent_of_baseline))
    : 0;
  const gaugeColor = outbreak?.outbreak_color ?? "#ff5c5c";
  const levelLabel = outbreak?.outbreak_level ?? "—";
  const caseCount = outbreak?.total_approved_abnormal ?? 0;

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
                [Math.min(1, Math.max(0.001, gaugePct / 100)), gaugeColor],
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
            formatter: (v: number) => `${Number(v).toFixed(2)}%`,
            color: gaugeColor,
            fontSize: 28,
            fontWeight: 700,
          },
          title: { show: false },
          data: [{ value: gaugePct }],
        },
      ],
    }),
    [gaugeColor, gaugePct]
  );

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr", md: "1fr minmax(240px, 320px)" },
        gap: 3,
        alignItems: "stretch",
        mx: 0,
      }}
    >
      <Box sx={{ minHeight: 320, width: "100%" }}>
        <ReactECharts option={chartOption} style={{ height: 360, width: "100%" }} />
      </Box>

      <Box
        sx={{
          borderRadius: 2,
          border: "1px solid rgba(255,255,255,0.10)",
          backgroundColor: "rgba(0,0,0,0.25)",
          p: 2,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography sx={{ color: "rgba(255,255,255,0.65)", fontSize: "0.85rem", fontWeight: 600, mb: 0.5 }}>
          Disease spread level
        </Typography>
        <Typography sx={{ color: gaugeColor, fontSize: "1.5rem", fontWeight: 800, mb: 1, textAlign: "center" }}>
          {levelLabel}
        </Typography>
        <ReactECharts option={gaugeOption} style={{ height: 200, width: "100%" }} />
        <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.8rem", textAlign: "center", mt: -1 }}>
          % of expected metro baseline ({caseCount.toLocaleString()} approved abnormal)
        </Typography>
      </Box>
    </Box>
  );
}
