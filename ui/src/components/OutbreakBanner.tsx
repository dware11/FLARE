import { useCallback, useEffect, useState } from "react";
import { Box, IconButton, Typography } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { fetchGeoSummary } from "../api/flareAPI";
import type { HospitalSummary } from "../api/flareAPI";

const POLL_MS = 30_000;

function pickOutbreakHospital(hospitals: HospitalSummary[]): HospitalSummary | null {
  const high = hospitals.filter((h) => h.approvedAbnormalCount >= 4);
  if (high.length === 0) return null;
  return high.reduce((a, b) => (a.approvedAbnormalCount >= b.approvedAbnormalCount ? a : b));
}

export default function OutbreakBanner() {
  const [dismissed, setDismissed] = useState(false);
  const [hospital, setHospital] = useState<HospitalSummary | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchGeoSummary();
      setHospital(pickOutbreakHospital(data.hospitals));
    } catch {
      setHospital(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  if (dismissed || !hospital) return null;

  const n = hospital.approvedAbnormalCount;

  return (
    <Box
      role="alert"
      sx={{
        position: "sticky",
        top: 0,
        zIndex: 1300,
        width: "100%",
        backgroundColor: "#7f1d1d",
        color: "#fff",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 2,
        px: 2,
        py: 1.25,
        boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
        "@keyframes outbreakBannerIn": {
          "0%": { opacity: 0, transform: "translateY(-10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        animation: "outbreakBannerIn 0.45s ease-out both",
      }}
    >
      <Typography component="p" sx={{ m: 0, flex: 1, fontSize: "0.95rem", lineHeight: 1.45 }}>
        ⚠️ {hospital.name} — {n} confirmed abnormal brain scans detected. Possible outbreak signal.
      </Typography>
      <IconButton
        size="small"
        onClick={() => setDismissed(true)}
        aria-label="Dismiss outbreak alert"
        sx={{ color: "inherit", flexShrink: 0 }}
      >
        <CloseIcon fontSize="small" />
      </IconButton>
    </Box>
  );
}
