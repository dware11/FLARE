import { Fragment, useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Alert,
  Button,
} from "@mui/material";
import { fetchGeoSummary } from "../api/flareAPI";
import type { GeoSummary, HospitalSummary } from "../api/flareAPI";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const cardSx = {
  backgroundColor: "rgba(0,0,0,0.30)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 3,
  boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
  color: "#fff",
};

/** Risk from approved_abnormal: 4+ → High, 3 → Medium, 2 or fewer → Low */
function markerColor(h: HospitalSummary): string {
  if (h.approvedAbnormalCount >= 4) return "#ef4444";
  if (h.approvedAbnormalCount === 3) return "#f97316";
  return "#22c55e";
}

function severityLabel(h: HospitalSummary): string {
  if (h.approvedAbnormalCount >= 4) return "High";
  if (h.approvedAbnormalCount === 3) return "Medium";
  return "Low";
}

function severityChipColor(h: HospitalSummary) {
  if (h.approvedAbnormalCount >= 4)
    return { bg: "rgba(239,68,68,0.18)", border: "rgba(239,68,68,0.45)", text: "#fca5a5" };
  if (h.approvedAbnormalCount === 3)
    return { bg: "rgba(249,115,22,0.18)", border: "rgba(249,115,22,0.45)", text: "#fdba74" };
  return { bg: "rgba(34,197,94,0.18)", border: "rgba(34,197,94,0.45)", text: "#86efac" };
}

function trendLabel(t: HospitalSummary["trend"]): string {
  if (t === "up") return "Trending up";
  if (t === "attention") return "Needs attention";
  return "Stable";
}

export default function GeoTracker() {
  const navigate = useNavigate();
  const [geo, setGeo] = useState<GeoSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadGeo = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      setGeo(await fetchGeoSummary());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load Geo Tracker data.");
      setGeo(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadGeo();
  }, [loadGeo]);

  useEffect(() => {
    const onRefreshApp = () => {
      void loadGeo();
    };
    window.addEventListener("flare:refresh-app", onRefreshApp);
    return () => window.removeEventListener("flare:refresh-app", onRefreshApp);
  }, [loadGeo]);

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
          Geo Tracker
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.6)", mt: 0.5, maxWidth: 720, lineHeight: 1.6 }}>
          Research-facing view: hospital-level signals for the Houston metro. Clinical review and
          patient-identifiable data belong in EHR Database.
        </Typography>
      </Box>

      <Alert
        severity="info"
        sx={{
          mb: 3,
          backgroundColor: "rgba(59,130,246,0.12)",
          color: "#fff",
          border: "1px solid rgba(59,130,246,0.25)",
        }}
      >
        Research view — patient-identifiable data is not shown. Approve or reject screening cases in{" "}
        <strong>EHR Database</strong>.
      </Alert>

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
                            Approved abnormal (aggregate): <b>{h.approvedAbnormalCount}</b>
                            <br />
                            Pending abnormal (aggregate): <b>{h.pendingCount}</b>
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
                  Total Pending (abnormal, aggregate)
                </Typography>
                <Typography sx={{ fontSize: "2.25rem", fontWeight: 900 }}>
                  {geo.totals.pending}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={cardSx}>
              <CardContent sx={{ p: 3 }}>
                <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.9rem", mb: 1 }}>
                  Total Approved Abnormal (aggregate)
                </Typography>
                <Typography sx={{ fontSize: "2.25rem", fontWeight: 900 }}>
                  {geo.totals.approvedAbnormal}
                </Typography>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, alignItems: "center" }}>
            <Button
              type="button"
              variant="outlined"
              size="small"
              onClick={() => navigate("/ehr-database")}
              sx={{
                textTransform: "none",
                fontWeight: 500,
                color: "rgba(255,255,255,0.65)",
                borderColor: "rgba(255,255,255,0.22)",
                fontSize: "0.8rem",
                py: 0.5,
                px: 1.25,
                "&:hover": {
                  borderColor: "rgba(255,255,255,0.35)",
                  backgroundColor: "rgba(255,255,255,0.04)",
                },
              }}
            >
              Open EHR Database
            </Button>
            <Typography sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.9rem" }}>
              Case-level actions are recorded only in EHR.
            </Typography>
          </Box>

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
                        Pending abnormal: {h.pendingCount}
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
        </Box>
      )}
    </Box>
  );
}
