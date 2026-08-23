import { Outlet } from "react-router-dom";
import { Box, Toolbar } from "@mui/material";
import Header from "../components/header";
import OutbreakBanner from "../components/OutbreakBanner";
import { CancerInferenceProvider } from "../context/CancerInferenceContext";

export default function MainLayout() {
  return (
    <Box sx={{ minHeight: "100vh" }}>
      <Header />
      <Toolbar />
      <CancerInferenceProvider>
        <OutbreakBanner />
        <Outlet />
      </CancerInferenceProvider>
    </Box>
  );
}
