import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/login";
import Home from "./pages/homePage";
import CancerDetection from "./pages/cancerDetection";
import ProtectedRoute from "./routes/protectedRoute";
import MainLayout from "./components/mainLayout";
import EhrDatabase from "./pages/ehrDatabase";
import GeoTracker from "./pages/geoTracker";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<MainLayout />}>
          <Route path="/home" element={<Home />} />
          <Route path="/cancer-detection" element={<CancerDetection />} />
          <Route path="/outbreak-tracker" element={<Navigate to="/geo-tracker" replace />} />
          <Route path="/geo-tracker" element={<GeoTracker />} />
          <Route path="/ehr-database" element={<EhrDatabase />} />
        </Route>
      </Route>
    </Routes>
  );
}

