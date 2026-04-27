import { Routes, Route } from "react-router-dom";
import Login from "./pages/login";
import Home from "./pages/homePage";
import CancerDetection from "./pages/cancerDetection";
import ProtectedRoute from "./routes/protectedRoute";
import MainLayout from "./components/mainLayout";
import EhrDatabase from "./pages/ehrDatabase";
import GeoTracker from "./pages/geoTracker";
import OutbreakTracker from "./pages/outbreakTracker";
import Privacy from "./pages/Privacy";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/privacy" element={<Privacy />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<MainLayout />}>
          <Route path="/home" element={<Home />} />
          <Route path="/cancer-detection" element={<CancerDetection />} />
          <Route path="/geo-tracker" element={<GeoTracker />} />
          <Route path="/outbreak-tracker" element={<OutbreakTracker />} />
          <Route path="/ehr-database" element={<EhrDatabase />} />
        </Route>
      </Route>
    </Routes>
  );
}

