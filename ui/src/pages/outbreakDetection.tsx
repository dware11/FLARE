import { Navigate } from "react-router-dom";

/** Legacy route; use Geo Tracker (merged Outbreak Analytics). */
export default function OutbreakTracker() {
  return <Navigate to="/geo-tracker" replace />;
}
