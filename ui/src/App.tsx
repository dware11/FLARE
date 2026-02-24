
import { Routes, Route } from 'react-router-dom'
import Login from './pages/login'
import Home from './pages/homePage'
import CancerDetection from './pages/cancerDetection'
import OutbreakTracker from './pages/outbreakDetection'
import ProtectedRoute from './routes/protectedRoute'
import MainLayout from './components/mainLayout'
import EhrDatabase from './pages/ehrDatabase'

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<Login />} />

      {/* Protected */}
      <Route element={<ProtectedRoute />}>
        <Route element={<MainLayout />}>
          <Route path="/home" element={<Home />} />
          <Route path="/cancer-detection" element={<CancerDetection />} />
          <Route path="/outbreak-tracker" element={<OutbreakTracker />} />
          <Route path="/ehr-database" element={<EhrDatabase />}/>
        </Route>
      </Route>
    </Routes>
  )
}

