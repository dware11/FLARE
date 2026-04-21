import { Navigate, Outlet } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'

export default function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuth0()

  // #region agent log
  fetch('http://127.0.0.1:7763/ingest/2925affb-f6c8-4554-8741-e0c866a0fdb9',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'d02beb'},body:JSON.stringify({sessionId:'d02beb',runId:'baseline',hypothesisId:'H3',location:'ui/src/routes/protectedRoute.tsx:6',message:'ProtectedRoute state',data:{isAuthenticated,isLoading,path:window.location.pathname,href:window.location.href},timestamp:Date.now()})}).catch(()=>{});
  // #endregion

  if (isLoading) return null

  return isAuthenticated ? <Outlet /> : <Navigate to="/" />
}

