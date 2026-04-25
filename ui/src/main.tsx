import ReactDOM from 'react-dom/client' //allows react on the browser
import { BrowserRouter } from 'react-router-dom' //enables page navigation
import App from './App' //imports main app
import Auth0ProviderWithNavigate from './auth/Auth0ProviderWithNavigate'
import { CssBaseline } from '@mui/material'

ReactDOM.createRoot(document.getElementById('root')!).render(
  //finds <div id = "root"> in index.html
  <BrowserRouter>
    <Auth0ProviderWithNavigate>
      <CssBaseline />
      <App />
    </Auth0ProviderWithNavigate>
  </BrowserRouter>
)