
import ReactDOM from 'react-dom/client' //allows react on the browser
import { BrowserRouter } from 'react-router-dom'//enables page navigation
import App from './App'//imports main app
//import { auth0Config } from './auth/auth0-config' //wraps app with auth0 provider
import { Auth0Provider } from '@auth0/auth0-react'
import { CssBaseline } from '@mui/material'


ReactDOM.createRoot(document.getElementById('root')!).render( //finds <div id = "root"> in index.html

 <Auth0Provider //makes auth0 available throughout the app
    domain={'dev-5lycfk2rcmzrt10t.us.auth0.com'}
    clientId={'xZOj1GNGzd2TCqQHyS9G7dk2wMXJLzqB'}
    authorizationParams={{
      redirect_uri: window.location.origin,
      audience: 'https://flare-api'
    }}
  >
    <BrowserRouter> 
      <CssBaseline />
      <App />
    </BrowserRouter>
  </Auth0Provider>
)