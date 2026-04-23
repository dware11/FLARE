
import ReactDOM from 'react-dom/client' //allows react on the browser
import { BrowserRouter } from 'react-router-dom'//enables page navigation
import App from './App'//imports main app
//import { auth0Config } from './auth/auth0-config' //wraps app with auth0 provider
import { Auth0Provider } from '@auth0/auth0-react'
import { CssBaseline } from '@mui/material'

const auth0Domain = 'dev-5lycfk2rcmzrt10t.us.auth0.com'
const auth0ClientId = 'xZOj1GNGzd2TCqQHyS9G7dk2wMXJLzqB'
const auth0Audience = 'https://flare-api'
const auth0RedirectUri = window.location.origin

ReactDOM.createRoot(document.getElementById('root')!).render( //finds <div id = "root"> in index.html

 <Auth0Provider //makes auth0 available throughout the app
    domain={auth0Domain}
    clientId={auth0ClientId}
    authorizationParams={{
      redirect_uri: auth0RedirectUri,
      audience: auth0Audience
    }}
  >
    <BrowserRouter> 
      <CssBaseline />
      <App />
    </BrowserRouter>
  </Auth0Provider>
)