
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

// #region agent log
fetch('http://127.0.0.1:7763/ingest/2925affb-f6c8-4554-8741-e0c866a0fdb9',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'d02beb'},body:JSON.stringify({sessionId:'d02beb',runId:'baseline',hypothesisId:'H1',location:'ui/src/main.tsx:13',message:'Auth0 provider init config',data:{origin:window.location.origin,href:window.location.href,domain:auth0Domain,clientId:auth0ClientId,audience:auth0Audience,redirectUri:auth0RedirectUri},timestamp:Date.now()})}).catch(()=>{});
// #endregion

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