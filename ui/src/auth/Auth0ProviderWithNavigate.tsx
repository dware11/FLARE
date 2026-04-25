import type { AppState, Auth0ProviderOptions } from '@auth0/auth0-react'
import { Auth0Provider } from '@auth0/auth0-react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth0Config } from './auth0-config'

type Props = { children: ReactNode }

const DEBUG_ENDPOINT =
  'http://127.0.0.1:7763/ingest/2925affb-f6c8-4554-8741-e0c866a0fdb9' as const

/**
 * Auth0 must sit *under* the router so `onRedirectCallback` can `navigate()`.
 * Without this, the post-Auth0 URL stays on `/` and login can appear to "loop" back to itself.
 */
export default function Auth0ProviderWithNavigate({ children }: Props) {
  const navigate = useNavigate()

  const onRedirectCallback: Auth0ProviderOptions['onRedirectCallback'] = (
    appState?: AppState
  ) => {
    // #region agent log
    if (import.meta.env.DEV) {
      fetch(DEBUG_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Debug-Session-Id': '39f496',
        },
        body: JSON.stringify({
          sessionId: '39f496',
          runId: 'auth-callback',
          hypothesisId: 'H1',
          location: 'Auth0ProviderWithNavigate.tsx:onRedirectCallback',
          message: 'post-login redirect',
          data: { returnTo: (appState as { returnTo?: string })?.returnTo },
          timestamp: Date.now(),
        }),
      }).catch(() => {})
    }
    // #endregion
    const returnTo = (appState as { returnTo?: string } | undefined)?.returnTo
    navigate(returnTo && returnTo.length > 0 ? returnTo : '/home', { replace: true })
  }

  return (
    <Auth0Provider
      domain={auth0Config.domain}
      clientId={auth0Config.clientId}
      cacheLocation="localstorage"
      authorizationParams={{
        redirect_uri: window.location.origin,
      }}
      onRedirectCallback={onRedirectCallback}
    >
      {children}
    </Auth0Provider>
  )
}
