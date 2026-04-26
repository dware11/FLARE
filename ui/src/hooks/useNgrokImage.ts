import { useEffect, useRef, useState } from "react";
import { NGROK_HEADERS, absolutizeApiAssetUrl } from "../api/flareAPI";

/**
 * Fetches API/static image URLs with ngrok headers; returns a blob: URL for <img src>.
 * Resolves relative paths against API_BASE. Revokes on change/unmount.
 */
export function useNgrokImage(url: string | null | undefined): string | null {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const revokeRef = useRef<string | null>(null);

  useEffect(() => {
    if (revokeRef.current) {
      URL.revokeObjectURL(revokeRef.current);
      revokeRef.current = null;
    }
    setObjectUrl(null);
    const absolute = absolutizeApiAssetUrl(url);
    if (!absolute) return;

    let cancelled = false;
    void fetch(absolute, { headers: NGROK_HEADERS })
      .then((res) => {
        // #region agent log
        fetch("http://127.0.0.1:7763/ingest/2925affb-f6c8-4554-8741-e0c866a0fdb9", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Debug-Session-Id": "4934c9",
          },
          body: JSON.stringify({
            sessionId: "4934c9",
            location: "useNgrokImage.ts:fetch",
            message: "ngrok image fetch",
            data: {
              absolute,
              status: res.status,
              contentType: res.headers.get("content-type"),
            },
            timestamp: Date.now(),
            hypothesisId: "H-fetch",
          }),
        }).catch(() => {});
        // #endregion
        if (!res.ok) throw new Error(`Image fetch failed (${res.status})`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const createdUrl = URL.createObjectURL(blob);
        revokeRef.current = createdUrl;
        setObjectUrl(createdUrl);
      })
      .catch((err: unknown) => {
        // #region agent log
        fetch("http://127.0.0.1:7763/ingest/2925affb-f6c8-4554-8741-e0c866a0fdb9", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Debug-Session-Id": "4934c9",
          },
          body: JSON.stringify({
            sessionId: "4934c9",
            location: "useNgrokImage.ts:catch",
            message: "ngrok image fetch failed",
            data: { absolute, err: err instanceof Error ? err.message : String(err) },
            timestamp: Date.now(),
            hypothesisId: "H-fetch",
          }),
        }).catch(() => {});
        // #endregion
        if (!cancelled) setObjectUrl(null);
      });

    return () => {
      cancelled = true;
      if (revokeRef.current) {
        URL.revokeObjectURL(revokeRef.current);
        revokeRef.current = null;
      }
    };
  }, [url]);

  return objectUrl;
}
