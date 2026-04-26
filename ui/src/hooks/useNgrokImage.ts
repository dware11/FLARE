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
        if (!res.ok) throw new Error(`Image fetch failed (${res.status})`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const createdUrl = URL.createObjectURL(blob);
        revokeRef.current = createdUrl;
        setObjectUrl(createdUrl);
      })
      .catch(() => {
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
