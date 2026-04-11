import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { CancerScanResult } from "../api/flareAPI";

export type CancerInferenceStored = {
  scanResult: CancerScanResult | null;
  ctScanResult: Record<string, unknown> | null;
  fusionScanResult: Record<string, unknown> | null;
  completedSeconds: number | null;
};

const defaultStored: CancerInferenceStored = {
  scanResult: null,
  ctScanResult: null,
  fusionScanResult: null,
  completedSeconds: null,
};

const CancerInferenceContext = createContext<{
  stored: CancerInferenceStored;
  patchStored: (p: Partial<CancerInferenceStored>) => void;
  clearStored: () => void;
} | null>(null);

export function CancerInferenceProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useState<CancerInferenceStored>(defaultStored);

  const patchStored = useCallback((p: Partial<CancerInferenceStored>) => {
    setStored((s) => ({ ...s, ...p }));
  }, []);

  const clearStored = useCallback(() => {
    setStored(defaultStored);
  }, []);

  const value = useMemo(
    () => ({ stored, patchStored, clearStored }),
    [stored, patchStored, clearStored]
  );

  return (
    <CancerInferenceContext.Provider value={value}>{children}</CancerInferenceContext.Provider>
  );
}

export function useCancerInference() {
  const ctx = useContext(CancerInferenceContext);
  if (!ctx) {
    throw new Error("useCancerInference must be used within CancerInferenceProvider");
  }
  return ctx;
}
