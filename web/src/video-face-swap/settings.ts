// Persisted UI settings for video-face-swap. Stores everything that isn't
// per-video or per-face (so no file inputs, no time ranges) so that the
// next visit can pick up where the last one left off. Mobile-friendly
// defaults are applied on first visit so the tool is usable out of the box
// on phones.

const STORAGE_KEY = "intabai-video-face-swap-settings";

export type WorkerMode = "off" | "perFrame" | "full";

export interface PersistedSettings {
  swapModel: string;
  detector: string;
  enhancer: string;
  useXseg: boolean;
  doPreview: boolean;
  downscale: string;
  rangeLimit: boolean;
  separatePreview: boolean;
  rangePanelOpen: boolean;
  advancedPanelOpen: boolean;
  profilePreview: boolean;
  gpuPaste: boolean;
  workerMode: WorkerMode;
}

export function isMobile(): boolean {
  if (typeof window === "undefined") return false;
  // pointer:coarse matches phones/tablets but not desktops with touchscreens
  return window.matchMedia?.("(pointer: coarse)").matches ?? false;
}

/** Mobile-tuned defaults: skip slow CPU work, use the fast detector. */
export function getMobileDefaults(): Partial<PersistedSettings> {
  return {
    detector: "scrfd_500m",
    enhancer: "",
    useXseg: false,
    doPreview: true,
    downscale: "720", // 720p target if available
    rangeLimit: false,
    separatePreview: false,
    rangePanelOpen: false,
    advancedPanelOpen: false,
    profilePreview: false,
    gpuPaste: true,
  };
}

export function loadSettings(): PersistedSettings | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedSettings;
  } catch {
    return null;
  }
}

export function saveSettings(s: PersistedSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // quota exceeded or storage disabled - silently ignore
  }
}

export function clearSettings(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

export function hasSettings(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== null;
  } catch {
    return false;
  }
}
