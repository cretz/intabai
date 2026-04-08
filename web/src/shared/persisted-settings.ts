// Generic localStorage-backed settings holder. Each tool defines its own
// settings shape and constructs a PersistedSettings<T> with a tool-unique
// storage key. The class deliberately does no merging or migration: load()
// returns whatever was last written verbatim (or null), and the caller is
// responsible for filling in defaults for any missing fields. This keeps the
// shared layer ignorant of tool-specific shapes.

export class PersistedSettings<T> {
  constructor(private readonly storageKey: string) {}

  load(): T | null {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return null;
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  }

  save(value: T): void {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(value));
    } catch {
      // quota exceeded or storage disabled - silently ignore
    }
  }

  clear(): void {
    try {
      localStorage.removeItem(this.storageKey);
    } catch {
      // ignore
    }
  }

  has(): boolean {
    try {
      return localStorage.getItem(this.storageKey) !== null;
    } catch {
      return false;
    }
  }
}

/**
 * Coarse-pointer media query as a mobile heuristic. Matches phones and
 * tablets but not desktops with touchscreens. Each tool decides what
 * "mobile" means for its defaults; this just exposes the signal.
 */
export function isMobile(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia?.("(pointer: coarse)").matches ?? false;
}
