// Tiny theme toggle shared across every tool page.
//
// Design:
// - `html { color-scheme: light dark }` in style.css tells the browser "this
//   page supports both; pick defaults based on the OS preference". With no
//   override, dark/light UA defaults (background, text, form controls,
//   scrollbars) all switch automatically with the user's system setting.
// - User override is an explicit `color-scheme: light` or `color-scheme: dark`
//   inline style on <html>, which wins over the stylesheet's pair.
// - Override lives in localStorage so it persists across tools and reloads.
// - The *initial* apply happens via a tiny inline <script> in each
//   index.html <head>, BEFORE style.css loads, so there is no flash of the
//   wrong theme. This file's `applyStoredTheme()` is a safety net that
//   re-applies in case the inline snippet was skipped.
// - This file also wires up a toggle button in the topbar that cycles
//   auto -> light -> dark -> auto and updates its own label.

export type ThemeChoice = "auto" | "light" | "dark";

const STORAGE_KEY = "intabai:theme";

function readStored(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark" || v === "auto") return v;
  } catch {
    // ignore (localStorage disabled)
  }
  return "auto";
}

function writeStored(choice: ThemeChoice): void {
  try {
    if (choice === "auto") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // ignore
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "auto") {
    root.style.removeProperty("color-scheme");
  } else {
    root.style.setProperty("color-scheme", choice);
  }
}

export function applyStoredTheme(): void {
  applyTheme(readStored());
}

/** Label the "auto" option with what it currently resolves to, so users
 *  understand the select controls theming and can see what the OS is
 *  asking for. Falls back to a plain "auto" when matchMedia is
 *  unavailable. */
function resolvedAutoLabel(): string {
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "auto: dark" : "auto: light";
  }
  return "auto";
}

/** Wire a <select> with three options (auto/light/dark) to the stored
 *  theme preference. The select is expected to exist in the DOM with the
 *  three option values already present - see the topbar markup in each
 *  tool's index.html. */
export function initThemeSelect(select: HTMLSelectElement): void {
  const current = readStored();
  applyTheme(current);
  select.value = current;

  const autoOpt = select.querySelector<HTMLOptionElement>('option[value="auto"]');
  if (autoOpt) {
    autoOpt.textContent = resolvedAutoLabel();
    // Keep the auto label in sync if the OS preference flips while the
    // page is open (OS-level dark-mode schedule, manual toggle, etc.).
    if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
      const mql = window.matchMedia("(prefers-color-scheme: dark)");
      const update = () => {
        autoOpt.textContent = resolvedAutoLabel();
      };
      if (typeof mql.addEventListener === "function") {
        mql.addEventListener("change", update);
      }
    }
  }

  select.addEventListener("change", () => {
    const v = select.value;
    const choice: ThemeChoice = v === "light" || v === "dark" ? v : "auto";
    writeStored(choice);
    applyTheme(choice);
  });
}
