// Entry script for the root index.html. The only thing the landing page
// actually needs JS for is wiring up the theme select.
import { initThemeSelect } from "./shared/theme";

const sel = document.getElementById("theme-select");
if (sel instanceof HTMLSelectElement) initThemeSelect(sel);
