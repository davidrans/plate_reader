// Manual light/dark override, layered on top of the OS-preference default
// set by the inline <head> script (which runs before first paint to avoid
// a flash of the wrong theme). Persists client-side only.
const THEME_KEY = "plate-reader:theme";

const themeToggle = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-toggle-icon");
const themeLabel = document.getElementById("theme-toggle-label");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-bs-theme", theme);
  themeIcon.textContent = theme === "dark" ? "🌙" : "☀️";
  themeLabel.textContent = theme === "dark" ? "Dark" : "Light";
}

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

applyTheme(document.documentElement.getAttribute("data-bs-theme"));
