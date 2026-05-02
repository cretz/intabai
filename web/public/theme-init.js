(function () {
  try {
    var v = localStorage.getItem("intabai:theme");
    if (v === "light" || v === "dark") {
      document.documentElement.style.colorScheme = v;
    }
  } catch (e) {}
})();
