// Expects a global PLAYLIST_CONFIG (loaded via playlist-config.js before this script).
// Optional PLAYLIST_CONFIG.mediaBase prefixes video and thumbnail paths.

(function () {
  if (typeof PLAYLIST_CONFIG === "undefined") {
    document.body.innerHTML =
      "<p style='padding:2rem;font-family:sans-serif'>Missing PLAYLIST_CONFIG. Load playlist-config.js before playlist.js.</p>";
    return;
  }

  const mediaBase = PLAYLIST_CONFIG.mediaBase || window.PLAYLIST_MEDIA_BASE || "";
  const STORAGE_KEY = `playlist-progress:${PLAYLIST_CONFIG.title}`;
  const player = document.getElementById("player");
  const playlistEl = document.getElementById("playlist");
  const progressFill = document.getElementById("progress-fill");
  const controlsEl = document.getElementById("controls");
  const progressBarEl = document.getElementById("progress-bar");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");

  const itemCount = PLAYLIST_CONFIG.items.length;
  let currentIndex = 0;
  let watched = new Set();
  const hasMultipleVideos = itemCount > 1;

  function mediaUrl(filename) {
    return encodeURI(`${mediaBase}${filename}`);
  }

  function loadProgress() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      if (typeof saved.index === "number") currentIndex = saved.index;
      if (Array.isArray(saved.watched)) watched = new Set(saved.watched);
    } catch {
      /* ignore corrupt storage */
    }
  }

  function saveProgress() {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ index: currentIndex, watched: [...watched] })
    );
  }

  function clampIndex(index) {
    if (itemCount === 0) return 0;
    return Math.max(0, Math.min(index, itemCount - 1));
  }

  function renderHeader() {
    document.title = `${PLAYLIST_CONFIG.title} — Video Playlist`;
    document.getElementById("series-title").textContent = PLAYLIST_CONFIG.title;
    document.getElementById("series-subtitle").textContent = PLAYLIST_CONFIG.subtitle;
    document.getElementById("playlist-count").textContent =
      itemCount === 1 ? "1 video" : `${itemCount} videos`;
  }

  function renderPlaylist() {
    playlistEl.innerHTML = "";

    PLAYLIST_CONFIG.items.forEach((item, index) => {
      const li = document.createElement("li");
      li.className = "playlist-item";
      li.dataset.index = index;
      if (index === currentIndex) li.classList.add("active");
      if (watched.has(index)) li.classList.add("watched");

      li.innerHTML = `
        <div class="thumb">
          <img src="${mediaUrl(item.thumbnail)}" alt="" loading="lazy" />
        </div>
        <div class="item-meta">
          <div class="item-episode">Episode ${index + 1}</div>
          <div class="item-title">${item.title}</div>
        </div>
      `;

      li.addEventListener("click", () => playIndex(index));
      playlistEl.appendChild(li);
    });

    updateProgressBar();
  }

  function updateNowPlaying(item, index) {
    const episodeEl = document.getElementById("current-episode");
    if (itemCount <= 1) {
      episodeEl.textContent = itemCount === 1 ? "Single video" : "No videos";
    } else {
      episodeEl.textContent = `Episode ${index + 1} of ${itemCount}`;
    }
    document.getElementById("current-title").textContent = item.title;
    document.getElementById("current-description").textContent = item.description;
  }

  function updateNavButtons() {
    if (!hasMultipleVideos) {
      controlsEl.hidden = true;
      progressBarEl.hidden = true;
      btnPrev.disabled = true;
      btnNext.disabled = true;
      return;
    }

    controlsEl.hidden = false;
    progressBarEl.hidden = false;
    btnPrev.disabled = currentIndex === 0;
    btnNext.disabled = currentIndex === itemCount - 1;
  }

  function updateProgressBar() {
    if (!hasMultipleVideos) {
      progressFill.style.width = itemCount === 1 ? "100%" : "0%";
      return;
    }
    const pct = (currentIndex / (itemCount - 1)) * 100;
    progressFill.style.width = `${pct}%`;
  }

  function highlightActive() {
    playlistEl.querySelectorAll(".playlist-item").forEach((el, i) => {
      el.classList.toggle("active", i === currentIndex);
      el.classList.toggle("watched", watched.has(i));
    });
    updateProgressBar();
    updateNavButtons();
  }

  function playIndex(index) {
    if (itemCount === 0) return;

    currentIndex = clampIndex(index);
    const item = PLAYLIST_CONFIG.items[currentIndex];

    player.src = mediaUrl(item.video);
    player.load();
    player.play().catch(() => {});

    updateNowPlaying(item, currentIndex);
    highlightActive();
    saveProgress();

    const activeItem = playlistEl.querySelector(".playlist-item.active");
    if (activeItem) {
      activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function playNext() {
    if (!hasMultipleVideos || btnNext.disabled) return;
    playIndex(currentIndex + 1);
  }

  function playPrev() {
    if (!hasMultipleVideos || btnPrev.disabled) return;
    playIndex(currentIndex - 1);
  }

  function markWatched(index) {
    watched.add(index);
    highlightActive();
    saveProgress();
  }

  btnNext.addEventListener("click", playNext);
  btnPrev.addEventListener("click", playPrev);

  player.addEventListener("ended", () => {
    markWatched(currentIndex);
    if (hasMultipleVideos && currentIndex < itemCount - 1) {
      playNext();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (!hasMultipleVideos) return;
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;

    if (event.key === "ArrowRight" || event.key === "n") {
      event.preventDefault();
      playNext();
    } else if (event.key === "ArrowLeft" || event.key === "p") {
      event.preventDefault();
      playPrev();
    }
  });

  function init() {
    loadProgress();
    currentIndex = clampIndex(currentIndex);
    renderHeader();
    renderPlaylist();
    updateNavButtons();

    if (itemCount === 0) {
      document.getElementById("current-episode").textContent = "No videos";
      document.getElementById("current-title").textContent = "Playlist is empty";
      document.getElementById("current-description").textContent =
        "Add entries to playlist-config.js to populate this playlist.";
      return;
    }

    playIndex(currentIndex);
  }

  init();
})();
