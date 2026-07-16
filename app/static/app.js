(() => {
  const queryInput = document.getElementById("album-query");
  const suggestions = document.getElementById("suggestions");
  const results = document.getElementById("results");
  const secondary = document.getElementById("secondary");
  const loading = document.getElementById("loading");
  const embeddingList = document.getElementById("embedding-list");
  const lastfmList = document.getElementById("lastfm-list");
  const embeddingError = document.getElementById("embedding-error");
  const lastfmError = document.getElementById("lastfm-error");
  const explainEmpty = document.getElementById("explain-empty");
  const explainBody = document.getElementById("explain-body");
  const explainPair = document.getElementById("explain-pair");
  const explainList = document.getElementById("explain-list");
  const explainError = document.getElementById("explain-error");
  const discoveryMode = document.getElementById("discovery-mode");
  const tabs = document.querySelectorAll(".tab");
  const tabPanels = document.querySelectorAll(".tab-panel");

  let seed = null;
  let debounceTimer = null;
  let requestId = 0;

  function titleCase(text) {
    return text.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function albumLabel(artist, album) {
    return `${titleCase(artist)} — ${titleCase(album)}`;
  }

  function setLoading(isLoading) {
    loading.hidden = !isLoading;
    results.classList.toggle("is-loading", isLoading);
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function hideSuggestions() {
    suggestions.hidden = true;
    suggestions.innerHTML = "";
  }

  function setTab(name) {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    tabPanels.forEach((panel) => {
      const active = panel.id === `panel-${name}`;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
  }

  function renderSuggestions(albums) {
    suggestions.innerHTML = "";
    if (!albums.length) {
      hideSuggestions();
      return;
    }
    for (const item of albums) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.innerHTML = `<span class="title">${titleCase(item.album)}</span><span class="artist">${titleCase(item.artist)}</span>`;
      btn.addEventListener("click", () => selectSeed(item.artist, item.album));
      li.appendChild(btn);
      suggestions.appendChild(li);
    }
    suggestions.hidden = false;
  }

  async function searchAlbums(q) {
    if (!q.trim()) {
      hideSuggestions();
      return;
    }
    try {
      const data = await fetchJson(
        `/albums/search?q=${encodeURIComponent(q)}&limit=10`
      );
      renderSuggestions(data.albums || []);
    } catch {
      hideSuggestions();
    }
  }

  function recRowHtml(rec) {
    const score =
      rec.score == null
        ? ""
        : `<span class="score">${Number(rec.score).toFixed(3)}</span>`;
    return `<span><span class="title">${titleCase(rec.album)}</span><span class="meta">${titleCase(rec.artist)}</span></span>${score}`;
  }

  function renderPrimaryList(payload) {
    embeddingList.innerHTML = "";
    embeddingError.hidden = true;
    embeddingError.textContent = "";

    if (payload.error) {
      embeddingError.textContent = payload.error;
      embeddingError.hidden = false;
      return;
    }

    for (const rec of payload.recommendations || []) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "row";
      btn.innerHTML = recRowHtml(rec);
      btn.addEventListener("click", () => explainRecommendation(btn, rec));
      li.appendChild(btn);
      embeddingList.appendChild(li);
    }
  }

  function renderLastfmList(payload) {
    lastfmList.innerHTML = "";
    lastfmError.hidden = true;
    lastfmError.textContent = "";

    if (payload.error) {
      lastfmError.textContent = payload.error;
      lastfmError.hidden = false;
      return;
    }

    for (const rec of payload.recommendations || []) {
      const li = document.createElement("li");
      li.innerHTML = `<div class="row">${recRowHtml(rec)}</div>`;
      lastfmList.appendChild(li);
    }
  }

  function resetExplain() {
    explainEmpty.hidden = false;
    explainBody.hidden = true;
    explainList.innerHTML = "";
    explainPair.textContent = "";
    explainError.hidden = true;
  }

  async function selectSeed(artist, album) {
    seed = { artist, album };
    queryInput.value = albumLabel(artist, album);
    hideSuggestions();

    const thisRequest = ++requestId;

    document.body.classList.add("has-results");
    results.hidden = false;
    secondary.hidden = false;
    setTab("explain");
    resetExplain();

    embeddingList.innerHTML = "";
    lastfmList.innerHTML = "";
    embeddingError.hidden = true;
    lastfmError.hidden = true;
    setLoading(true);

    const params = new URLSearchParams({
      artist,
      album,
      k: "5",
      discovery: discoveryMode.checked ? "true" : "false",
    });

    try {
      const [embeddingResult, lastfmResult] = await Promise.all([
        fetchJson(`/recommend/embedding?${params}`)
          .then((data) => ({ data }))
          .catch((err) => ({ error: err.message })),
        fetchJson(`/recommend/lastfm?${params}`)
          .then((data) => ({ data }))
          .catch((err) => ({ error: err.message })),
      ]);

      if (thisRequest !== requestId) return;

      renderPrimaryList(
        embeddingResult.error
          ? { error: embeddingResult.error }
          : embeddingResult.data
      );
      renderLastfmList(
        lastfmResult.error ? { error: lastfmResult.error } : lastfmResult.data
      );
    } finally {
      if (thisRequest === requestId) setLoading(false);
    }
  }

  async function explainRecommendation(button, rec) {
    if (!seed) return;

    document
      .querySelectorAll(".rec-list--primary .row.active")
      .forEach((el) => el.classList.remove("active"));
    button.classList.add("active");

    setTab("explain");
    explainEmpty.hidden = true;
    explainBody.hidden = false;
    explainPair.textContent = `${albumLabel(seed.artist, seed.album)} → ${albumLabel(rec.artist, rec.album)}`;
    explainList.innerHTML = "";
    explainError.hidden = true;
    explainPair.textContent += " · generating explanation…";

    const params = new URLSearchParams({
      query_artist: seed.artist,
      query_album: seed.album,
      rec_artist: rec.artist,
      rec_album: rec.album,
      n: "3",
    });

    try {
      const data = await fetchJson(`/explain?${params}`);
      explainPair.textContent = `${albumLabel(seed.artist, seed.album)} → ${albumLabel(rec.artist, rec.album)}`;
      for (const q of data.qualities || []) {
        const li = document.createElement("li");
        li.innerHTML = `
          <p class="quality">${q.quality}</p>
          <blockquote>${q.seed_quote}</blockquote>
          <blockquote>${q.rec_quote}</blockquote>
        `;
        explainList.appendChild(li);
      }
    } catch (err) {
      explainPair.textContent = `${albumLabel(seed.artist, seed.album)} → ${albumLabel(rec.artist, rec.album)}`;
      explainError.textContent = err.message;
      explainError.hidden = false;
    }
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
  });

  discoveryMode.addEventListener("change", () => {
    if (seed) selectSeed(seed.artist, seed.album);
  });

  queryInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => searchAlbums(queryInput.value), 250);
  });

  queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideSuggestions();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search")) hideSuggestions();
  });
})();
