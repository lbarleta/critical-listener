(() => {
  const queryInput = document.getElementById("album-query");
  const suggestions = document.getElementById("suggestions");
  const results = document.getElementById("results");
  const secondary = document.getElementById("secondary");
  const embeddingList = document.getElementById("embedding-list");
  const embeddingError = document.getElementById("embedding-error");
  const seedInfoEl = document.getElementById("seed-info");
  const autoplayStack = document.getElementById("autoplay-stack");
  const autoplayError = document.getElementById("autoplay-error");
  const benchmarkEmbedding = document.getElementById("benchmark-embedding");
  const benchmarkLastfm = document.getElementById("benchmark-lastfm");
  const benchmarkEmbeddingError = document.getElementById(
    "benchmark-embedding-error"
  );
  const benchmarkLastfmError = document.getElementById("benchmark-lastfm-error");
  const explainEmpty = document.getElementById("explain-empty");
  const explainBody = document.getElementById("explain-body");
  const explainPair = document.getElementById("explain-pair");
  const explainSeedLabel = document.getElementById("explain-seed-label");
  const explainRecLabel = document.getElementById("explain-rec-label");
  const explainList = document.getElementById("explain-list");
  const explainError = document.getElementById("explain-error");
  const discoveryMode = document.getElementById("discovery-mode");
  const tabs = document.querySelectorAll(".tab");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const baselineSpinners = document.querySelectorAll(
    "#tab-benchmark .tab-spinner, #tab-metrics .tab-spinner"
  );
  const autoplaySpinner = document.querySelector("#tab-autoplay .tab-spinner");
  const explainSpinner = document.querySelector("#tab-explain .tab-spinner");

  let seed = null;
  let debounceTimer = null;
  let requestId = 0;
  let explainRequestId = 0;
  let embeddingPayload = null;
  let lastfmPayload = null;

  function titleCase(text) {
    return text.replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function albumLabel(artist, album) {
    return `${titleCase(artist)} — ${titleCase(album)}`;
  }

  function setBaselineLoading(isLoading) {
    baselineSpinners.forEach((el) => {
      el.hidden = !isLoading;
    });
  }

  function setAutoPlayLoading(isLoading) {
    if (autoplaySpinner) autoplaySpinner.hidden = !isLoading;
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function setExplainLoading(isLoading) {
    if (explainSpinner) explainSpinner.hidden = !isLoading;
  }

  function formatCount(n) {
    if (n == null || Number.isNaN(Number(n))) return null;
    const value = Number(n);
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (value >= 10_000) return `${Math.round(value / 1000)}k`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
    return String(value);
  }

  function albumKey(artist, album) {
    return `${String(artist || "").trim().toLowerCase()}::${String(album || "").trim().toLowerCase()}`;
  }

  function metaLinesHtml(album) {
    const parts = [];
    const reviews = formatCount(album.review_count);
    const listeners = formatCount(album.listeners);
    const stats = [];
    if (reviews != null) stats.push(`${reviews} reviews`);
    if (listeners != null) stats.push(`${listeners} listeners`);
    if (stats.length) {
      parts.push(`<p class="benchmark-stats">${stats.join(" · ")}</p>`);
    }
    const genres = (album.genres || []).slice(0, 3);
    if (genres.length) {
      parts.push(
        `<p class="benchmark-meta-line">Genres · ${genres.join(", ")}</p>`
      );
    }
    const tags = (album.tags || []).slice(0, 5);
    if (tags.length) {
      parts.push(`<p class="benchmark-meta-line">Tags · ${tags.join(", ")}</p>`);
    }
    return parts.join("");
  }

  function renderQualities(qualities) {
    explainList.innerHTML = "";
    for (const item of qualities || []) {
      const li = document.createElement("li");
      li.className = "explain-theme";
      li.innerHTML = `
        <p class="quality"></p>
        <div class="explain-quotes">
          <figure class="explain-quote">
            <figcaption>Seed</figcaption>
            <blockquote class="seed-quote"></blockquote>
          </figure>
          <figure class="explain-quote explain-quote--match">
            <figcaption>Match</figcaption>
            <blockquote class="rec-quote"></blockquote>
          </figure>
        </div>
      `;
      li.querySelector(".quality").textContent = item.quality || "";
      li.querySelector(".seed-quote").textContent = item.seed_quote || "";
      li.querySelector(".rec-quote").textContent = item.rec_quote || "";
      explainList.appendChild(li);
    }
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

  function benchmarkItemHtml(rec, shared) {
    const score =
      rec.score == null
        ? ""
        : `<span class="benchmark-score">${Number(rec.score).toFixed(3)}</span>`;
    const sharedMark = shared
      ? `<span class="benchmark-shared" title="Also recommended by the other system" aria-label="Also in both recommenders">◎</span>`
      : "";
    return `
      <div class="benchmark-item${shared ? " is-shared" : ""}">
        <div class="benchmark-item-head">
          <span class="benchmark-rank"></span>
          <div class="benchmark-item-titles">
            <span class="benchmark-title">${titleCase(rec.album)}${sharedMark}</span>
            <span class="benchmark-artist">${titleCase(rec.artist)}</span>
          </div>
          ${score}
        </div>
        ${metaLinesHtml(rec)}
      </div>
    `;
  }

  function sharedAlbumKeys() {
    const keys = new Set();
    const emb = new Set(
      (embeddingPayload?.recommendations || []).map((r) =>
        albumKey(r.artist, r.album)
      )
    );
    for (const rec of lastfmPayload?.recommendations || []) {
      const key = albumKey(rec.artist, rec.album);
      if (emb.has(key)) keys.add(key);
    }
    return keys;
  }

  function renderSeedInfo() {
    const fromApi = embeddingPayload?.seed || lastfmPayload?.seed;
    const artist = seed?.artist || fromApi?.artist;
    const album = seed?.album || fromApi?.album;
    if (!artist || !album) {
      seedInfoEl.hidden = true;
      seedInfoEl.innerHTML = "";
      return;
    }
    const info = { ...(fromApi || {}), artist, album };
    seedInfoEl.hidden = false;
    seedInfoEl.innerHTML = `
      <p class="seed-info-name">${albumLabel(artist, album)}</p>
      ${metaLinesHtml(info)}
    `;
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

  function renderBenchmarkColumn(listEl, errorEl, payload, sharedKeys) {
    listEl.innerHTML = "";
    errorEl.hidden = true;
    errorEl.textContent = "";

    if (!payload) return;

    if (payload.error) {
      errorEl.textContent = payload.error;
      errorEl.hidden = false;
      return;
    }

    for (const rec of payload.recommendations || []) {
      const li = document.createElement("li");
      const shared = sharedKeys.has(albumKey(rec.artist, rec.album));
      li.innerHTML = benchmarkItemHtml(rec, shared);
      listEl.appendChild(li);
    }
  }

  function renderBenchmark() {
    const sharedKeys = sharedAlbumKeys();
    renderBenchmarkColumn(
      benchmarkEmbedding,
      benchmarkEmbeddingError,
      embeddingPayload,
      sharedKeys
    );
    renderBenchmarkColumn(
      benchmarkLastfm,
      benchmarkLastfmError,
      lastfmPayload,
      sharedKeys
    );
  }

  function renderAutoPlay() {
    autoplayStack.innerHTML = "";
    autoplayError.hidden = true;
    autoplayError.textContent = "";

    if (!embeddingPayload) return;

    if (embeddingPayload.error) {
      autoplayError.textContent = embeddingPayload.error;
      autoplayError.hidden = false;
      return;
    }

    const stack = embeddingPayload.recommendations || [];
    stack.forEach((item, index) => {
      const li = document.createElement("li");
      li.className = "autoplay-card";
      if (index === 0) li.classList.add("is-current");
      const score =
        item.score == null
          ? ""
          : `<span class="autoplay-score">${Number(item.score).toFixed(3)}</span>`;
      li.innerHTML = `
        <div class="autoplay-card-head">
          <span class="autoplay-role">${index === 0 ? "Now" : String(index + 1).padStart(2, "0")}</span>
          <div class="autoplay-card-titles">
            <span class="autoplay-title">${titleCase(item.album)}</span>
            <span class="autoplay-artist">${titleCase(item.artist)}</span>
          </div>
          ${score}
        </div>
        ${metaLinesHtml(item)}
      `;
      autoplayStack.appendChild(li);
    });
  }

  function resetExplain() {
    explainRequestId += 1;
    setExplainLoading(false);
    explainEmpty.hidden = false;
    explainBody.hidden = true;
    explainList.innerHTML = "";
    explainPair.hidden = true;
    explainSeedLabel.textContent = "";
    explainRecLabel.textContent = "";
    explainError.hidden = true;
    explainError.textContent = "";
  }

  function resetPage() {
    seed = null;
    requestId += 1;
    embeddingPayload = null;
    lastfmPayload = null;
    hideSuggestions();
    document.body.classList.remove("has-results");
    results.hidden = true;
    secondary.hidden = true;
    setBaselineLoading(false);
    setAutoPlayLoading(false);
    embeddingList.innerHTML = "";
    embeddingError.hidden = true;
    renderSeedInfo();
    renderBenchmark();
    renderAutoPlay();
    resetExplain();
    setTab("explain");
  }

  async function selectSeed(artist, album) {
    seed = { artist, album };
    queryInput.value = albumLabel(artist, album);
    hideSuggestions();

    const thisRequest = ++requestId;
    embeddingPayload = null;
    lastfmPayload = null;

    document.body.classList.add("has-results");
    results.hidden = false;
    secondary.hidden = false;
    setTab("explain");
    resetExplain();

    embeddingList.innerHTML = "";
    embeddingError.hidden = true;
    renderSeedInfo();
    renderBenchmark();
    renderAutoPlay();
    setBaselineLoading(true);
    setAutoPlayLoading(true);

    const params = new URLSearchParams({
      artist,
      album,
      k: "5",
      discovery: discoveryMode.checked ? "true" : "false",
    });

    fetchJson(`/recommend/embedding?${params}`)
      .then((data) => {
        if (thisRequest !== requestId) return;
        embeddingPayload = data;
        renderPrimaryList(data);
        renderSeedInfo();
        renderBenchmark();
        renderAutoPlay();
      })
      .catch((err) => {
        if (thisRequest !== requestId) return;
        embeddingPayload = { error: err.message };
        renderPrimaryList(embeddingPayload);
        renderSeedInfo();
        renderBenchmark();
        renderAutoPlay();
      })
      .finally(() => {
        if (thisRequest === requestId) setAutoPlayLoading(false);
      });

    fetchJson(`/recommend/lastfm?${params}`)
      .then((data) => {
        if (thisRequest !== requestId) return;
        lastfmPayload = data;
        renderSeedInfo();
        renderBenchmark();
      })
      .catch((err) => {
        if (thisRequest !== requestId) return;
        lastfmPayload = { error: err.message };
        renderSeedInfo();
        renderBenchmark();
      })
      .finally(() => {
        if (thisRequest === requestId) setBaselineLoading(false);
      });
  }

  async function explainRecommendation(button, rec) {
    if (!seed) return;

    document
      .querySelectorAll(".rec-list--primary .row.active")
      .forEach((el) => el.classList.remove("active"));
    button.classList.add("active");

    const thisExplain = ++explainRequestId;
    setTab("explain");
    explainEmpty.hidden = true;
    explainBody.hidden = false;
    explainPair.hidden = false;
    explainSeedLabel.textContent = albumLabel(seed.artist, seed.album);
    explainRecLabel.textContent = albumLabel(rec.artist, rec.album);
    explainList.innerHTML = "";
    explainError.hidden = true;
    explainError.textContent = "";
    setExplainLoading(true);

    try {
      const data = await fetchJson("/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: { artist: seed.artist, album: seed.album },
          recommendation: { artist: rec.artist, album: rec.album },
          n: 3,
        }),
      });
      if (thisExplain !== explainRequestId) return;
      renderQualities(data.qualities);
      if (!data.qualities || !data.qualities.length) {
        explainError.textContent = "No shared qualities returned for this pair.";
        explainError.hidden = false;
      }
    } catch (err) {
      if (thisExplain !== explainRequestId) return;
      explainList.innerHTML = "";
      explainError.textContent = err.message || "Explanation failed";
      explainError.hidden = false;
    } finally {
      if (thisExplain === explainRequestId) setExplainLoading(false);
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
    if (!queryInput.value.trim()) {
      resetPage();
      return;
    }
    debounceTimer = setTimeout(() => searchAlbums(queryInput.value), 250);
  });

  queryInput.addEventListener("search", () => {
    if (!queryInput.value.trim()) resetPage();
  });

  queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideSuggestions();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search")) hideSuggestions();
  });
})();
