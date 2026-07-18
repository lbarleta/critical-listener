(() => {
  const queryInput = document.getElementById("album-query");
  const suggestions = document.getElementById("suggestions");
  const results = document.getElementById("results");
  const secondary = document.getElementById("secondary");
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
  const baselineSpinners = document.querySelectorAll(
    "#tab-lastfm .tab-spinner, #tab-metrics .tab-spinner"
  );
  const explainSpinner = document.querySelector("#tab-explain .tab-spinner");

  let seed = null;
  let debounceTimer = null;
  let requestId = 0;
  let explainRequestId = 0;

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

  function renderQualities(qualities) {
    explainList.innerHTML = "";
    for (const item of qualities || []) {
      const li = document.createElement("li");
      li.innerHTML = `
        <p class="quality"></p>
        <blockquote class="seed-quote"></blockquote>
        <blockquote class="rec-quote"></blockquote>
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
    explainRequestId += 1;
    setExplainLoading(false);
    explainEmpty.hidden = false;
    explainBody.hidden = true;
    explainList.innerHTML = "";
    explainPair.textContent = "";
    explainError.hidden = true;
    explainError.textContent = "";
  }

  function resetPage() {
    seed = null;
    requestId += 1;
    hideSuggestions();
    document.body.classList.remove("has-results");
    results.hidden = true;
    secondary.hidden = true;
    setBaselineLoading(false);
    embeddingList.innerHTML = "";
    lastfmList.innerHTML = "";
    embeddingError.hidden = true;
    lastfmError.hidden = true;
    resetExplain();
    setTab("explain");
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
    setBaselineLoading(true);

    const params = new URLSearchParams({
      artist,
      album,
      k: "5",
      discovery: discoveryMode.checked ? "true" : "false",
    });

    fetchJson(`/recommend/embedding?${params}`)
      .then((data) => {
        if (thisRequest !== requestId) return;
        renderPrimaryList(data);
      })
      .catch((err) => {
        if (thisRequest !== requestId) return;
        renderPrimaryList({ error: err.message });
      });

    fetchJson(`/recommend/lastfm?${params}`)
      .then((data) => {
        if (thisRequest !== requestId) return;
        renderLastfmList(data);
      })
      .catch((err) => {
        if (thisRequest !== requestId) return;
        renderLastfmList({ error: err.message });
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
    explainPair.textContent = `${albumLabel(seed.artist, seed.album)} → ${albumLabel(rec.artist, rec.album)}`;
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
