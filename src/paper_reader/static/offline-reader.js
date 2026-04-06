(function () {
  const manifestNode = document.getElementById("offline-manifest");
  const treeRoot = document.getElementById("offline-paper-tree");
  const viewerRoot = document.getElementById("offline-viewer-root");
  const searchInput = document.getElementById("offline-search-input");
  const paperCountNode = document.getElementById("offline-paper-count");
  const promptCountNode = document.getElementById("offline-prompt-count");

  if (!manifestNode || !treeRoot || !viewerRoot) {
    return;
  }

  const manifest = JSON.parse(manifestNode.textContent || "{\"papers\":[],\"prompts\":[]}");
  const papers = Array.isArray(manifest.papers) ? manifest.papers.slice() : [];
  const prompts = Array.isArray(manifest.prompts) ? manifest.prompts.slice() : [];
  const paperMap = new Map(papers.map((paper) => [paper.rel_path, paper]));

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function formatBytes(size) {
    const numeric = Number(size) || 0;
    const units = ["B", "KB", "MB", "GB"];
    let value = numeric;
    for (let index = 0; index < units.length; index += 1) {
      if (value < 1024 || index === units.length - 1) {
        if (index === 0) return `${Math.round(value)} ${units[index]}`;
        return `${value.toFixed(1)} ${units[index]}`;
      }
      value /= 1024;
    }
    return `${numeric} B`;
  }

  function getHashState() {
    const hash = new URLSearchParams((window.location.hash || "").replace(/^#/, ""));
    return {
      paper: hash.get("paper") || "",
      tab: hash.get("tab") || "source",
      q: hash.get("q") || "",
    };
  }

  function setHashState(next) {
    const params = new URLSearchParams();
    if (next.paper) params.set("paper", next.paper);
    if (next.tab && next.tab !== "source") params.set("tab", next.tab);
    if (next.q) params.set("q", next.q);
    const nextHash = params.toString();
    if (`#${nextHash}` !== window.location.hash) {
      window.location.hash = nextHash;
    } else {
      render();
    }
  }

  function buildGroups(items) {
    const groups = new Map();
    items.forEach((paper) => {
      const yearKey = paper.sort_date ? String(paper.sort_date).slice(0, 4) : "unknown";
      const monthKey = paper.sort_date ? String(paper.sort_date).slice(0, 7) : "unknown";
      if (!groups.has(yearKey)) {
        groups.set(yearKey, {
          key: yearKey,
          label: yearKey === "unknown" ? "未提取日期" : yearKey,
          months: new Map(),
        });
      }
      const yearGroup = groups.get(yearKey);
      if (!yearGroup.months.has(monthKey)) {
        yearGroup.months.set(monthKey, {
          key: monthKey,
          label: monthKey === "unknown" ? "未分类" : monthKey,
          papers: [],
        });
      }
      yearGroup.months.get(monthKey).papers.push(paper);
    });

    const yearKeys = Array.from(groups.keys()).sort((left, right) => {
      if (left === "unknown") return 1;
      if (right === "unknown") return -1;
      return right.localeCompare(left);
    });

    return yearKeys.map((yearKey, yearIndex) => {
      const yearGroup = groups.get(yearKey);
      const monthKeys = Array.from(yearGroup.months.keys()).sort((left, right) => {
        if (left === "unknown") return 1;
        if (right === "unknown") return -1;
        return right.localeCompare(left);
      });
      return {
        key: yearKey,
        label: yearGroup.label,
        isOpen: yearIndex === 0,
        months: monthKeys.map((monthKey, monthIndex) => ({
          ...yearGroup.months.get(monthKey),
          isOpen: yearIndex === 0 && monthIndex === 0,
        })),
      };
    });
  }

  function filteredPapers(query) {
    const text = String(query || "").trim().toLowerCase();
    if (!text) return papers.slice();
    return papers.filter((paper) => (
      [paper.display_title, paper.file_name, paper.extracted_date, paper.folder]
        .join(" ")
        .toLowerCase()
        .includes(text)
    ));
  }

  function renderSidebar(items, selectedPaper) {
    if (!items.length) {
      treeRoot.innerHTML = '<div class="offline-list-empty">没有匹配的论文。</div>';
      return;
    }

    const groups = buildGroups(items);
    treeRoot.innerHTML = `
      <div class="paper-compact-list paper-group-tree">
        ${groups.map((yearGroup) => `
          <details class="paper-year-group" ${yearGroup.isOpen ? "open" : ""}>
            <summary>
              <span>${escapeHtml(yearGroup.label)}</span>
              <small>${yearGroup.months.reduce((count, month) => count + month.papers.length, 0)}</small>
            </summary>
            <div class="paper-month-group-list">
              ${yearGroup.months.map((monthGroup) => `
                <details class="paper-month-group" ${monthGroup.isOpen || monthGroup.papers.some((paper) => paper.rel_path === selectedPaper) ? "open" : ""}>
                  <summary>
                    <span>${escapeHtml(monthGroup.label)}</span>
                    <small>${monthGroup.papers.length}</small>
                  </summary>
                  <div class="paper-month-list">
                    ${monthGroup.papers.map((paper) => `
                      <a class="paper-row ${paper.rel_path === selectedPaper ? "active" : ""}" href="#paper=${encodeURIComponent(paper.rel_path)}">
                        <div class="paper-row-top">
                          <span class="paper-row-type">${escapeHtml(paper.extension)}</span>
                          ${paper.extracted_date ? `<span class="paper-row-date">${escapeHtml(paper.extracted_date)}</span>` : ""}
                        </div>
                        <strong class="paper-row-title">${escapeHtml(paper.display_title)}</strong>
                        <span class="paper-row-file">${escapeHtml(paper.file_name)}</span>
                        <div class="paper-row-bottom">
                          <div class="paper-row-flags">
                            ${paper.folder ? `<span class="paper-row-folder">${escapeHtml(paper.folder)}</span>` : ""}
                          </div>
                          <span class="paper-row-summary">${Object.values(paper.prompt_results || {}).filter((item) => item.exists).length}/${prompts.length}</span>
                        </div>
                      </a>
                    `).join("")}
                  </div>
                </details>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    `;

    treeRoot.querySelectorAll(".paper-row").forEach((node) => {
      node.addEventListener("click", (event) => {
        event.preventDefault();
        const href = node.getAttribute("href") || "";
        const paper = new URLSearchParams(href.replace(/^#/, "")).get("paper") || "";
        setHashState({ paper, tab: "source", q: searchInput ? searchInput.value : "" });
      });
    });
  }

  function sourceHtml(paper) {
    const safeTitle = escapeHtml(paper.display_title);
    const sourceUrl = encodeURI(paper.source_rel_path || "");
    if (paper.preview_kind === "pdf") {
      return `
        <div class="offline-source-pane">
          <div class="source-viewer-head offline-source-actions">
            <div>
              <strong>源文件</strong>
              <span>PDF 原文预览</span>
            </div>
            <div class="offline-viewer-actions">
              <a class="button-link secondary" href="${sourceUrl}" target="_blank" rel="noreferrer">打开原文件</a>
            </div>
          </div>
          <iframe class="main-viewer" title="${safeTitle}" src="${sourceUrl}"></iframe>
        </div>
      `;
    }
    if (paper.preview_kind === "docx") {
      const paragraphs = (paper.preview_paragraphs || []).length
        ? paper.preview_paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")
        : "<p>没有提取到可预览的内容。</p>";
      return `
        <div class="offline-source-pane">
          <div class="source-viewer-head offline-source-actions">
            <div>
              <strong>源文件</strong>
              <span>Word 文档预览</span>
            </div>
            <div class="offline-viewer-actions">
              <a class="button-link secondary" href="${sourceUrl}" target="_blank" rel="noreferrer">打开原文件</a>
            </div>
          </div>
          <div class="text-viewer">
            <div class="docx-preview wide">${paragraphs}</div>
          </div>
        </div>
      `;
    }
    if (paper.preview_kind === "doc") {
      return `
        <div class="offline-source-pane">
          <div class="source-viewer-head offline-source-actions">
            <div>
              <strong>源文件</strong>
              <span>DOC 文件预览</span>
            </div>
            <div class="offline-viewer-actions">
              <a class="button-link secondary" href="${sourceUrl}" target="_blank" rel="noreferrer">打开原文件</a>
            </div>
          </div>
          <div class="text-viewer">
            <div class="docx-preview wide">
              <p>离线包保留了原始 .doc 文件，也保留了和在线阅读器一致的轻量预览体验。</p>
            </div>
          </div>
        </div>
      `;
    }
    return `
      <div class="viewer-empty">
        <h3>当前格式暂不支持预览</h3>
        <p>你仍然可以直接打开原始文件。</p>
        <a class="button-link secondary" href="${sourceUrl}" target="_blank" rel="noreferrer">打开原文件</a>
      </div>
    `;
  }

  function promptHtml(paper, tab) {
    const info = (paper.prompt_results || {})[tab.slug] || { exists: false };
    if (!info.exists) {
      return `
        <div class="viewer-empty prompt-empty-state">
          <h3>${escapeHtml(tab.name)}</h3>
          <p>这个 Prompt 还没有应用到当前论文上，所以这里暂时是空的。</p>
        </div>
      `;
    }
    const resultUrl = encodeURI(info.result_rel_path || "");
    const sourceUrl = encodeURI(paper.source_rel_path || "");
    return `
      <div class="prompt-viewer">
        <div class="prompt-viewer-head">
          <div>
            <strong>${escapeHtml(tab.name)}</strong>
            <span>${escapeHtml(info.updated_at || "已生成")}</span>
          </div>
          <div class="offline-viewer-actions">
            <a class="button-link secondary" href="${sourceUrl}" target="_blank" rel="noreferrer">打开原文件</a>
            <a class="button-link secondary" href="${resultUrl}" target="_blank" rel="noreferrer">打开 MD</a>
          </div>
        </div>
        <article class="markdown-viewer markdown-render">${info.html || ""}</article>
      </div>
    `;
  }

  function renderViewer(paper, selectedTab) {
    if (!paper) {
      viewerRoot.className = "viewer-empty";
      viewerRoot.innerHTML = `
        <h3>没有匹配的论文</h3>
        <p>请修改检索条件或从左侧选择一篇论文。</p>
      `;
      return;
    }

    const tabSlugs = new Set(["source"].concat(prompts.map((prompt) => prompt.slug)));
    const resolvedTab = tabSlugs.has(selectedTab) ? selectedTab : "source";

    viewerRoot.className = "";
    viewerRoot.classList.add("offline-viewer-content");
    viewerRoot.innerHTML = `
      <div class="viewer-head">
        <div class="offline-viewer-head-row">
          <div>
            <h2>${escapeHtml(paper.display_title)}</h2>
            <p class="viewer-subtitle">${escapeHtml(paper.file_name)}</p>
          </div>
          <div class="viewer-meta">
            ${paper.extracted_date ? `<span>${escapeHtml(paper.extracted_date)}</span>` : ""}
            <span>${escapeHtml(formatBytes(paper.file_size))}</span>
            <span>${Object.values(paper.prompt_results || {}).filter((item) => item.exists).length}/${prompts.length} 个 Prompt 已生成</span>
          </div>
        </div>
      </div>
      <div class="viewer-tabs">
        <a class="viewer-tab ${resolvedTab === "source" ? "active" : ""}" href="#paper=${encodeURIComponent(paper.rel_path)}">
          <span>源文件</span>
        </a>
        ${prompts.map((prompt) => {
          const info = (paper.prompt_results || {})[prompt.slug] || { exists: false };
          const tabClass = [
            "viewer-tab",
            resolvedTab === prompt.slug ? "active" : "",
            info.exists ? "" : "pending",
          ].filter(Boolean).join(" ");
          return `
            <a class="${tabClass}" href="#paper=${encodeURIComponent(paper.rel_path)}&tab=${encodeURIComponent(prompt.slug)}">
              <span>${escapeHtml(prompt.name)}</span>
              <small>${info.exists ? "已生成" : "未应用"}</small>
            </a>
          `;
        }).join("")}
      </div>
      ${resolvedTab === "source"
        ? sourceHtml(paper)
        : promptHtml(paper, prompts.find((prompt) => prompt.slug === resolvedTab) || { slug: resolvedTab, name: resolvedTab })}
    `;

    viewerRoot.querySelectorAll(".viewer-tab").forEach((node) => {
      node.addEventListener("click", (event) => {
        event.preventDefault();
        const href = node.getAttribute("href") || "";
        const hashParams = new URLSearchParams(href.replace(/^#/, ""));
        setHashState({
          paper: hashParams.get("paper") || paper.rel_path,
          tab: hashParams.get("tab") || "source",
          q: searchInput ? searchInput.value : "",
        });
      });
    });
  }

  function render() {
    const state = getHashState();
    if (searchInput && searchInput.value !== state.q) {
      searchInput.value = state.q;
    }
    const matchedPapers = filteredPapers(state.q);
    const selectedPaper = matchedPapers.find((paper) => paper.rel_path === state.paper) || matchedPapers[0] || null;

    renderSidebar(matchedPapers, selectedPaper ? selectedPaper.rel_path : "");
    renderViewer(selectedPaper, state.tab);

    if (paperCountNode) {
      paperCountNode.textContent = String(matchedPapers.length);
    }
    if (promptCountNode) {
      promptCountNode.textContent = String(prompts.length);
    }
  }

  if (searchInput) {
    searchInput.value = getHashState().q;
    searchInput.addEventListener("input", () => {
      const state = getHashState();
      setHashState({
        paper: state.paper,
        tab: state.tab,
        q: searchInput.value || "",
      });
    });
  }

  window.addEventListener("hashchange", render);
  render();
})();
