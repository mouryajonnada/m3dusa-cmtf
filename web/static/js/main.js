/**
 * Frontend Application Logic: M3DUSA vs CMTF Fake News Detector
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadSamplePresets();
    loadBenchmarkTable();
    setupEventListeners();

    // Run initial analysis on first sample after short delay
    setTimeout(() => {
        const firstPreset = document.querySelector(".chip");
        if (firstPreset) {
            firstPreset.click();
        }
    }, 400);
});

/* ==========================================================================
   Tab Navigation
   ========================================================================== */

function initTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const activePane = document.getElementById(`pane-${target}`);
            if (activePane) {
                activePane.classList.add("active");
            }
        });
    });
}

/* ==========================================================================
   Sample Presets
   ========================================================================== */

async function loadSamplePresets() {
    const container = document.getElementById("preset-chips");
    try {
        const res = await fetch("/api/samples");
        if (!res.ok) throw new Error("Failed to load samples");
        const samples = await res.json();

        container.innerHTML = "";
        samples.forEach(sample => {
            const chip = document.createElement("button");
            chip.className = "chip";
            const tagClass = sample.ground_truth === "FAKE" ? "chip-fake" : "chip-real";

            chip.innerHTML = `
                <span class="chip-tag ${tagClass}">${sample.ground_truth}</span>
                <span>${sample.title}</span>
            `;

            chip.addEventListener("click", () => {
                const textarea = document.getElementById("claim-input");
                textarea.value = sample.title;
                runAnalysis(sample.title);
            });

            container.appendChild(chip);
        });
    } catch (err) {
        console.error("Presets loading error:", err);
        container.innerHTML = `<span class="empty-hint">Failed to load preset claims.</span>`;
    }
}

/* ==========================================================================
   Live Inference Engine
   ========================================================================== */

function setupEventListeners() {
    const analyzeBtn = document.getElementById("btn-analyze");
    const clearBtn = document.getElementById("btn-clear");
    const textarea = document.getElementById("claim-input");

    analyzeBtn.addEventListener("click", () => {
        const text = textarea.value.trim();
        if (text) {
            runAnalysis(text);
        } else {
            textarea.focus();
        }
    });

    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        textarea.focus();
    });

    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            const text = textarea.value.trim();
            if (text) runAnalysis(text);
        }
    });
}

async function runAnalysis(text) {
    const analyzeBtn = document.getElementById("btn-analyze");
    analyzeBtn.classList.add("loading");

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || "Prediction request failed.");
        }

        const data = await res.json();
        renderResults(data);
    } catch (err) {
        console.error("Inference Error:", err);
        alert(`Analysis failed: ${err.message}`);
    } finally {
        analyzeBtn.classList.remove("loading");
    }
}

/* ==========================================================================
   Render Inference Results
   ========================================================================== */

function renderResults(data) {
    const { baseline, cmtf, agreement, hashtags, graph_stats } = data;

    // 1. Agreement Banner
    const banner = document.getElementById("agreement-banner");
    const agreementTitle = document.getElementById("agreement-title");
    const agreementDesc = document.getElementById("agreement-desc");
    const agreementIcon = document.getElementById("agreement-icon");

    banner.className = "agreement-banner " + (agreement ? "agree" : "disagree");
    if (agreement) {
        agreementIcon.textContent = "✓";
        agreementTitle.textContent = `Consensus: Both Models Predict ${cmtf ? cmtf.prediction : baseline.prediction}`;
        agreementDesc.textContent = `Baseline M3DUSA and Novel CMTF concordantly classify this claim as ${cmtf.prediction} with high certainty.`;
    } else {
        agreementIcon.textContent = "!";
        agreementTitle.textContent = "Model Divergence Detected";
        agreementDesc.textContent = `Baseline predicted ${baseline.prediction} (${baseline.confidence}%), while CMTF predicted ${cmtf.prediction} (${cmtf.confidence}%). Cross-modal attention extracted conflicting social context signals.`;
    }

    // 2. Baseline Model Card
    if (baseline) {
        const baseLabel = document.getElementById("base-label");
        const baseConf = document.getElementById("base-conf");
        const baseFakeProb = document.getElementById("base-prob-fake");
        const baseRealProb = document.getElementById("base-prob-real");
        const baseFakeBar = document.getElementById("base-bar-fake");
        const baseRealBar = document.getElementById("base-bar-real");

        baseLabel.textContent = baseline.prediction;
        baseLabel.className = `prediction-tag ${baseline.prediction === "REAL" ? "tag-real" : "tag-fake"}`;
        baseConf.textContent = `${baseline.confidence.toFixed(1)}% Conf.`;

        baseFakeProb.textContent = `${(baseline.prob_fake * 100).toFixed(1)}%`;
        baseRealProb.textContent = `${(baseline.prob_real * 100).toFixed(1)}%`;
        baseFakeBar.style.width = `${(baseline.prob_fake * 100).toFixed(1)}%`;
        baseRealBar.style.width = `${(baseline.prob_real * 100).toFixed(1)}%`;
    }

    // 3. CMTF Model Card
    if (cmtf) {
        const cmtfLabel = document.getElementById("cmtf-label");
        const cmtfConf = document.getElementById("cmtf-conf");
        const cmtfFakeProb = document.getElementById("cmtf-prob-fake");
        const cmtfRealProb = document.getElementById("cmtf-prob-real");
        const cmtfFakeBar = document.getElementById("cmtf-bar-fake");
        const cmtfRealBar = document.getElementById("cmtf-bar-real");

        cmtfLabel.textContent = cmtf.prediction;
        cmtfLabel.className = `prediction-tag ${cmtf.prediction === "REAL" ? "tag-real" : "tag-fake"}`;
        cmtfConf.textContent = `${cmtf.confidence.toFixed(1)}% Conf.`;

        cmtfFakeProb.textContent = `${(cmtf.prob_fake * 100).toFixed(1)}%`;
        cmtfRealProb.textContent = `${(cmtf.prob_real * 100).toFixed(1)}%`;
        cmtfFakeBar.style.width = `${(cmtf.prob_fake * 100).toFixed(1)}%`;
        cmtfRealBar.style.width = `${(cmtf.prob_real * 100).toFixed(1)}%`;
    }

    // 4. Graph Metadata Badges
    document.getElementById("badge-news-nodes").textContent = `News Nodes: ${graph_stats.num_news_nodes}`;
    document.getElementById("badge-hash-nodes").textContent = `Hashtags: ${graph_stats.num_hashtag_nodes}`;
    document.getElementById("badge-edges").textContent = `Edges: ${graph_stats.num_edges}`;

    // 5. Extracted Hashtags List
    const hashtagContainer = document.getElementById("hashtags-list");
    if (hashtags && hashtags.length > 0) {
        hashtagContainer.innerHTML = hashtags
            .map(tag => `<span class="entity-tag">#${tag}</span>`)
            .join("");
    } else {
        hashtagContainer.innerHTML = `<span class="empty-hint">No explicit hashtags in text (single claim graph fallback).</span>`;
    }

    // 6. Render SVG Graph Layout
    renderSvgGraph(hashtags || []);
}

/* ==========================================================================
   Dynamic SVG Graph Visualizer
   ========================================================================== */

function renderSvgGraph(hashtags) {
    const svg = document.getElementById("graph-svg");
    const width = svg.clientWidth || 600;
    const height = 220;
    const centerX = width / 2;
    const centerY = height / 2;

    svg.innerHTML = "";

    // If no hashtags, show single central claim node
    if (hashtags.length === 0) {
        svg.innerHTML = `
            <circle cx="${centerX}" cy="${centerY}" r="28" fill="#3B82F6" opacity="0.9" stroke="#60A5FA" stroke-width="2"/>
            <circle cx="${centerX}" cy="${centerY}" r="38" fill="none" stroke="#3B82F6" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"/>
            <text x="${centerX}" y="${centerY + 5}" text-anchor="middle" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="700" font-size="12">CLAIM</text>
            <text x="${centerX}" y="${centerY + 50}" text-anchor="middle" fill="#94A3B8" font-size="11">Single Node Evidence Graph (No hashtags)</text>
        `;
        return;
    }

    // Draw connected graph
    const radius = Math.min(centerX - 60, 80);
    const angleStep = (2 * Math.PI) / hashtags.length;

    let edgesSvg = "";
    let nodesSvg = "";

    hashtags.forEach((tag, idx) => {
        const angle = idx * angleStep - Math.PI / 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);

        // Edge line
        edgesSvg += `
            <line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" 
                  stroke="#8B5CF6" stroke-width="2" stroke-opacity="0.6" stroke-dasharray="3 3"/>
        `;

        // Hashtag node
        nodesSvg += `
            <g transform="translate(${x}, ${y})">
                <circle r="18" fill="#8B5CF6" opacity="0.85" stroke="#C4B5FD" stroke-width="1.5"/>
                <text y="4" text-anchor="middle" fill="#FFFFFF" font-family="'Inter', sans-serif" font-size="9" font-weight="600">#${tag.slice(0, 4)}</text>
                <text y="28" text-anchor="middle" fill="#CBD5E1" font-size="10">#${tag}</text>
            </g>
        `;
    });

    // Central Claim Node
    const centerNodeSvg = `
        <circle cx="${centerX}" cy="${centerY}" r="26" fill="#3B82F6" stroke="#93C5FD" stroke-width="2.5"/>
        <text x="${centerX}" y="${centerY + 4}" text-anchor="middle" fill="#FFFFFF" font-family="'Outfit', sans-serif" font-weight="700" font-size="11">NEWS</text>
    `;

    svg.innerHTML = edgesSvg + centerNodeSvg + nodesSvg;
}

/* ==========================================================================
   Benchmark Table
   ========================================================================== */

async function loadBenchmarkTable() {
    const tbody = document.querySelector("#benchmark-table tbody");
    try {
        const res = await fetch("/api/benchmark");
        if (!res.ok) throw new Error("Failed to load benchmark data");
        const rows = await res.json();

        tbody.innerHTML = "";
        rows.forEach(row => {
            const tr = document.createElement("tr");
            const isLoss = row.Metric === "Cross-Entropy Loss";
            const deltaPositive = isLoss ? row.Delta_Absolute <= 0 : row.Delta_Absolute >= 0;
            const deltaClass = deltaPositive ? "positive" : "negative";

            tr.innerHTML = `
                <td>${row.Metric}</td>
                <td>${row.Baseline_M3DUSA}</td>
                <td><strong>${row.CMTF_Novel}</strong></td>
                <td class="${deltaClass}">${row.Delta_Absolute > 0 ? "+" : ""}${row.Delta_Absolute}</td>
                <td class="${deltaClass} font-bold">${row.Delta_Pct}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Benchmark table error:", err);
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: #94A3B8;">Failed to load benchmark table.</td></tr>`;
    }
}
