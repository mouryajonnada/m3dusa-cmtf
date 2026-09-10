/**
 * M3DUSA · CMTF — Frontend Logic v2.0
 * New: animated SVG gauges, dot-grid canvas background, sidebar tabs
 */

/* ── Constants ── */
const GAUGE_R   = 50;          // SVG circle radius
const GAUGE_C   = 2 * Math.PI * GAUGE_R; // ≈ 314.16

/* ============================================================
   Initialisation
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    initBgCanvas();
    initTabs();
    loadSamplePresets();
    loadBenchmarkTable();
    setupEventListeners();

    // Auto-run first sample
    setTimeout(() => {
        const first = document.querySelector(".chip");
        if (first) first.click();
    }, 600);
});

/* ============================================================
   Animated Dot-Grid Background Canvas
   ============================================================ */

function initBgCanvas() {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let W, H, dots;
    const SPACING = 34;
    const DOT_R   = 1.1;

    function build() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
        dots = [];
        for (let x = SPACING; x < W; x += SPACING) {
            for (let y = SPACING; y < H; y += SPACING) {
                dots.push({ x, y, phase: Math.random() * Math.PI * 2 });
            }
        }
    }

    let raf;
    function draw(t) {
        ctx.clearRect(0, 0, W, H);
        const ts = t / 4000;
        for (const d of dots) {
            const alpha = 0.08 + 0.06 * Math.sin(ts + d.phase);
            ctx.beginPath();
            ctx.arc(d.x, d.y, DOT_R, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 217, 181, ${alpha})`;
            ctx.fill();
        }
        raf = requestAnimationFrame(draw);
    }

    build();
    window.addEventListener("resize", build);
    raf = requestAnimationFrame(draw);
}

/* ============================================================
   Sidebar Tab Navigation
   ============================================================ */

function initTabs() {
    const items = document.querySelectorAll(".nav-item");
    const panes = document.querySelectorAll(".view");

    items.forEach(item => {
        item.addEventListener("click", () => {
            const target = item.dataset.tab;

            items.forEach(i => i.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            item.classList.add("active");
            const pane = document.getElementById(`pane-${target}`);
            if (pane) pane.classList.add("active");
        });
    });
}

/* ============================================================
   Sample Presets
   ============================================================ */

async function loadSamplePresets() {
    const container = document.getElementById("preset-chips");
    try {
        const res = await fetch("/api/samples");
        if (!res.ok) throw new Error("Failed");
        const samples = await res.json();

        container.innerHTML = "";
        samples.forEach(s => {
            const btn = document.createElement("button");
            btn.className = "chip";
            const cls = s.ground_truth === "FAKE" ? "chip-fake" : "chip-real";
            btn.innerHTML = `<span class="chip-tag ${cls}">${s.ground_truth}</span><span>${truncate(s.title, 48)}</span>`;
            btn.title = s.title;
            btn.addEventListener("click", () => {
                document.getElementById("claim-input").value = s.title;
                runAnalysis(s.title);
            });
            container.appendChild(btn);
        });
    } catch (err) {
        container.innerHTML = `<span class="empty-hint">Presets unavailable.</span>`;
    }
}

function truncate(str, n) {
    return str.length > n ? str.slice(0, n) + "…" : str;
}

/* ============================================================
   Event Listeners
   ============================================================ */

function setupEventListeners() {
    const analyzeBtn = document.getElementById("btn-analyze");
    const clearBtn   = document.getElementById("btn-clear");
    const textarea   = document.getElementById("claim-input");

    analyzeBtn.addEventListener("click", () => {
        const text = textarea.value.trim();
        if (text) runAnalysis(text);
        else textarea.focus();
    });

    clearBtn.addEventListener("click", () => {
        textarea.value = "";
        textarea.focus();
    });

    textarea.addEventListener("keydown", e => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            const text = textarea.value.trim();
            if (text) runAnalysis(text);
        }
    });
}

/* ============================================================
   Inference Engine
   ============================================================ */

async function runAnalysis(text) {
    const btn = document.getElementById("btn-analyze");
    btn.classList.add("loading");

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction failed.");
        }

        const data = await res.json();
        renderResults(data);
    } catch (err) {
        console.error("Inference error:", err);
        alert(`Analysis failed: ${err.message}`);
    } finally {
        btn.classList.remove("loading");
    }
}

/* ============================================================
   Render Results
   ============================================================ */

function renderResults(data) {
    const { baseline, cmtf, agreement, hashtags, graph_stats } = data;

    // ── Agreement Strip ──
    const strip = document.getElementById("agreement-banner");
    const dot   = document.getElementById("agreement-icon");
    const title = document.getElementById("agreement-title");
    const desc  = document.getElementById("agreement-desc");

    strip.className = "agreement-strip " + (agreement ? "agree" : "disagree");

    if (agreement) {
        const verdict = cmtf ? cmtf.prediction : baseline.prediction;
        dot.textContent   = "✓";
        title.textContent = `Consensus: Both models predict ${verdict}`;
        desc.textContent  = `Baseline M3DUSA and CMTF concordantly classify this claim as ${verdict}.`;
    } else {
        dot.textContent   = "!";
        title.textContent = "Model Divergence Detected";
        desc.textContent  = `Baseline → ${baseline.prediction} (${baseline.confidence.toFixed(1)}%) · CMTF → ${cmtf.prediction} (${cmtf.confidence.toFixed(1)}%). Cross-modal attention captured conflicting social context signals.`;
    }

    // ── Baseline Card ──
    if (baseline) updateModelCard("base", baseline);

    // ── CMTF Card ──
    if (cmtf) updateModelCard("cmtf", cmtf);

    // ── Graph Stats ──
    document.getElementById("badge-news-nodes").textContent = graph_stats.num_news_nodes;
    document.getElementById("badge-hash-nodes").textContent = graph_stats.num_hashtag_nodes;
    document.getElementById("badge-edges").textContent      = graph_stats.num_edges;

    // ── Hashtag Chips ──
    const hc = document.getElementById("hashtags-list");
    if (hashtags && hashtags.length > 0) {
        hc.innerHTML = hashtags.map(t => `<span class="entity-tag">#${t}</span>`).join("");
    } else {
        hc.innerHTML = `<span class="empty-hint">No explicit hashtags — single-node graph fallback.</span>`;
    }

    // ── SVG Graph ──
    renderGraph(hashtags || []);
}

function updateModelCard(prefix, model) {
    const isReal = model.prediction === "REAL";
    const conf   = model.confidence;          // already 0-100
    const fakeP  = (model.prob_fake * 100);
    const realP  = (model.prob_real * 100);

    // Gauge arc
    const gaugeEl = document.getElementById(`${prefix}-gauge`);
    const arc     = (conf / 100) * GAUGE_C;
    if (gaugeEl) {
        gaugeEl.setAttribute("stroke-dasharray", `${arc.toFixed(2)} ${GAUGE_C.toFixed(2)}`);
    }

    // Gauge centre text
    const pctEl = document.getElementById(`${prefix}-conf-pct`);
    if (pctEl) pctEl.textContent = `${conf.toFixed(0)}%`;

    // Verdict pill
    const labelEl = document.getElementById(`${prefix}-label`);
    if (labelEl) {
        labelEl.textContent = model.prediction;
        labelEl.className   = `verdict-tag ${isReal ? "tag-real" : "tag-fake"}`;
    }

    // Probability bars
    document.getElementById(`${prefix}-prob-real`).textContent = `${realP.toFixed(1)}%`;
    document.getElementById(`${prefix}-prob-fake`).textContent = `${fakeP.toFixed(1)}%`;
    document.getElementById(`${prefix}-bar-real`).style.width  = `${realP.toFixed(1)}%`;
    document.getElementById(`${prefix}-bar-fake`).style.width  = `${fakeP.toFixed(1)}%`;
}

/* ============================================================
   Evidence Graph SVG Renderer
   ============================================================ */

function renderGraph(hashtags) {
    const svg  = document.getElementById("graph-svg");
    const W    = svg.clientWidth  || 700;
    const H    = 200;
    const cx   = W / 2;
    const cy   = H / 2;

    svg.innerHTML = "";

    // Defs: gradients + glows
    svg.innerHTML = `
        <defs>
            <radialGradient id="ng-news" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#38BDF8"/>
                <stop offset="100%" stop-color="#0369A1"/>
            </radialGradient>
            <radialGradient id="ng-hash" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#00D9B5"/>
                <stop offset="100%" stop-color="#047857"/>
            </radialGradient>
            <filter id="glow-news" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-hash" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
    `;

    if (hashtags.length === 0) {
        svg.innerHTML += `
            <circle cx="${cx}" cy="${cy}" r="32" fill="url(#ng-news)" opacity="0.9" filter="url(#glow-news)"/>
            <circle cx="${cx}" cy="${cy}" r="44" fill="none" stroke="#38BDF8" stroke-width="1.5"
                    stroke-dasharray="5 5" opacity="0.4">
                <animateTransform attributeName="transform" type="rotate"
                    from="0 ${cx} ${cy}" to="360 ${cx} ${cy}" dur="18s" repeatCount="indefinite"/>
            </circle>
            <text x="${cx}" y="${cy+5}" text-anchor="middle" fill="#fff"
                  font-family="'Space Grotesk',sans-serif" font-weight="700" font-size="11">NEWS</text>
            <text x="${cx}" y="${cy+52}" text-anchor="middle" fill="#4D6080" font-size="11">
                Single-node graph · no hashtags detected
            </text>
        `;
        return;
    }

    const maxRadius = Math.min((W / 2) - 64, 78);
    const step      = (2 * Math.PI) / hashtags.length;

    let edgesHTML = "", nodesHTML = "";

    hashtags.forEach((tag, i) => {
        const angle = i * step - Math.PI / 2;
        const nx    = cx + maxRadius * Math.cos(angle);
        const ny    = cy + maxRadius * Math.sin(angle);

        // animated edge
        edgesHTML += `
            <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}"
                  stroke="#00D9B5" stroke-width="1.5" opacity="0.35" stroke-dasharray="4 4">
            </line>
            <circle cx="${lerp(cx, nx, 0.5)}" cy="${lerp(cy, ny, 0.5)}" r="2.5"
                    fill="#00D9B5" opacity="0.6"/>
        `;

        // hashtag node
        const short = "#" + tag.slice(0, 5);
        nodesHTML += `
            <g transform="translate(${nx}, ${ny})">
                <circle r="20" fill="url(#ng-hash)" opacity="0.85" filter="url(#glow-hash)"
                        stroke="#00D9B5" stroke-width="1"/>
                <text y="4" text-anchor="middle" fill="#fff"
                      font-family="'JetBrains Mono',monospace" font-size="9" font-weight="600">${short}</text>
                <text y="32" text-anchor="middle" fill="#8899BB" font-size="10">#${tag}</text>
            </g>
        `;
    });

    // Central news node
    const centerHTML = `
        <circle cx="${cx}" cy="${cy}" r="28" fill="url(#ng-news)" filter="url(#glow-news)"
                stroke="#38BDF8" stroke-width="1.5"/>
        <text x="${cx}" y="${cy+5}" text-anchor="middle" fill="#fff"
              font-family="'Space Grotesk',sans-serif" font-weight="700" font-size="11">NEWS</text>
    `;

    svg.innerHTML += edgesHTML + centerHTML + nodesHTML;
}

function lerp(a, b, t) { return a + (b - a) * t; }

/* ============================================================
   Benchmark Table
   ============================================================ */

async function loadBenchmarkTable() {
    const tbody = document.querySelector("#benchmark-table tbody");
    if (!tbody) return;

    try {
        const res = await fetch("/api/benchmark");
        if (!res.ok) throw new Error("Failed to load benchmark");
        const rows = await res.json();

        tbody.innerHTML = "";
        rows.forEach(row => {
            const tr = document.createElement("tr");
            const isLoss    = row.Metric === "Cross-Entropy Loss";
            const positive  = isLoss ? row.Delta_Absolute <= 0 : row.Delta_Absolute >= 0;
            const cls       = positive ? "positive" : "negative";

            tr.innerHTML = `
                <td>${row.Metric}</td>
                <td>${row.Baseline_M3DUSA}</td>
                <td><strong>${row.CMTF_Novel}</strong></td>
                <td class="${cls}">${row.Delta_Absolute > 0 ? "+" : ""}${row.Delta_Absolute}</td>
                <td class="${cls} font-bold">${row.Delta_Pct}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Benchmark error:", err);
        if (tbody) tbody.innerHTML = `
            <tr><td colspan="5" style="text-align:center;padding:24px;color:#4D6080;">
                Failed to load benchmark data.
            </td></tr>`;
    }
}
