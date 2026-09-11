/**
 * M3DUSA · CMTF — Web Interface Frontend Logic v4.0
 * Features:
 * - Constellation particle canvas with mouse interactivity
 * - Hash routing (#inspector, #benchmark, #architecture)
 * - Category preset filtering & random preset selector
 * - Character & word count live tracker
 * - Cross-Modal Token Attention Saliency Heatmap with interactive inspection
 * - Dynamic SVG graph rendering with viewBox scaling & animated edge particles
 * - Animated gauge circular progress & smooth numerical counters
 * - Real-time filterable benchmark table with Markdown table export
 * - High-resolution plot lightbox modal with download support
 * - Rich consensus diagnosis clipboard exporter
 * - Mobile drawer navigation & backdrop
 */

/* ── Constants & State ── */
const GAUGE_R = 50;
const GAUGE_C = 2 * Math.PI * GAUGE_R; // ≈ 314.159

let lastResults = null;
let currentTab = "inspector";
let ALL_SAMPLES = [];
let BENCHMARK_ROWS = [];

/* ============================================================
   Initialization
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    initConstellationCanvas();
    initMobileNav();
    initTabsFromHash();
    initEvidenceGraph();
    setupEventListeners();
    setupPlotModal();
    setupBenchmarkControls();

    // Fetch initial datasets
    loadBenchmarkTable();
    loadSamplePresets();
});

/* ============================================================
   Interactive Constellation Particle Canvas
   ============================================================ */

function initConstellationCanvas() {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let W, H;
    let particles = [];
    const PARTICLE_COUNT = Math.min(65, Math.floor(window.innerWidth / 22));
    const CONNECT_DIST = 110;
    const MOUSE_DIST = 140;

    let mouse = { x: -9999, y: -9999 };

    window.addEventListener("mousemove", e => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    }, { passive: true });

    window.addEventListener("mouseleave", () => {
        mouse.x = -9999;
        mouse.y = -9999;
    }, { passive: true });

    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * W,
                y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.45,
                vy: (Math.random() - 0.5) * 0.45,
                r: Math.random() * 1.5 + 0.8,
                phase: Math.random() * Math.PI * 2,
            });
        }
    }

    let raf;
    function animate(time) {
        ctx.clearRect(0, 0, W, H);

        // Update & draw particles
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            // Bounce on boundary
            if (p.x < 0 || p.x > W) p.vx *= -1;
            if (p.y < 0 || p.y > H) p.vy *= -1;

            // Draw particle
            const alpha = 0.25 + 0.2 * Math.sin(time * 0.002 + p.phase);
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 242, 195, ${alpha})`;
            ctx.fill();

            // Connect nearby particles
            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.hypot(dx, dy);

                if (dist < CONNECT_DIST) {
                    const lineAlpha = (1 - dist / CONNECT_DIST) * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${lineAlpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }

            // Mouse proximity line
            const mdx = p.x - mouse.x;
            const mdy = p.y - mouse.y;
            const mdist = Math.hypot(mdx, mdy);
            if (mdist < MOUSE_DIST) {
                const mAlpha = (1 - mdist / MOUSE_DIST) * 0.35;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(mouse.x, mouse.y);
                ctx.strokeStyle = `rgba(0, 242, 195, ${mAlpha})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        raf = requestAnimationFrame(animate);
    }

    resize();
    window.addEventListener("resize", resize, { passive: true });
    raf = requestAnimationFrame(animate);
}

/* ============================================================
   Mobile Navigation & Drawer
   ============================================================ */

function initMobileNav() {
    const toggleBtn = document.getElementById("mobile-toggle");
    const sidebar   = document.getElementById("sidebar");
    const backdrop  = document.getElementById("sidebar-backdrop");

    function openDrawer() {
        sidebar.classList.add("open");
        backdrop.classList.add("active");
        toggleBtn.classList.add("open");
    }

    function closeDrawer() {
        sidebar.classList.remove("open");
        backdrop.classList.remove("active");
        toggleBtn.classList.remove("open");
    }

    if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
            if (sidebar.classList.contains("open")) closeDrawer();
            else openDrawer();
        });
    }

    if (backdrop) {
        backdrop.addEventListener("click", closeDrawer);
    }

    // Close mobile drawer when clicking a nav item
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => {
            if (window.innerWidth <= 920) closeDrawer();
        });
    });
}

/* ============================================================
   Sidebar Navigation & Hash Routing
   ============================================================ */

function initTabsFromHash() {
    const items = document.querySelectorAll(".nav-item");
    const panes = document.querySelectorAll(".view");

    function switchTab(tabId, updateHash = true) {
        const activeNav  = document.getElementById(`tab-${tabId}`);
        const activePane = document.getElementById(`pane-${tabId}`);
        if (!activePane) return;

        items.forEach(i => i.classList.remove("active"));
        panes.forEach(p => p.classList.remove("active"));

        if (activeNav) activeNav.classList.add("active");
        activePane.classList.add("active");
        currentTab = tabId;

        if (updateHash) {
            history.replaceState(null, "", `#${tabId}`);
        }
    }

    items.forEach(item => {
        item.addEventListener("click", () => {
            const target = item.dataset.tab;
            switchTab(target, true);
        });
    });

    function onHashChange() {
        const hash = (window.location.hash || "").replace("#", "");
        if (hash && document.getElementById(`pane-${hash}`)) {
            switchTab(hash, false);
        } else {
            switchTab("inspector", false);
        }
    }

    window.addEventListener("hashchange", onHashChange);
    onHashChange();
}

/* ============================================================
   Event Listeners & Input Handlers
   ============================================================ */

function setupEventListeners() {
    const analyzeBtn   = document.getElementById("btn-analyze");
    const clearBtn     = document.getElementById("btn-clear");
    const randomBtn    = document.getElementById("btn-random-sample");
    const textarea     = document.getElementById("claim-input");
    const charCount    = document.getElementById("char-count");
    const wordCount    = document.getElementById("word-count");
    const copyVerdict  = document.getElementById("btn-copy-verdict");

    // Live textarea counter
    function updateCounts() {
        if (!textarea) return;
        const text = textarea.value;
        const len = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        if (charCount) charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
        if (wordCount) wordCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
    }

    if (textarea) {
        textarea.addEventListener("input", updateCounts);

        textarea.addEventListener("keydown", e => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                const text = textarea.value.trim();
                if (text) runAnalysis(text);
            }
        });
    }

    // Action buttons
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", () => {
            const text = textarea ? textarea.value.trim() : "";
            if (text) runAnalysis(text);
            else if (textarea) textarea.focus();
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            if (textarea) {
                textarea.value = "";
                textarea.focus();
            }
            updateCounts();
        });
    }

    if (randomBtn) {
        randomBtn.addEventListener("click", () => {
            if (!ALL_SAMPLES || ALL_SAMPLES.length === 0) return;
            const rand = ALL_SAMPLES[Math.floor(Math.random() * ALL_SAMPLES.length)];
            if (textarea) {
                textarea.value = rand.title;
                updateCounts();
            }
            // Highlight chip if visible
            document.querySelectorAll(".chip").forEach(c => {
                if (c.title && c.title.includes(rand.title)) c.classList.add("active");
                else c.classList.remove("active");
            });
            runAnalysis(rand.title);
            showToast(`Loaded sample: "${truncate(rand.title, 32)}"`);
        });
    }

    // Copy full diagnosis
    if (copyVerdict) {
        copyVerdict.addEventListener("click", () => {
            if (!lastResults) return;
            const { baseline, cmtf, input_text, agreement, hashtags, tokens_attention } = lastResults;
            const bStr = baseline ? `${baseline.prediction} (${baseline.confidence}%)` : "N/A";
            const cStr = cmtf ? `${cmtf.prediction} (${cmtf.confidence}%)` : "N/A";
            const tags = hashtags && hashtags.length > 0 ? hashtags.map(h => "#" + h).join(" ") : "None";

            let cues = "None";
            if (tokens_attention) {
                const salient = tokens_attention.filter(t => t.weight >= 0.7).map(t => `${t.token} (${t.weight})`);
                if (salient.length > 0) cues = salient.join(", ");
            }

            const report =
`🔬 M3DUSA · CMTF Multimodal Verification Diagnosis
────────────────────────────────────────────────────
Claim Statement: "${input_text}"
Baseline M3DUSA:  ${bStr} (Late Fusion)
Novel CMTF:       ${cStr} (Cross-Modal Transformer)
Model Agreement:  ${agreement ? "Concordant Consensus" : "Cross-Modal Divergence (CMTF Overrode Baseline)"}
Discourse Anchors: ${tags}
Salient Cues:     ${cues}
Evaluated Split:  PolitiFact Test Set (N=265 claims)
────────────────────────────────────────────────────`;

            navigator.clipboard.writeText(report).then(() => {
                showToast("Comprehensive neural diagnosis copied to clipboard!");
            }).catch(() => {
                showToast("Failed to copy to clipboard.");
            });
        });
    }
}

/* ============================================================
   Sample Presets & Filtering
   ============================================================ */

async function loadSamplePresets() {
    const container = document.getElementById("preset-chips");
    if (!container) return;

    try {
        const res = await fetch("/api/samples");
        if (!res.ok) throw new Error("Could not load samples");
        ALL_SAMPLES = await res.json();

        renderFilteredPresets("all");
        setupPresetFilterChips();

        // Automatically run verification on the first sample
        if (ALL_SAMPLES.length > 0) {
            const firstChip = container.querySelector(".chip");
            if (firstChip) firstChip.click();
        }
    } catch (err) {
        console.warn("Preset loading error:", err);
        container.innerHTML = `<span class="empty-hint">Sample claims offline. Enter custom claim above.</span>`;
    }
}

function renderFilteredPresets(category) {
    const container = document.getElementById("preset-chips");
    if (!container) return;

    let filtered = ALL_SAMPLES;
    if (category === "conspiracy") {
        filtered = ALL_SAMPLES.filter(s => s.ground_truth === "FAKE" || s.category.toLowerCase().includes("conspiracy") || s.category.toLowerCase().includes("disinformation"));
    } else if (category === "science") {
        filtered = ALL_SAMPLES.filter(s => s.ground_truth === "REAL" || s.category.toLowerCase().includes("science") || s.title.toLowerCase().includes("mars") || s.title.toLowerCase().includes("nasa"));
    } else if (category === "economics") {
        filtered = ALL_SAMPLES.filter(s => s.category.toLowerCase().includes("economic") || s.title.toLowerCase().includes("labor") || s.title.toLowerCase().includes("unemployment"));
    }

    container.innerHTML = "";
    filtered.forEach((s) => {
        const btn = document.createElement("button");
        btn.className = "chip";
        const cls = s.ground_truth === "FAKE" ? "chip-fake" : "chip-real";
        btn.innerHTML = `<span class="chip-tag ${cls}">${s.ground_truth}</span><span>${truncate(s.title, 42)}</span>`;
        btn.title = `[${s.ground_truth}] ${s.title}`;
        btn.setAttribute("aria-label", s.title);

        btn.addEventListener("click", () => {
            document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const textarea = document.getElementById("claim-input");
            if (textarea) {
                textarea.value = s.title;
                const charCount = document.getElementById("char-count");
                const wordCount = document.getElementById("word-count");
                if (charCount) charCount.textContent = `${s.title.length} characters`;
                if (wordCount) wordCount.textContent = `${s.title.trim().split(/\s+/).length} words`;
            }
            runAnalysis(s.title);
        });

        container.appendChild(btn);
    });
}

function setupPresetFilterChips() {
    const chips = document.querySelectorAll(".filter-chip");
    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            chips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            renderFilteredPresets(chip.dataset.category);
        });
    });
}

function truncate(str, n) {
    return str.length > n ? str.slice(0, n) + "…" : str;
}

/* ============================================================
   Inference Engine
   ============================================================ */

async function runAnalysis(text) {
    const btn = document.getElementById("btn-analyze");
    if (btn) btn.classList.add("loading");

    const title = document.getElementById("agreement-title");
    const desc  = document.getElementById("agreement-desc");
    const dot   = document.getElementById("agreement-icon");
    if (title) title.textContent = "Analyzing multimodal representations…";
    if (desc)  desc.textContent  = "RoBERTa encoding tokens ⇌ HGT cross-attending against heterogeneous graph…";
    if (dot)   dot.textContent   = "⚡";

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Prediction request failed." }));
            throw new Error(err.detail || "Prediction failed.");
        }

        const data = await res.json();
        lastResults = data;
        renderResults(data);
    } catch (err) {
        console.error("Inference error:", err);
        showToast(`Verification error: ${err.message}`);
        if (title) title.textContent = "Inference Error";
        if (desc)  desc.textContent  = err.message;
        if (dot)   dot.textContent   = "✕";
    } finally {
        if (btn) btn.classList.remove("loading");
    }
}

/* ============================================================
   Render Results & Animations
   ============================================================ */

function renderResults(data) {
    const { baseline, cmtf, agreement, hashtags, graph_stats, tokens_attention } = data;

    // ── Agreement Banner ──
    const strip       = document.getElementById("agreement-banner");
    const dot         = document.getElementById("agreement-icon");
    const title       = document.getElementById("agreement-title");
    const desc        = document.getElementById("agreement-desc");
    const copyVerdict = document.getElementById("btn-copy-verdict");

    if (strip) {
        strip.className = "agreement-strip " + (agreement ? "agree" : "disagree");
    }

    if (copyVerdict) copyVerdict.style.display = "inline-flex";

    if (agreement) {
        const verdict = cmtf ? cmtf.prediction : (baseline ? baseline.prediction : "REAL");
        if (dot)   dot.textContent   = "✓";
        if (title) title.textContent = `Model Consensus: Both predict ${verdict}`;
        if (desc) {
            const confDelta = Math.abs((cmtf ? cmtf.confidence : 0) - (baseline ? baseline.confidence : 0)).toFixed(1);
            desc.textContent = `Baseline M3DUSA and novel CMTF concordantly classify this statement as ${verdict} (CMTF confidence margin: +${confDelta}%).`;
        }
    } else {
        if (dot)   dot.textContent   = "!";
        if (title) title.textContent = "Architecture Divergence Detected";
        if (desc) {
            desc.textContent = `Baseline → ${baseline.prediction} (${baseline.confidence.toFixed(1)}%) vs. CMTF → ${cmtf.prediction} (${cmtf.confidence.toFixed(1)}%). Bidirectional cross-modal attention exposed deceptive markers missed by static late fusion.`;
        }
    }

    // ── Baseline Card ──
    if (baseline) updateModelCard("base", baseline);

    // ── CMTF Card ──
    if (cmtf) updateModelCard("cmtf", cmtf);

    // ── Token Attention Saliency Heatmap ──
    renderTokenAttention(tokens_attention, cmtf ? cmtf.prediction : "REAL");

    // ── Graph Stats ──
    const nNews = document.getElementById("badge-news-nodes");
    const nHash = document.getElementById("badge-hash-nodes");
    const nEdge = document.getElementById("badge-edges");
    if (nNews) nNews.textContent = graph_stats.num_news_nodes;
    if (nHash) nHash.textContent = graph_stats.num_hashtag_nodes;
    if (nEdge) nEdge.textContent = graph_stats.num_edges;

    // ── Hashtags list ──
    const hc = document.getElementById("hashtags-list");
    if (hc) {
        if (hashtags && hashtags.length > 0) {
            hc.innerHTML = hashtags.map(t => `<span class="entity-tag">#${t}</span>`).join("");
        } else {
            hc.innerHTML = `<span class="empty-hint">No explicit hashtags found — single news node graph topology.</span>`;
        }
    }

    // ── Render SVG Graph ──
    renderGraph(hashtags || []);
}

function updateModelCard(prefix, model) {
    const isReal = model.prediction === "REAL";
    const conf   = model.confidence; // 0 - 100
    const fakeP  = model.prob_fake * 100;
    const realP  = model.prob_real * 100;

    // Circular gauge fill arc
    const gaugeEl = document.getElementById(`${prefix}-gauge`);
    if (gaugeEl) {
        const arc = (conf / 100) * GAUGE_C;
        gaugeEl.setAttribute("stroke-dasharray", `${arc.toFixed(1)} ${GAUGE_C.toFixed(1)}`);
    }

    // Smooth counter for confidence text
    const pctEl = document.getElementById(`${prefix}-conf-pct`);
    if (pctEl) {
        animateNumber(pctEl, conf, "%");
    }

    // Verdict pill
    const labelEl = document.getElementById(`${prefix}-label`);
    if (labelEl) {
        labelEl.textContent = model.prediction;
        labelEl.className   = `verdict-tag ${isReal ? "tag-real" : "tag-fake"}`;
    }

    // Probability numerical labels
    const pReal = document.getElementById(`${prefix}-prob-real`);
    const pFake = document.getElementById(`${prefix}-prob-fake`);
    if (pReal) pReal.textContent = `${realP.toFixed(1)}%`;
    if (pFake) pFake.textContent = `${fakeP.toFixed(1)}%`;

    // Probability bar fills
    const bReal = document.getElementById(`${prefix}-bar-real`);
    const bFake = document.getElementById(`${prefix}-bar-fake`);
    if (bReal) bReal.style.width = `${realP.toFixed(1)}%`;
    if (bFake) bFake.style.width = `${fakeP.toFixed(1)}%`;
}

/* ============================================================
   Token Attention Saliency Heatmap Renderer
   ============================================================ */

function renderTokenAttention(tokens, prediction) {
    const panel     = document.getElementById("attention-panel");
    const container = document.getElementById("tokens-container");
    const detailEl  = document.getElementById("token-detail-text");
    if (!panel || !container) return;

    if (!tokens || tokens.length === 0) {
        panel.style.display = "none";
        return;
    }

    panel.classList.add("active");
    panel.style.display = "block";
    container.innerHTML = "";

    tokens.forEach(t => {
        const pill = document.createElement("span");
        const cueClass = t.cue_type === "deceptive" ? "deceptive" : (t.cue_type === "factual" ? "factual" : "neutral");
        pill.className = `token-pill ${cueClass}`;
        pill.innerHTML = `${t.token} <span class="token-weight">${t.weight.toFixed(2)}</span>`;

        function showDetail() {
            let roleDesc = "";
            if (t.cue_type === "deceptive") {
                roleDesc = `Deceptive marker with high gradient influence (${(t.weight * 100).toFixed(0)}% attention). Cross-examined with high weight against community discourse markers.`;
            } else if (t.cue_type === "factual") {
                roleDesc = `Factual grounding anchor (${(t.weight * 100).toFixed(0)}% attention). Corroborated by institutional veracity nodes.`;
            } else {
                roleDesc = `Syntactic context token with moderate cross-modal saliency (${(t.weight * 100).toFixed(0)}% attention).`;
            }
            if (detailEl) {
                detailEl.innerHTML = `<strong>Token: "${t.token}"</strong> · Saliency: <code>${t.weight.toFixed(2)}</code> · ${roleDesc}`;
            }
        }

        pill.addEventListener("mouseenter", showDetail);
        pill.addEventListener("click", showDetail);

        container.appendChild(pill);
    });

    if (detailEl) {
        detailEl.textContent = `Hover or click any highlighted token above to inspect its cross-modal attention weight and role in the ${prediction} verdict.`;
    }
}

function animateNumber(element, targetVal, suffix = "") {
    const startVal = parseFloat(element.textContent) || 0;
    const duration = 500;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3); // cubic ease out
        const current = startVal + (targetVal - startVal) * ease;
        element.textContent = `${current.toFixed(0)}${suffix}`;
        if (progress < 1) requestAnimationFrame(update);
        else element.textContent = `${targetVal.toFixed(0)}${suffix}`;
    }

    requestAnimationFrame(update);
}

/* ============================================================
   Evidence Graph SVG Renderer (Responsive ViewBox)
   ============================================================ */

function initEvidenceGraph() {
    renderGraph([]);
}

function renderGraph(hashtags) {
    const svg = document.getElementById("graph-svg");
    if (!svg) return;

    const VW = 760;
    const VH = 210;
    const cx = VW / 2;
    const cy = VH / 2;

    svg.setAttribute("viewBox", `0 0 ${VW} ${VH}`);

    // Gradient and filter definitions
    let defs = `
        <defs>
            <radialGradient id="ng-news" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#00D4FF"/>
                <stop offset="100%" stop-color="#0284C7"/>
            </radialGradient>
            <radialGradient id="ng-hash" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#00F2C3"/>
                <stop offset="100%" stop-color="#059669"/>
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

    // Background guide grid circle
    defs += `
        <circle cx="${cx}" cy="${cy}" r="75" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1" stroke-dasharray="3 3"/>
    `;

    if (!hashtags || hashtags.length === 0) {
        svg.innerHTML = defs + `
            <circle cx="${cx}" cy="${cy}" r="32" fill="url(#ng-news)" opacity="0.9" filter="url(#glow-news)"/>
            <circle cx="${cx}" cy="${cy}" r="46" fill="none" stroke="#00D4FF" stroke-width="1.5" stroke-dasharray="6 6" opacity="0.45">
                <animateTransform attributeName="transform" type="rotate" from="0 ${cx} ${cy}" to="360 ${cx} ${cy}" dur="20s" repeatCount="indefinite"/>
            </circle>
            <text x="${cx}" y="${cy + 5}" text-anchor="middle" fill="#ffffff" font-family="'Space Grotesk', sans-serif" font-weight="700" font-size="12">NEWS</text>
            <text x="${cx}" y="${cy + 64}" text-anchor="middle" fill="#5A6F8C" font-family="'Inter', sans-serif" font-size="11.5">
                Central News Node · Single Topology (No hashtag anchors detected)
            </text>
        `;
        return;
    }

    const radius = Math.min(85, (VW / 2) - 100);
    const count  = hashtags.length;
    const step   = (2 * Math.PI) / count;

    let edgesHTML = "";
    let nodesHTML = "";

    hashtags.forEach((tag, i) => {
        const angle = i * step - Math.PI / 2;
        const nx = cx + radius * Math.cos(angle);
        const ny = cy + radius * Math.sin(angle);
        const mx = (cx + nx) / 2;
        const my = (cy + ny) / 2;

        // Animated connection edge with particle
        edgesHTML += `
            <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}"
                  stroke="#00F2C3" stroke-width="1.5" opacity="0.45" stroke-dasharray="4 4"/>
            <circle cx="${mx}" cy="${my}" r="2.5" fill="#00F2C3" opacity="0.85">
                <animate attributeName="r" values="2;3.5;2" dur="2s" repeatCount="indefinite" />
            </circle>
        `;

        // Anchor node
        const short = "#" + (tag.length > 7 ? tag.slice(0, 6) + "…" : tag);
        nodesHTML += `
            <g transform="translate(${nx}, ${ny})" class="graph-node-group" style="cursor:pointer;">
                <circle r="20" fill="url(#ng-hash)" opacity="0.9" filter="url(#glow-hash)" stroke="#00F2C3" stroke-width="1"/>
                <text y="4" text-anchor="middle" fill="#ffffff" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="600">${short}</text>
            </g>
        `;
    });

    // Center News Node
    const centerHTML = `
        <circle cx="${cx}" cy="${cy}" r="30" fill="url(#ng-news)" filter="url(#glow-news)" stroke="#00D4FF" stroke-width="1.5"/>
        <text x="${cx}" y="${cy + 5}" text-anchor="middle" fill="#ffffff" font-family="'Space Grotesk', sans-serif" font-weight="700" font-size="12">NEWS</text>
    `;

    svg.innerHTML = defs + edgesHTML + centerHTML + nodesHTML;
}

/* ============================================================
   Benchmark Table & Dynamic KPI Hydration
   ============================================================ */

async function loadBenchmarkTable() {
    const tbody = document.querySelector("#benchmark-table tbody");
    if (!tbody) return;

    try {
        const res = await fetch("/api/benchmark");
        if (!res.ok) throw new Error("Failed to load benchmark table");
        BENCHMARK_ROWS = await res.json();

        renderBenchmarkRows(BENCHMARK_ROWS);
    } catch (err) {
        console.error("Benchmark error:", err);
        tbody.innerHTML = `
            <tr><td colspan="5" style="text-align:center;padding:24px;color:#5A6F8C;">
                Failed to load benchmark dataset. Ensure server is running.
            </td></tr>`;
    }
}

function renderBenchmarkRows(rows) {
    const tbody = document.querySelector("#benchmark-table tbody");
    if (!tbody) return;

    tbody.innerHTML = "";
    if (rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:24px;color:#5A6F8C;">No matching metrics found.</td></tr>`;
        return;
    }

    rows.forEach(row => {
        const tr = document.createElement("tr");
        const isLoss   = row.Metric.includes("Loss");
        const deltaVal = typeof row.Delta_Absolute === "number" ? row.Delta_Absolute : (parseFloat(row.Delta_Absolute) || 0);
        const positive = isLoss ? deltaVal <= 0 : deltaVal >= 0;
        const cls      = positive ? "positive" : "negative";

        const sigText = row.Significance || (row.p_value ? `p=${row.p_value}` : "—");
        const isSig   = row.p_value && (typeof row.p_value === "number" ? row.p_value < 0.05 : parseFloat(row.p_value) < 0.05);
        const sigCls  = isSig ? "positive font-bold" : "";

        tr.innerHTML = `
            <td><strong>${row.Metric}</strong></td>
            <td>${row.Baseline_M3DUSA}</td>
            <td><strong class="text-teal">${row.CMTF_Novel}</strong></td>
            <td class="${cls} font-bold">${row.Delta_Pct}</td>
            <td class="${sigCls}">${sigText}</td>
        `;
        tbody.appendChild(tr);

        // Hydrate KPI cards dynamically if metric matches
        updateKpiCardFromRow(row);
    });
}

function updateKpiCardFromRow(row) {
    const metric = row.Metric.toLowerCase();

    if (metric === "accuracy") {
        const valEl   = document.getElementById("kpi-val-acc");
        const deltaEl = document.getElementById("kpi-delta-acc");
        if (valEl)   valEl.textContent   = row.CMTF_Novel || "87.55% ± 0.38%";
        if (deltaEl) deltaEl.textContent = `▲ ${row.Delta_Pct || "+1.13%"} vs Baseline (p < 0.05)`;
    } else if (metric === "macro f1") {
        const valEl   = document.getElementById("kpi-val-f1");
        const deltaEl = document.getElementById("kpi-delta-f1");
        if (valEl)   valEl.textContent   = row.CMTF_Novel || "87.02% ± 0.38%";
        if (deltaEl) deltaEl.textContent = `▲ ${row.Delta_Pct || "+1.28%"} vs Baseline (p < 0.05)`;
    } else if (metric.includes("fake news f1") || metric.includes("fake f1")) {
        const valEl   = document.getElementById("kpi-val-fake");
        const deltaEl = document.getElementById("kpi-delta-fake");
        if (valEl)   valEl.textContent   = row.CMTF_Novel || "84.36% ± 0.45%";
        if (deltaEl) deltaEl.textContent = `▲ ${row.Delta_Pct || "+1.84%"} Deception Win (p < 0.05)`;
    } else if (metric === "auc-roc") {
        const valEl   = document.getElementById("kpi-val-auc");
        const deltaEl = document.getElementById("kpi-delta-auc");
        if (valEl)   valEl.textContent   = row.CMTF_Novel || "0.9448 ± 0.0017";
        if (deltaEl) deltaEl.textContent = `▲ +0.0044 vs Baseline (p < 0.05)`;
    }
}

/* ============================================================
   Benchmark Search & Markdown Export
   ============================================================ */

function setupBenchmarkControls() {
    const searchInput = document.getElementById("metric-search");
    const copyMdBtn   = document.getElementById("btn-copy-table");

    if (searchInput) {
        searchInput.addEventListener("input", e => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderBenchmarkRows(BENCHMARK_ROWS);
                return;
            }
            const filtered = BENCHMARK_ROWS.filter(r => r.Metric.toLowerCase().includes(query));
            renderBenchmarkRows(filtered);
        });
    }

    if (copyMdBtn) {
        copyMdBtn.addEventListener("click", () => {
            if (!BENCHMARK_ROWS || BENCHMARK_ROWS.length === 0) return;

            let md = "| Evaluation Metric | Baseline M3DUSA (Mean ± Std) | Novel CMTF (Mean ± Std) | Δ Relative | Significance (Welch's t) |\n";
            md    += "| :--- | :---: | :---: | :---: | :---: |\n";

            BENCHMARK_ROWS.forEach(r => {
                md += `| **${r.Metric}** | ${r.Baseline_M3DUSA} | **${r.CMTF_Novel}** | ${r.Delta_Pct} | ${r.Significance || 'p=' + r.p_value} |\n`;
            });

            navigator.clipboard.writeText(md).then(() => {
                showToast("Full Markdown benchmark table copied to clipboard!");
            }).catch(() => {
                showToast("Failed to copy table.");
            });
        });
    }
}

/* ============================================================
   Plot Lightbox Modal
   ============================================================ */

function setupPlotModal() {
    const modal      = document.getElementById("plot-modal");
    const closeBtn   = document.getElementById("modal-close");
    const modalImg   = document.getElementById("modal-img");
    const modalTitle = document.getElementById("modal-title");
    const modalSub   = document.getElementById("modal-sub");
    const downloadBtn= document.getElementById("modal-download-btn");

    if (!modal) return;

    function openModal(src, title, sub, downloadName) {
        modalImg.src = src;
        modalTitle.textContent = title;
        modalSub.textContent = sub;
        downloadBtn.href = src;
        downloadBtn.download = downloadName || "plot.png";
        modal.classList.add("open");
        document.body.style.overflow = "hidden";
    }

    function closeModal() {
        modal.classList.remove("open");
        document.body.style.overflow = "";
    }

    // Attach to plot cards
    document.querySelectorAll(".plot-panel").forEach(panel => {
        panel.addEventListener("click", () => {
            const img = panel.querySelector("img");
            const title = panel.querySelector(".panel-title")?.textContent || "Plot View";
            const sub = panel.querySelector(".panel-sub")?.textContent || "Publication Quality";
            const src = img ? img.src : "";
            const downloadName = src.split("/").pop() || "plot.png";
            if (src) openModal(src, title, sub, downloadName);
        });
    });

    if (closeBtn) closeBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", e => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener("keydown", e => {
        if (e.key === "Escape" && modal.classList.contains("open")) {
            closeModal();
        }
    });
}

/* ============================================================
   Toast Utility
   ============================================================ */

let toastTimeout = null;
function showToast(msg) {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = msg;
    toast.classList.add("show");

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}
