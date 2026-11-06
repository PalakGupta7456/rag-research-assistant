// Paper Trail: hybrid retrieval and cited answers, entirely in the browser.
// BM25 + MiniLM embeddings (Transformers.js) fused with Reciprocal Rank Fusion.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const STOP = new Set(("a an and are as at be by for from has have in is it its of on or that the this to was were with we our these those " +
  "their they which using use based via can into than also such not more new paper propose proposed show how what why do does").split(" "));
const tokenize = (t) => (t.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g) || []).filter((w) => w.length > 1 && !STOP.has(w));
const splitSentences = (t) => t.split(/(?<=[.!?])\s+(?=[A-Z])/).filter((s) => s.length > 40);
const EXAMPLES = [
  "How are people reducing hallucination in retrieval-augmented generation?",
  "What's new in efficient attention for long contexts?",
  "Uncertainty estimates for LLM outputs",
  "Diffusion models for video generation",
  "Evaluating AI agents on real software tasks",
];

let papers = [], emb = null, dim = 384, bm25 = null, embedder = null, lastHits = [], lastQuery = "";

// ---------- BM25 (same formula as rag/retrieval.py) ----------
function buildBM25(docs, k1 = 1.5, b = 0.75) {
  const tfs = docs.map((d) => { const m = new Map(); for (const w of tokenize(d)) m.set(w, (m.get(w) || 0) + 1); return m; });
  const lens = tfs.map((m) => [...m.values()].reduce((a, c) => a + c, 0));
  const avg = lens.reduce((a, c) => a + c, 0) / Math.max(lens.length, 1);
  const df = new Map();
  for (const m of tfs) for (const w of m.keys()) df.set(w, (df.get(w) || 0) + 1);
  const n = docs.length, idf = new Map();
  for (const [w, f] of df) idf.set(w, Math.log(1 + (n - f + 0.5) / (f + 0.5)));
  return (query) => {
    const q = tokenize(query), out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const norm = k1 * (1 - b + (b * lens[i]) / avg);
      let s = 0;
      for (const w of q) { const tf = tfs[i].get(w); if (tf) s += idf.get(w) * (tf * (k1 + 1)) / (tf + norm); }
      out[i] = s;
    }
    return out;
  };
}

const argsortDesc = (arr) => Array.from(arr.keys()).sort((a, b) => arr[b] - arr[a]);
function rrf(orders, k = 60) {
  const n = orders[0].length, out = new Float32Array(n);
  for (const order of orders) order.forEach((doc, rank) => { out[doc] += 1 / (k + rank + 1); });
  return out;
}
function denseScores(qv) {
  const n = papers.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) s += emb[off + j] * qv[j];
    out[i] = s / 127;
  }
  return out;
}
async function embed(texts) {
  const out = await embedder(texts, { pooling: "mean", normalize: true });
  return out.tolist();
}

// ---------- load ----------
async function init() {
  $("examples").innerHTML = EXAMPLES.map((e) => `<button type="button">${esc(e)}</button>`).join("");
  $("examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { $("q").value = b.textContent; ask(); } });
  try { const k = sessionStorage.getItem("anthropic-key"); if (k) $("key").value = k; } catch { /* storage unavailable */ }

  const [p, buf, stats] = await Promise.all([
    fetch("data/papers.json").then((r) => r.json()),
    fetch("data/embeddings.bin").then((r) => r.arrayBuffer()),
    fetch("data/stats.json").then((r) => r.json()),
  ]).catch(() => [null, null, null]);
  if (!p) { $("meta").textContent = "The index hasn't been published yet. The daily build creates it."; return; }
  papers = p; emb = new Int8Array(buf); dim = stats.dim;
  bm25 = buildBM25(papers.map((x) => `${x.t}. ${x.a}`));
  const days = Math.round((Date.now() - new Date(stats.generated_at)) / 86400000);
  $("meta").textContent = `${stats.n_papers.toLocaleString()} papers from ${stats.date_range[0]} to ${stats.date_range[1]}, refreshed ${days === 0 ? "today" : days + " d ago"}`;
  $("lede").textContent = `Search ${stats.n_papers.toLocaleString()} recent arXiv papers. Every answer is assembled from the papers it cites.`;
  renderBench(stats.benchmark);

  try {
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2");
    env.allowLocalModels = false;
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8" });
    $("model-status").textContent = "Semantic model ready, running in your browser";
    $("model-status").classList.add("ready");
  } catch (err) {
    console.warn(err);
    $("model-status").textContent = "Semantic model unavailable here, so keyword search only";
    document.querySelector('input[value="bm25"]').checked = true;
  }
}

// ---------- ask ----------
async function ask() {
  const q = $("q").value.trim();
  if (!q || !papers.length) return;
  lastQuery = q;
  let mode = document.querySelector('input[name="mode"]:checked').value;
  if (!embedder && mode !== "bm25") mode = "bm25";
  $("results").hidden = false;
  $("answer").innerHTML = `<h2>Answer</h2><p class="none">Searching…</p>`;
  $("llm-out").innerHTML = "";

  const lex = mode !== "dense" ? bm25(q) : null;
  const qv = mode !== "bm25" ? (await embed([q]))[0] : null;
  const dense = qv ? denseScores(qv) : null;
  const scores = mode === "bm25" ? lex : mode === "dense" ? dense : rrf([argsortDesc(lex), argsortDesc(dense)]);
  const top = argsortDesc(scores).slice(0, 5);
  lastHits = top.map((i) => papers[i]);

  // Extractive answer: the sentences closest to the question across the top papers, breadth first.
  const cands = top.flatMap((i, rank) => splitSentences(papers[i].a).map((s) => ({ s, rank })));
  let sims;
  if (qv) {
    const sv = await embed(cands.map((c) => c.s));
    sims = sv.map((v) => v.reduce((a, x, j) => a + x * qv[j], 0));
  } else {
    const qt = new Set(tokenize(q));
    sims = cands.map((c) => { const t = tokenize(c.s); return t.filter((w) => qt.has(w)).length / Math.sqrt(t.length + 1); });
  }
  const order = argsortDesc(Float32Array.from(sims));
  const picked = [], used = new Set();
  for (const j of order) {
    if (picked.length >= 4) break;
    if (qv && sims[j] < 0.3) break;
    const c = cands[j];
    if (used.has(c.rank) && used.size < Math.min(3, top.length)) continue;
    picked.push(c); used.add(c.rank);
  }
  const marks = new Map(picked.map((c) => [c.s, c.rank]));
  $("answer").innerHTML = `<h2>What the papers say</h2>` + (picked.length
    ? `<p>${picked.map((c) => `${esc(c.s)}<a class="cite" href="#src-${c.rank + 1}" data-n="${c.rank + 1}">[${c.rank + 1}]</a>`).join(" ")}</p>`
    : `<p class="none">None of the closest papers address this directly. Try different wording, or use Keyword mode for exact terms.</p>`);

  $("sources").innerHTML = top.map((i, r) => {
    const p = papers[i];
    let abs = esc(p.a);
    for (const [s, rank] of marks) if (rank === r) abs = abs.replace(esc(s), `<mark>${esc(s)}</mark>`);
    const why = mode === "hybrid" ? `keyword rank ${argsortDesc(lex).indexOf(i) + 1}, semantic rank ${argsortDesc(dense).indexOf(i) + 1}` : "";
    return `<li id="src-${r + 1}"><h3><a href="https://arxiv.org/abs/${esc(p.id)}" target="_blank" rel="noopener">${esc(p.t)}</a></h3>
      <p class="by">${esc(p.au)} · ${esc(p.c)} · ${esc(p.d)}</p><p class="abs">${abs}</p>${why ? `<p class="why">${why}</p>` : ""}</li>`;
  }).join("");
}

document.addEventListener("click", (e) => {
  const a = e.target.closest(".cite");
  if (!a) return;
  const li = $(`src-${a.dataset.n}`);
  li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash");
});

// ---------- optional: Claude writes the answer, restricted to the sources ----------
$("gen").addEventListener("click", async () => {
  const key = $("key").value.trim(), model = $("model").value.trim();
  if (!key || !lastHits.length) { $("llm-out").innerHTML = `<p class="err">Ask a question and enter your API key first.</p>`; return; }
  try { sessionStorage.setItem("anthropic-key", key); } catch { /* storage unavailable */ }
  const context = lastHits.map((p, i) => `[${i + 1}] ${p.t} (arXiv:${p.id})\n${p.a}`).join("\n\n");
  $("llm-out").innerHTML = `<p class="small">Writing…</p>`;
  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" },
      body: JSON.stringify({ model, max_tokens: 400,
        system: "Answer using ONLY the numbered sources. Cite every claim like [2]. If the sources don't answer the question, say so. At most 150 words.",
        messages: [{ role: "user", content: `Sources:\n\n${context}\n\nQuestion: ${lastQuery}` }] }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d?.error?.message || r.status);
    const text = d.content.filter((b) => b.type === "text").map((b) => b.text).join("");
    $("llm-out").innerHTML = `<p>${esc(text).replace(/\[(\d+)\]/g, (m, n) => `<a class="cite" href="#src-${n}" data-n="${n}">[${n}]</a>`)}</p>`;
  } catch (err) {
    $("llm-out").innerHTML = `<p class="err">Claude request failed: ${esc(err.message)}. Check the key and model name.</p>`;
  }
});

// ---------- benchmark table ----------
function renderBench(b) {
  if (!b) return;
  const names = { bm25: "BM25 (keyword)", dense: "Dense (MiniLM)", hybrid: "Hybrid (RRF)", "hybrid+rerank": "Hybrid + cross-encoder re-rank" };
  const rows = Object.entries(b.results);
  const best = rows.reduce((a, r) => (r[1]["mrr@10"] > a[1]["mrr@10"] ? r : a), rows[0])[0];
  $("btable").querySelector("tbody").innerHTML = rows.map(([k, v]) => `<tr class="${k === best ? "best" : ""}"><td>${names[k] || k}</td>
    <td>${v["hit@1"].toFixed(3)}</td><td>${v["hit@5"].toFixed(3)}</td><td>${v["hit@10"].toFixed(3)}</td>
    <td><span class="bar" style="width:${v["mrr@10"] * 80}px"></span>${v["mrr@10"].toFixed(3)}</td><td>${v.ms_per_query}</td></tr>`).join("");
  $("bnote").textContent = `${b.n_queries} queries over ${b.corpus_size.toLocaleString()} papers. Server-side timings in Python; in-browser search is similar.`;
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => { if (lastQuery) ask(); }));
init();
