import json
import os

with open('local_data/vis_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Qwen-14B vs DeBERTa-v3-Large Embedding Orthogonality & Gemini Specialist Visualizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    .active-tab {{
      border-bottom: 2px solid #38bdf8;
      color: #38bdf8;
      font-weight: 600;
    }}
  </style>
</head>
<body class="p-4 md:p-6 antialiased">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-700 gap-4">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span class="p-1.5 bg-sky-500/10 text-sky-400 rounded-lg border border-sky-500/20">⚡</span>
          Qwen-14B vs DeBERTa-v3-Large Representation Geometry
        </h1>
        <p class="text-sm text-slate-400 mt-1">
          Empirical comparison across 400 unseen Gemini sister responses (Clean 1,800 Holdout Benchmark)
        </p>
      </div>
      <div class="flex items-center gap-3">
        <div class="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono font-medium">
          Orthogonality: 87.7% (CKA: 0.1233)
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-mono font-medium">
          Grand Blend: 57.50% (+1.50% Lift)
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-700 text-sm gap-6">
      <button onclick="switchTab('tab-pca')" id="btn-tab-pca" class="pb-3 active-tab transition-colors">
        1. 2D Representation Manifold (PCA)
      </button>
      <button onclick="switchTab('tab-cka')" id="btn-tab-cka" class="pb-3 text-slate-400 hover:text-slate-200 transition-colors">
        2. Layer-by-Layer CKA Orthogonality (8×8 Heatmap)
      </button>
      <button onclick="switchTab('tab-ensemble')" id="btn-tab-ensemble" class="pb-3 text-slate-400 hover:text-slate-200 transition-colors">
        3. Specialist Gates & Grand Ensemble (EXP-03c / EXP-04a + EXP-06c)
      </button>
    </div>

    <!-- TAB 1: 2D Representation Space -->
    <div id="tab-pca" class="space-y-4">
      <div class="flex flex-wrap items-center justify-between gap-4 bg-slate-800/60 p-3 rounded-xl border border-slate-700 text-xs">
        <div class="flex items-center gap-2">
          <span class="text-slate-400 font-medium">Model Projection:</span>
          <div class="inline-flex p-1 rounded-lg bg-slate-900 border border-slate-700">
            <button onclick="setModel('qwen')" id="btn-qwen" class="px-3 py-1 rounded-md bg-sky-500 text-white font-medium transition-all">
              Qwen-14B (40,960-D)
            </button>
            <button onclick="setModel('deb')" id="btn-deb" class="px-3 py-1 rounded-md text-slate-400 hover:text-white transition-all">
              DeBERTa-v3-Large (8,192-D)
            </button>
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-4">
          <div class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-blue-500 inline-block"></span>
            <span class="text-slate-300">Pro-3.1-preview</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span>
            <span class="text-slate-300">Flash-3.6</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-amber-500 inline-block"></span>
            <span class="text-slate-300">Flash-3.7</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-purple-500 inline-block"></span>
            <span class="text-slate-300">Flash-3.8</span>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div class="lg:col-span-3 bg-slate-900/80 rounded-xl border border-slate-700 p-4 relative overflow-hidden flex flex-col items-center justify-center min-h-[460px]">
          <svg id="pca-svg" viewBox="-110 -110 220 220" class="w-full max-w-[500px] h-auto overflow-visible">
            <!-- Grid lines -->
            <line x1="-100" y1="0" x2="100" y2="0" stroke="#334155" stroke-dasharray="2,2" stroke-width="0.8" />
            <line x1="0" y1="-100" x2="0" y2="100" stroke="#334155" stroke-dasharray="2,2" stroke-width="0.8" />
            <!-- Centroids & Points rendered by JS -->
            <g id="pca-points"></g>
          </svg>
          <div id="tooltip" class="absolute hidden bg-slate-950/90 text-slate-200 border border-slate-700 text-xs px-3 py-1.5 rounded-lg shadow-xl pointer-events-none z-10 font-mono"></div>
        </div>

        <div class="space-y-4 text-xs">
          <div class="bg-slate-800/40 border border-slate-700 p-4 rounded-xl space-y-2">
            <h4 class="font-semibold text-slate-200 text-sm flex items-center gap-1.5">
              <span class="text-sky-400">✦</span> Geometric Insight
            </h4>
            <div id="model-insight-text" class="text-slate-300 leading-relaxed">
              Loading insight...
            </div>
          </div>

          <div class="bg-slate-800/40 border border-slate-700 p-4 rounded-xl space-y-3">
            <h4 class="font-semibold text-slate-200 text-sm">Cluster Dispersion Metrics</h4>
            <div class="space-y-2 font-mono">
              <div class="flex justify-between pb-1 border-b border-slate-700/60">
                <span class="text-slate-400">Linear CKA:</span>
                <span class="text-emerald-400 font-bold">0.1233 (87.7% Orthogonal)</span>
              </div>
              <div class="flex justify-between pb-1 border-b border-slate-700/60">
                <span class="text-slate-400">Pro-3.1 Separation:</span>
                <span class="text-sky-400 font-bold">High (d' = 1.42)</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">Flash Sister Overlap:</span>
                <span id="overlap-stat" class="text-amber-400 font-bold">Moderate</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: CKA Layer Heatmap -->
    <div id="tab-cka" class="hidden space-y-4">
      <div class="bg-slate-800/40 p-4 rounded-xl border border-slate-700 text-xs text-slate-300 flex flex-col md:flex-row items-center justify-between gap-4">
        <p>
          <strong>Cross-Model Representation Alignment:</strong> Centered Kernel Alignment (CKA) between Qwen-14B's 8 tapped layers (y-axis) and DeBERTa-v3-Large's 8 tapped layers (x-axis). Low values (&lt;0.10) denote extreme orthogonality.
        </p>
        <div class="flex items-center gap-2 text-xs shrink-0">
          <span class="text-slate-400">0.0 (Orthogonal)</span>
          <div class="w-24 h-3 rounded bg-gradient-to-r from-indigo-950 via-purple-600 to-amber-300"></div>
          <span class="text-slate-400">0.80 (Aligned)</span>
        </div>
      </div>

      <div class="bg-slate-900/80 rounded-xl border border-slate-700 p-6 flex flex-col items-center justify-center overflow-x-auto">
        <div class="min-w-[500px]">
          <div class="text-center text-xs font-semibold text-sky-400 mb-2">DeBERTa-v3-Large Tapped Layers &rarr;</div>
          <div class="flex">
            <div class="flex flex-col justify-around text-xs font-mono font-semibold text-sky-400 pr-3 text-right">
              <div>Qwen L02</div><div>Qwen L04</div><div>Qwen L06</div><div>Qwen L09</div>
              <div>Qwen L13</div><div>Qwen L17</div><div>Qwen L22</div><div>Qwen L27</div>
            </div>
            <div id="heatmap-grid" class="grid grid-cols-8 gap-1.5"></div>
          </div>
          <div class="grid grid-cols-8 gap-1.5 pl-20 pt-2 text-center text-xs font-mono text-slate-400">
            <div>L03</div><div>L06</div><div>L09</div><div>L12</div><div>L15</div><div>L18</div><div>L21</div><div>L24</div>
          </div>
        </div>
        <div id="cka-tooltip" class="mt-4 text-xs font-mono text-slate-300 bg-slate-800/80 px-4 py-2 rounded-lg border border-slate-700 min-h-[36px] flex items-center justify-center">
          Hover over any matrix cell to inspect layer alignment
        </div>
      </div>
    </div>

    <!-- TAB 3: Ensemble Blending & Error Rescue -->
    <div id="tab-ensemble" class="hidden space-y-6">
      
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700 space-y-1">
          <div class="text-xs text-slate-400 font-medium">1. Qwen-14B Plain Head</div>
          <div class="text-2xl font-bold font-mono text-white">{data['bacc_qwen']*100:.2f}%</div>
          <div class="text-xs text-slate-400">224 / 400 correct</div>
        </div>
        <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700 space-y-1">
          <div class="text-xs text-slate-400 font-medium">2. Qwen Specialist (EXP-03c)</div>
          <div class="text-2xl font-bold font-mono text-sky-300">{data['bacc_exp03c']*100:.2f}%</div>
          <div class="text-xs text-slate-400">226 / 400 correct</div>
        </div>
        <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700 space-y-1">
          <div class="text-xs text-slate-400 font-medium">3. DeBERTa Specialist Gate</div>
          <div class="text-2xl font-bold font-mono text-white">{data['bacc_deb']*100:.2f}%</div>
          <div class="text-xs text-slate-400">199 / 400 correct</div>
        </div>
        <div class="bg-emerald-950/40 p-4 rounded-xl border border-emerald-500/30 space-y-1">
          <div class="text-xs text-emerald-400 font-medium">4. Grand Blend (Qwen+DeB)</div>
          <div class="text-2xl font-bold font-mono text-emerald-300">{data['bacc_grand']*100:.2f}%</div>
          <div class="text-xs text-emerald-400 font-medium">+1.50% Lift (230 / 400 correct)</div>
        </div>
      </div>

      <!-- Live Slider -->
      <div class="bg-slate-800/40 p-5 rounded-xl border border-slate-700 space-y-4">
        <div class="flex items-center justify-between">
          <label for="blend-slider" class="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <span>⚡ Interactive Grand Ensemble Simulator (Qwen EXP-03c + DeBERTa Gate):</span>
            <span id="slider-weight" class="font-mono text-sky-400 font-bold text-base">w_deb = 0.15</span>
          </label>
          <div class="text-xs font-mono">
            <span class="text-slate-400">Holdout Accuracy: </span>
            <span id="live-bacc" class="text-emerald-400 font-bold text-sm">57.50%</span>
          </div>
        </div>
        <input id="blend-slider" type="range" min="0" max="1" step="0.05" value="0.15" class="w-full accent-sky-400 cursor-pointer" oninput="updateBlend(this.value)" />
        <div class="flex justify-between text-xs text-slate-400 font-mono">
          <span>0.0 (Pure Qwen EXP-03c: 56.50%)</span>
          <span class="text-emerald-400 font-bold">0.15 - 0.25 (Peak Grand Blend: 57.50%)</span>
          <span>1.0 (Pure DeBERTa: 49.75%)</span>
        </div>
      </div>

      <!-- Error Rescue Callout -->
      <div class="p-4 rounded-xl bg-sky-950/30 border border-sky-500/30 flex items-start gap-4 text-xs text-slate-300">
        <div class="p-2 bg-sky-500/20 text-sky-400 rounded-lg text-lg shrink-0">🎯</div>
        <div class="space-y-1">
          <div class="font-semibold text-sky-300 text-sm">Orthogonal Synergy: DeBERTa Rescues Flash 3.8 Blindspots</div>
          <p class="leading-relaxed text-slate-400">
            Qwen\'s generative representations suffer from 41 confusions where <code>gemini-3.8-flash</code> is mispredicted as <code>gemini-3.7-flash</code>. DeBERTa\'s bidirectional attention achieves <strong>45.0% recall on 3.8-flash</strong> (vs Qwen\'s 30.0%). Blending the two specialists lifts 3.8-flash recall to <strong>39.0%</strong> and boosts total correct predictions to <strong>230 / 400</strong> (+6 net flips).
          </p>
        </div>
      </div>

      <!-- Confusion Matrices -->
      <div class="space-y-3">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h4 class="text-sm font-semibold text-slate-200">Confusion Matrix Inspector (400 Gemini Samples)</h4>
          <div class="inline-flex p-1 rounded-lg bg-slate-900 border border-slate-700 text-xs">
            <button onclick="setCM('qwen')" id="btn-cm-qwen" class="px-2.5 py-1 rounded text-slate-400 hover:text-white transition-all">Qwen Plain (56.0%)</button>
            <button onclick="setCM('exp03c')" id="btn-cm-exp03c" class="px-2.5 py-1 rounded text-slate-400 hover:text-white transition-all">Qwen Gate EXP-03c (56.5%)</button>
            <button onclick="setCM('deb')" id="btn-cm-deb" class="px-2.5 py-1 rounded text-slate-400 hover:text-white transition-all">DeBERTa Gate (49.8%)</button>
            <button onclick="setCM('grand')" id="btn-cm-grand" class="px-2.5 py-1 rounded bg-emerald-500 text-white font-medium transition-all">Grand Blend (57.5%)</button>
          </div>
        </div>
        <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-700 overflow-x-auto flex justify-center">
          <div id="cm-display" class="text-xs font-mono"></div>
        </div>
      </div>

    </div>

  </div>

  <script>
    const data = {json.dumps(data)};
    
    let currentModel = 'qwen';
    let currentCM = 'grand';

    const colors = {{
      'gemini-3.1-pro-preview': '#3b82f6',
      'gemini-3.6-flash': '#10b981',
      'gemini-3.7-flash': '#f59e0b',
      'gemini-3.8-flash': '#a855f7'
    }};

    function switchTab(tabId) {{
      ['tab-pca', 'tab-cka', 'tab-ensemble'].forEach(id => {{
        document.getElementById(id).classList.add('hidden');
        document.getElementById('btn-' + id).classList.remove('active-tab');
        document.getElementById('btn-' + id).classList.add('text-slate-400');
      }});
      document.getElementById(tabId).classList.remove('hidden');
      document.getElementById('btn-' + tabId).classList.add('active-tab');
      document.getElementById('btn-' + tabId).classList.remove('text-slate-400');
    }}

    function setModel(m) {{
      currentModel = m;
      if (m === 'qwen') {{
        document.getElementById('btn-qwen').className = 'px-3 py-1 rounded-md bg-sky-500 text-white font-medium transition-all';
        document.getElementById('btn-deb').className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition-all';
        document.getElementById('model-insight-text').innerHTML = '<strong>Qwen-14B Manifold:</strong> High intra-family semantic clustering. Because Qwen is a causal decoder predicting next tokens, sister Flash models cluster tightly together (cosine sim ≈ 0.85-0.90) due to shared prompt topic semantics.';
        document.getElementById('overlap-stat').innerText = 'High (Flash sisters blend)';
      }} else {{
        document.getElementById('btn-deb').className = 'px-3 py-1 rounded-md bg-sky-500 text-white font-medium transition-all';
        document.getElementById('btn-qwen').className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition-all';
        document.getElementById('model-insight-text').innerHTML = '<strong>DeBERTa-v3-Large Manifold:</strong> Bidirectional disentangled attention separates Flash sisters into distinct stylistic subspaces (cosine sim ≈ 0.56). Fine-tuning isolates structural syntax independent of prompt topic.';
        document.getElementById('overlap-stat').innerText = 'Low (Cleanly separated)';
      }}
      renderPCA();
    }}

    function renderPCA() {{
      const pts = currentModel === 'qwen' ? data.qwen_pts : data.deb_pts;
      const svg = document.getElementById('pca-points');
      let html = '';

      pts.forEach((pt, i) => {{
        const model = data.models[i];
        const col = colors[model];
        html += `<circle cx="${{pt[0]}}" cy="${{pt[1]}}" r="3.5" fill="${{col}}" opacity="0.75" class="hover:opacity-100 hover:r-5 cursor-pointer transition-all" onmouseover="showTip(event, '${{model}}', ${{pt[0].toFixed(1)}}, ${{pt[1].toFixed(1)}})" onmouseout="hideTip()" />`;
      }});

      svg.innerHTML = html;
    }}

    function showTip(e, m, x, y) {{
      const tip = document.getElementById('tooltip');
      tip.classList.remove('hidden');
      tip.innerHTML = `<strong>${{m}}</strong><br/>PC1: ${{x}}, PC2: ${{y}}`;
      tip.style.left = (e.pageX + 10) + 'px';
      tip.style.top = (e.pageY + 10) + 'px';
    }}

    function hideTip() {{
      document.getElementById('tooltip').classList.add('hidden');
    }}

    function renderHeatmap() {{
      const grid = document.getElementById('heatmap-grid');
      let html = '';
      const matrix = data.cka_matrix;

      matrix.forEach((row, r) => {{
        row.forEach((val, c) => {{
          const norm = Math.min(1.0, Math.max(0.0, (val - 0.0188) / (0.7083 - 0.0188)));
          const hue = Math.round(270 - norm * 220);
          const lightness = Math.round(15 + norm * 55);
          const bg = `hsl(${{hue}}, 85%, ${{lightness}}%)`;
          const textCol = lightness > 45 ? '#0f172a' : '#f8fafc';

          const qLay = data.qwen_layers[r];
          const dLay = data.deb_layers[c];

          html += `<div style="background: ${{bg}}; color: ${{textCol}}" class="w-11 h-11 flex items-center justify-center rounded font-mono text-[10px] font-bold cursor-pointer transition-transform hover:scale-110 shadow-sm" onmouseover="showCKA('${{qLay}}', '${{dLay}}', ${{val}})" onmouseout="resetCKA()">${{val.toFixed(2)}}</div>`;
        }});
      }});
      grid.innerHTML = html;
    }}

    function showCKA(ql, dl, val) {{
      const orth = ((1 - val) * 100).toFixed(1);
      document.getElementById('cka-tooltip').innerHTML = `Alignment between <strong>${{ql}}</strong> and <strong>${{dl}}</strong>: <span class="text-sky-300 font-bold ml-1.5 mr-1.5">CKA = ${{val}}</span> (<span class="text-emerald-400 font-bold">${{orth}}% Orthogonal</span>)`;
    }}

    function resetCKA() {{
      document.getElementById('cka-tooltip').innerText = 'Hover over any matrix cell to inspect layer alignment';
    }}

    function updateBlend(w) {{
      w = parseFloat(w);
      document.getElementById('slider-weight').innerText = `w_deb = ${{w.toFixed(2)}}`;
      const pt = data.grand_blend_curve ? data.grand_blend_curve.find(p => Math.abs(p.w - w) < 0.01) : null;
      if (pt) {{
        document.getElementById('live-bacc').innerText = `${{pt.bacc.toFixed(2)}}%`;
      }}
    }}

    function setCM(type) {{
      currentCM = type;
      ['qwen', 'exp03c', 'deb', 'grand'].forEach(t => {{
        const btn = document.getElementById('btn-cm-' + t);
        if (t === type) {{
          btn.className = 'px-2.5 py-1 rounded bg-emerald-500 text-white font-medium transition-all';
        }} else {{
          btn.className = 'px-2.5 py-1 rounded text-slate-400 hover:text-white transition-all';
        }}
      }});
      renderCM();
    }}

    function renderCM() {{
      let matrix = data.cm_grand;
      if (currentCM === 'qwen') matrix = data.cm_qwen;
      else if (currentCM === 'exp03c') matrix = data.cm_exp03c;
      else if (currentCM === 'deb') matrix = data.cm_deb;
      
      const labels = ['Pro-3.1', 'Flash-3.6', 'Flash-3.7', 'Flash-3.8'];
      
      let html = '<table class="border-collapse text-center">';
      html += '<thead><tr><th class="p-2 text-slate-400">True \\\\ Pred</th>';
      labels.forEach(l => html += `<th class="p-2 text-sky-400">${{l}}</th>`);
      html += '<th class="p-2 text-slate-300">Recall</th></tr></thead><tbody>';

      matrix.forEach((row, i) => {{
        const total = row.reduce((a, b) => a + b, 0);
        const correct = row[i];
        const recall = ((correct / total) * 100).toFixed(1) + '%';
        html += `<tr><td class="p-2 text-left font-bold text-slate-300">${{labels[i]}}</td>`;
        row.forEach((cnt, j) => {{
          const isDiag = i === j;
          const bg = isDiag ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40' : (cnt > 20 ? 'bg-rose-500/10 text-rose-300' : 'text-slate-400');
          html += `<td class="p-2.5 ${{bg}} rounded m-0.5">${{cnt}}</td>`;
        }});
        html += `<td class="p-2 text-emerald-400 font-bold">${{recall}}</td></tr>`;
      }});
      html += '</tbody></table>';

      document.getElementById('cm-display').innerHTML = html;
    }}

    setModel('qwen');
    renderHeatmap();
    renderCM();
  </script>
</body>
</html>
"""

target_path = r'C:\Users\anil\.gemini\antigravity\brain\7935af2a-2f5f-4962-a2e9-d38151e49c87\embedding_orthogonality_visualizer.html'
with open(target_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('Updated visualizer HTML written successfully! Length:', len(html))
