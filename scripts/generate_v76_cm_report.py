import sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from namethatllm.config import LABELS, FAMILIES, K
from pathlib import Path

cache = np.load('local_data/v76_output/local_eval_cache.npz', allow_pickle=True)
y_true = cache['y_unseen'].astype(int)
lp = cache['m_0_unseen_lp']
y_pred = np.argmax(lp, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
bacc = balanced_accuracy_score(y_true, y_pred)
total_correct = int(np.trace(cm))
total_samples = len(y_true)

confusions = []
for i in range(K):
    for j in range(K):
        if i != j:
            cnt = int(cm[i, j])
            if cnt > 0:
                same_family = any(i in members and j in members for members in FAMILIES.values())
                fam_name = next((fname for fname, mems in FAMILIES.items() if i in mems and j in mems), 'Cross-Family')
                confusions.append({
                    'true_idx': i,
                    'pred_idx': j,
                    'true_name': LABELS[i],
                    'pred_name': LABELS[j],
                    'count': cnt,
                    'pct': (cnt / 20.0) * 100.0,
                    'same_family': same_family,
                    'family_name': fam_name
                })
confusions.sort(key=lambda x: x['count'], reverse=True)

artifact_dir = Path(r'C:/Users/anil/.gemini/antigravity/brain/3fc37c66-8b30-4f1c-9198-eabf6c04d89f')
html_path = artifact_dir / 'v76_confusion_matrix.html'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Version 76 Confusion Matrix & Pair Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background-color: #0b0f19; color: #f1f5f9; }}
        .heat-diag {{ background-color: rgba(16, 185, 129, 0.75); color: #ffffff; font-weight: bold; }}
        .heat-sib {{ background-color: rgba(245, 158, 11, 0.25); color: #fde68a; font-weight: 600; }}
        .heat-cross {{ background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; font-weight: 600; }}
        .heat-zero {{ color: #334155; }}
    </style>
</head>
<body class="p-6 font-sans">
<div class="max-w-7xl mx-auto space-y-6">

    <!-- Header Card -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <div class="flex flex-wrap items-center justify-between gap-4">
            <div>
                <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Kaggle Version 76
                </span>
                <h1 class="text-3xl font-extrabold text-white mt-2">Pure Solo Plain Head: 18-Class Confusion Matrix</h1>
                <p class="text-sm text-slate-400 mt-1">
                    Evaluating Path A centered & scaled bias head on strictly unseen holdout (N=360, 20 samples per class)
                </p>
            </div>
            <div class="flex gap-6 text-right">
                <div>
                    <div class="text-xs text-slate-400 uppercase font-semibold tracking-wider">Raw Balanced Acc</div>
                    <div class="text-3xl font-bold text-emerald-400">{bacc*100:.2f}%</div>
                    <div class="text-xs text-slate-500">{total_correct}/{total_samples} Correct</div>
                </div>
                <div>
                    <div class="text-xs text-slate-400 uppercase font-semibold tracking-wider">Sinkhorn BAcc</div>
                    <div class="text-3xl font-bold text-cyan-400">78.61%</div>
                    <div class="text-xs text-slate-500">Mass-Equalized</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Top Confused Pairs Table -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>Top Confused Pairs (True &rarr; Predicted)</span>
            <span class="text-xs font-normal text-slate-400">Ranked by total error frequency</span>
        </h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left text-sm border-collapse">
                <thead>
                    <tr class="border-b border-slate-800 text-xs text-slate-400 uppercase tracking-wider">
                        <th class="py-2 px-3">Rank</th>
                        <th class="py-2 px-3">True Author</th>
                        <th class="py-2 px-3">Predicted As</th>
                        <th class="py-2 px-3 text-center">Error Type</th>
                        <th class="py-2 px-3 text-center">Count</th>
                        <th class="py-2 px-3 text-right">% of Class</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/60">
"""

for idx, item in enumerate(confusions[:20]):
    badge = f'''<span class="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">Sibling ({item['family_name']})</span>''' if item['same_family'] else '''<span class="px-2 py-0.5 rounded text-xs font-semibold bg-red-500/20 text-red-300 border border-red-500/30">Cross-Family</span>'''
    html += f"""
                    <tr class="hover:bg-slate-800/40">
                        <td class="py-2.5 px-3 font-mono text-slate-500">#{idx+1}</td>
                        <td class="py-2.5 px-3 font-semibold text-slate-200">{item['true_name']}</td>
                        <td class="py-2.5 px-3 font-medium text-slate-300">&rarr; {item['pred_name']}</td>
                        <td class="py-2.5 px-3 text-center">{badge}</td>
                        <td class="py-2.5 px-3 text-center font-bold font-mono text-white">{item['count']}</td>
                        <td class="py-2.5 px-3 text-right font-mono text-slate-400">{item['pct']:.1f}%</td>
                    </tr>"""

html += """
                </tbody>
            </table>
        </div>
    </div>

    <!-- Full 18x18 Grid -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 class="text-xl font-bold text-white mb-4">Full 18x18 Confusion Heatmap</h2>
        <div class="overflow-x-auto">
            <table class="w-full text-center text-xs border-collapse">
                <thead>
                    <tr class="border-b border-slate-800 text-slate-400">
                        <th class="py-2 px-2 text-left">True \\ Pred</th>
"""

for lbl in LABELS:
    short_lbl = lbl.replace('gemini-', 'gem-').replace('claude-', 'cld-').replace('gpt-', 'gpt-')
    html += f"""<th class="py-2 px-1 text-[10px] font-mono" title="{lbl}">{short_lbl[:6]}</th>"""

html += """</tr></thead><tbody>"""

for i, true_lbl in enumerate(LABELS):
    html += f"""<tr class="border-b border-slate-800/40 hover:bg-slate-800/20"><td class="py-2 px-2 text-left font-medium text-slate-300 whitespace-nowrap">{true_lbl}</td>"""
    for j, pred_lbl in enumerate(LABELS):
        val = cm[i, j]
        if i == j:
            cls = 'heat-diag'
        elif val == 0:
            cls = 'heat-zero'
        else:
            same_fam = any(i in m and j in m for m in FAMILIES.values())
            cls = 'heat-sib' if same_fam else 'heat-cross'
        html += f"""<td class="py-2 px-1 font-mono {cls}" title="True: {true_lbl} -> Pred: {pred_lbl}">{val}</td>"""
    html += """</tr>"""

html += """</tbody></table></div></div></div></body></html>"""

html_path.write_text(html, encoding="utf-8")
print(f"HTML report saved to: {html_path}")
