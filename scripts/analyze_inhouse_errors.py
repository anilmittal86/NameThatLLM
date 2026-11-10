import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.namethatllm.config import LABELS, FAMILIES

cache = np.load('local_data/local_eval_cache.npz', allow_pickle=True)
y_unseen = cache['y_unseen']
inhouse_lp = cache['m_4_unseen_lp']  # qwen3_14b_inhouse
pred = np.argmax(inhouse_lp, axis=1)

print("=" * 80)
print("INHOUSE MODEL (qwen3_14b_inhouse) ERROR RATES ON UNSEEN HOLDOUT (N=360)")
print("=" * 80)

results = []
for i, name in enumerate(LABELS):
    mask = (y_unseen == i)
    total = int(np.sum(mask))
    correct = int(np.sum(pred[mask] == i))
    err = total - correct
    err_rate = (err / total * 100.0) if total > 0 else 0.0
    results.append({
        'idx': i,
        'model': name,
        'total': total,
        'correct': correct,
        'errors': err,
        'err_rate': err_rate
    })

df_res = pd.DataFrame(results).sort_values(by=['err_rate', 'errors'], ascending=[False, False])
print(f"{'Idx':>3} | {'Class Name':<23} | {'Total':>5} | {'Correct':>7} | {'Errors':>6} | {'Error Rate':>10}")
print("-" * 65)
for _, r in df_res.iterrows():
    print(f"{int(r['idx']):3d} | {r['model']:<23} | {int(r['total']):5d} | {int(r['correct']):7d} | {int(r['errors']):6d} | {r['err_rate']:9.1f}%")

print("\n" + "=" * 80)
print("TOP CONFUSION SIBLING & CROSS-FAMILY PAIRS (True -> Predicted)")
print("=" * 80)

confusions = []
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        if i != j:
            cnt = int(np.sum((y_unseen == i) & (pred == j)))
            if cnt > 0:
                confusions.append((LABELS[i], LABELS[j], cnt, i, j))

confusions.sort(key=lambda x: x[2], reverse=True)
for true_lbl, pred_lbl, cnt, ti, pi in confusions[:16]:
    same_family = any(ti in members and pi in members for members in FAMILIES.values())
    fam_type = "[SIBLING ERROR]" if same_family else "[CROSS-FAMILY ERROR]"
    print(f"{true_lbl:<22} --> {pred_lbl:<22} : {cnt:2d} times  {fam_type}")
