import numpy as np

data = np.load('kaggle_logs/submission_v57/local_eval_cache.npz', allow_pickle=True)
print("=== V57 Models and Weights ===")
for name, w, tier in zip(data['model_names'], data['model_learned_weights'], data['model_tiers']):
    print(f"{name:<25} Weight: {w:.2f} Tier: {tier}")

print("\n=== Extra models ===")
for i in range(data['num_extra'][0]):
    print(f"{data[f'extra_{i}_name'][0]:<25} Weight: {data[f'extra_{i}_weight'][0]:.2f}")
