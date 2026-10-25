import json
from pathlib import Path

nb_path = Path("kaggle_submission.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source_str = "".join(cell.get("source", []))
    
    # 1. Update Step 2: Fail immediately if model directory is missing or fails to load
    if "def find_offline_model_path" in source_str:
        lines = cell["source"]
        new_lines = []
        for line in lines:
            if "TRANSFORMER_PATH = find_offline_model_path()" in line:
                new_lines.append(line)
                new_lines.append("print(f'Active Transformer Path: {TRANSFORMER_PATH}')\n")
                new_lines.append("if Path('/kaggle/input').exists() and TRANSFORMER_PATH is None:\n")
                new_lines.append("    dirs_found = [str(p) for p in Path('/kaggle/input').glob('*')]\n")
                new_lines.append("    raise FileNotFoundError(\n")
                new_lines.append("        f\"CRITICAL ERROR: DeBERTa model (config.json) not found under /kaggle/input!\\n\"\n")
                new_lines.append("        f\"Available directories under /kaggle/input: {dirs_found}.\\n\"\n")
                new_lines.append("        f\"Please check kernel-metadata.json model_sources.\"\n")
                new_lines.append("    )\n")
            elif "print(f'Active Transformer Path: {TRANSFORMER_PATH}')" in line:
                continue
            elif "if self.model_path is None:" in line:
                new_lines.append(line)
                new_lines.append("            raise FileNotFoundError('CRITICAL ERROR: model_path is None! DeBERTa cannot be loaded.')\n")
            elif "return False" in line and len(new_lines) > 0 and "model_path is None" in new_lines[-2]:
                continue
            elif "print(f'[Transformer] Notice: Could not load offline model" in line:
                new_lines.append("            print(f'[Transformer] FATAL ERROR: Could not load offline model from {self.model_path} ({e}).')\n")
                new_lines.append("            raise e\n")
            elif "return False" in line and len(new_lines) > 0 and "FATAL ERROR: Could not load offline model" in new_lines[-1]:
                continue
            elif "except (torch.cuda.OutOfMemoryError, Exception) as e:" in line:
                new_lines.append("        except Exception as e:\n")
                new_lines.append("            print(f'[Transformer] FATAL ERROR during feature extraction: {e}')\n")
                new_lines.append("            raise e\n")
            elif "Gracefully cleaning VRAM and using SVD semantic fallback" in line or "self.model = None" in line or "return None, None" in line:
                continue
            elif "gc.collect()" in line or "torch.cuda.empty_cache()" in line:
                if "if torch.cuda.is_available():" in new_lines[-1]:
                    new_lines.pop() # remove that check
                    continue
                new_lines.append(line)
            else:
                new_lines.append(line)
        cell["source"] = new_lines

    # 2. Update Step 6: Ensure training fails if features were not produced
    if "Training Root Authorship Classifier on TF-IDF" in source_str:
        lines = cell["source"]
        new_lines = []
        for line in lines:
            if "E_style_train, E_sem_train = transformer_extractor.extract(train_texts)" in line:
                new_lines.append(line)
                new_lines.append("\n")
                new_lines.append("assert E_style_train is not None, 'FATAL ERROR: E_style_train is None! DeBERTa extraction failed.'\n")
            elif "clf_neural = None" in line or "scaler_style = None" in line or "if E_style_train is not None:" in line:
                continue
            elif "else:" in line and "print('Using dense SVD semantic projector" in "".join(lines):
                break # stop before else fallback branch
            else:
                new_lines.append(line)
        new_lines.append("\n")
        new_lines.append("print(f'Full v2 pipeline trained in {time.time() - t_start:.2f}s!')\n")
        cell["source"] = new_lines

    # 3. Update Step 7: Ensure test inference fails if neural model is missing
    if "t_infer_start = time.time()" in source_str:
        lines = cell["source"]
        new_lines = []
        for line in lines:
            if "if clf_neural is not None and scaler_style is not None:" in line:
                new_lines.append("assert clf_neural is not None and scaler_style is not None, 'FATAL ERROR: Neural head not trained!'\n")
            elif "if E_style_test is not None:" in line:
                new_lines.append("assert E_style_test is not None, 'FATAL ERROR: Test transformer features are None!'\n")
            else:
                new_lines.append(line)
        cell["source"] = new_lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Successfully injected strict DeBERTa validation into kaggle_submission.ipynb!")
