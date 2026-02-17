import json
import numpy as np
from pathlib import Path
import sys
from pathlib import Path as _Path
# ensure project root is on sys.path so we can import train_model
project_root = str(_Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train_model import PreferenceModel

features_file = Path('data/features.json')
model_file = Path('data/preference_model.pkl')
output_file = Path('data/predictions_all.json')

if not features_file.exists():
    print('features.json not found')
    raise SystemExit(1)
if not model_file.exists():
    print('preference_model.pkl not found')
    raise SystemExit(1)

with open(features_file, 'r', encoding='utf-8') as f:
    features = json.load(f)

# prepare X and filenames
filenames = list(features.keys())
X = np.array([features[n] for n in filenames])

# load model
model = PreferenceModel()
model.load_model(str(model_file))

# predict
scores = model.predict(X)

preds = []
for name, s in zip(filenames, scores):
    preds.append({'filename': name, 'score': float(s), 'path': str(Path('data/faces') / name)})

# sort
preds.sort(key=lambda x: x['score'], reverse=True)

# save
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)

# print summary
print(f"Saved {len(preds)} predictions to {output_file}")
print('\nTop 5:')
for i, p in enumerate(preds[:5], 1):
    print(f"{i}. {p['filename']:30s} {p['score']:.2f}")
print('\nBottom 5:')
for i, p in enumerate(preds[-5:], 1):
    print(f"{i}. {p['filename']:30s} {p['score']:.2f}")
