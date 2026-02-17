import json
import numpy as np
from pathlib import Path

out = Path('data/features.json')
out.parent.mkdir(parents=True, exist_ok=True)

features = {}
for i in range(1, 31):
    name = f'sample_{i:05d}.jpg'
    vec = np.random.randn(512).tolist()
    features[name] = vec

with open(out, 'w', encoding='utf-8') as f:
    json.dump(features, f)

print(f'Wrote {len(features)} dummy features to {out}')
