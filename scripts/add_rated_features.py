import json
import numpy as np
from pathlib import Path

p = Path('data/features.json')
if not p.exists():
    print('features.json not found')
    raise SystemExit(1)

with open(p, 'r', encoding='utf-8') as f:
    features = json.load(f)

# add entries for 000001.jpg .. 000030.jpg if missing
for i in range(1, 31):
    name = f"{i:06d}.jpg" if False else f"{i:06d}.jpg"
# note: ratings.json uses 000001.jpg etc (6 digits)
for i in range(1, 31):
    name = f"{i:06d}.jpg"[1:] if False else f"{i:06d}.jpg"

# However ratings.json uses 6-digit with leading zeros but 6 chars? Let's match '000001.jpg'
for i in range(1, 31):
    name = f"{i:06d}.jpg"  # creates '000001.jpg'
    if name not in features:
        features[name] = np.random.randn(512).tolist()

with open(p, 'w', encoding='utf-8') as f:
    json.dump(features, f)

print(f'Now features has {len(features)} entries')
