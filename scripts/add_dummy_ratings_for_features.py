import json
import random
from pathlib import Path

feat_p = Path('data/features.json')
ratings_p = Path('data/ratings.json')
if not feat_p.exists():
    print('features.json not found')
    raise SystemExit(1)

with open(feat_p, 'r', encoding='utf-8') as f:
    features = json.load(f)

if ratings_p.exists():
    with open(ratings_p, 'r', encoding='utf-8') as f:
        ratings = json.load(f)
else:
    ratings = {}

added = 0
for name in features.keys():
    if name not in ratings:
        ratings[name] = random.randint(1, 10)
        added += 1

with open(ratings_p, 'w', encoding='utf-8') as f:
    json.dump(ratings, f, ensure_ascii=False, indent=2)

print(f'Added {added} dummy ratings to {ratings_p}. Total ratings: {len(ratings)}')
