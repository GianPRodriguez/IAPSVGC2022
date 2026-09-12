"""Extract team URLs from schema.org ItemList + hunt API cursors."""
from __future__ import annotations

import json
import re
from pathlib import Path

from scrape_metavgc import BASE, OUT, fetch, is_reg_mc_slug

html = fetch(f'{BASE}/teams/with/salamence-mega')
root = Path(__file__).resolve().parents[2] / 'logs'

# Extract ItemList JSON-LD blocks
blocks = re.findall(
    r'<script type="application/ld\+json">(.*?)</script>', html, flags=re.S
)
urls = []
for b in blocks:
    try:
        data = json.loads(b)
    except json.JSONDecodeError:
        continue
    if not isinstance(data, dict) or data.get('@type') != 'ItemList':
        continue
    print('ItemList numberOfItems=', data.get('numberOfItems'), 'elements=', len(data.get('itemListElement') or []))
    for el in data.get('itemListElement') or []:
        item = el.get('item') if isinstance(el, dict) else None
        if not isinstance(item, dict):
            continue
        uid = item.get('@id') or item.get('url')
        if uid and '/teams/' in uid:
            urls.append(uid)

print('from ld+json', len(urls), 'unique', len(set(urls)))

# Also from RSC payload: look for /teams/slug patterns longer
rsc_urls = re.findall(r'https://metavgc\.com/teams/[a-z0-9-]+', html)
print('rsc https urls', len(set(rsc_urls)))
path_urls = re.findall(r'"(/teams/[a-z0-9-]{20,})"', html)
print('quoted paths', len(set(path_urls)))

# cursor / graphql / fetch patterns
for pat in [
    r'https://[^"\']+/api/[^"\']+',
    r'/api/[a-zA-Z0-9_/?=&%-]+',
    r'cursor["\']?\s*[:=]\s*["\']([^"\']+)',
    r'teams\?[^"\']+',
]:
    ms = re.findall(pat, html)
    print('PAT', pat[:40], 'count', len(ms), 'sample', ms[:5])

existing = {t.get('slug') for t in json.loads(OUT.read_text(encoding='utf-8'))}
all_slugs = set()
for u in list(urls) + rsc_urls + [BASE + p for p in path_urls]:
    slug = u.rstrip('/').split('/')[-1]
    if slug in ('finder', 'tournaments', 'with') or slug.startswith('with'):
        continue
    all_slugs.add(slug)

new = [s for s in sorted(all_slugs) if s not in existing and is_reg_mc_slug(s)]
new_any = [s for s in sorted(all_slugs) if s not in existing]
print('all slugs', len(all_slugs), 'new reg-mc', len(new), 'new any', len(new_any))
for s in new[:20]:
    print(' NEW', s[:90])
for s in new_any[:10]:
    if s not in new:
        print(' ANY', s[:90])

(root / '_metavgc_extracted_slugs.json').write_text(
    json.dumps({'new_reg_mc': new, 'new_any': new_any}, indent=2), encoding='utf-8'
)
