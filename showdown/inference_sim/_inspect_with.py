"""Inspect embedded data in a /teams/with page."""
from __future__ import annotations

import re
from pathlib import Path

from scrape_metavgc import BASE, fetch

html = fetch(f'{BASE}/teams/with/salamence-mega')
root = Path(__file__).resolve().parents[2] / 'logs'
root.mkdir(parents=True, exist_ok=True)
(root / '_metavgc_with_salamence.html').write_text(html, encoding='utf-8')

# script tags / json blobs
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, flags=re.S | re.I)
print('scripts', len(scripts), 'html', len(html))
for i, s in enumerate(scripts):
    if 'teams/' in s or 'slug' in s or 'regulation' in s.lower():
        print(f'--- script {i} len={len(s)} ---')
        print(s[:500].replace('\n', ' '))

# all /teams/ hrefs unique
hrefs = sorted(set(re.findall(r'/teams/[a-z0-9-]+', html)))
print('unique /teams/ paths', len(hrefs))
for h in hrefs:
    print(' ', h[:90])

# look for "Load more" or total count
for pat in [r'(\d+)\s*teams?', r'Load more', r'showing', r'total', r'hasNext', r'cursor']:
    ms = re.findall(pat, html, flags=re.I)
    if ms:
        print('pat', pat, '->', ms[:8])
