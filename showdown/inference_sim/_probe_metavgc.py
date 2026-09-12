"""Probe MetaVGC listing pages for team URLs / pagination / API."""
from __future__ import annotations

import re
from pathlib import Path

from scrape_metavgc import BASE, fetch, is_reg_mc_slug, team_links_in

OUT = Path(__file__).resolve().parents[2] / 'logs' / '_metavgc_probe.txt'


def probe(url: str) -> str:
    lines = [f'URL {url}']
    try:
        html = fetch(url)
    except Exception as exc:
        lines.append(f'  FAIL {exc!r}')
        return '\n'.join(lines)
    links = team_links_in(html)
    reg = [l for l in links if is_reg_mc_slug(l.rstrip('/').split('/')[-1])]
    all_href = re.findall(r'href="([^"]+)"', html)
    team_href = [h for h in all_href if '/teams/' in h]
    pages = [h for h in all_href if ('page=' in h or 'offset' in h or 'cursor' in h)]
    lines.append(f'  html_len={len(html)} team_links_in={len(links)} reg_mc={len(reg)}')
    lines.append(f'  sample={team_href[:12]}')
    lines.append(f'  pageish={pages[:12]}')
    lines.append(f'  NEXT_DATA={"__NEXT_DATA__" in html}')
    # Look for JSON endpoints referenced
    apis = re.findall(r'["\'](/api/[^"\']+)["\']', html)[:20]
    lines.append(f'  api_paths={apis}')
    # Save one sample html snippet of featured
    if 'featured' in url:
        (Path(__file__).resolve().parents[2] / 'logs' / '_metavgc_featured.html').write_text(
            html, encoding='utf-8'
        )
    return '\n'.join(lines)


def main() -> None:
    urls = [
        f'{BASE}/teams/featured/regulation-mc',
        f'{BASE}/teams/finder?format=pokemonchampions-regc',
        f'{BASE}/teams/finder?format=pokemonchampions-regc&page=2',
        f'{BASE}/teams',
        f'{BASE}/pokemon/salamence-mega',
        f'{BASE}/pokemon/rillaboom',
    ]
    text = '\n\n'.join(probe(u) for u in urls)
    OUT.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
