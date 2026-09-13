"""Genera logs/metavgc_pkmn_usage.html (estático, sin JS) desde el JSON MetaVGC."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / 'showdown' / 'inference_sim' / 'metavgc_regmc_teams.json'
OUT_HTML = ROOT / 'logs' / 'metavgc_pkmn_usage.html'
OUT_DATA = ROOT / 'logs' / '_usage_canvas_data.json'


def base_species(name: str) -> str:
    n = (name or '').strip()
    for suf in ('-Mega-X', '-Mega-Y', '-Mega'):
        if n.endswith(suf):
            return n[: -len(suf)]
    # MetaVGC a veces pone "Mega Golisopod Mega"
    if n.lower().startswith('mega '):
        n = n[5:].strip()
        if n.lower().endswith(' mega'):
            n = n[: -5].strip()
    return n


def tier_of(pct: float) -> str:
    if pct >= 40:
        return 'S'
    if pct >= 20:
        return 'A'
    if pct >= 10:
        return 'B'
    if pct >= 5:
        return 'C'
    return 'D'


def main() -> None:
    raw = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    teams = raw if isinstance(raw, list) else []
    n = len(teams)

    counts: Counter[str] = Counter()
    co: dict[str, Counter[str]] = defaultdict(Counter)
    for team in teams:
        names: list[str] = []
        seen = set()
        for mon in team.get('mons') or []:
            sp = base_species(mon.get('species') or '')
            if not sp or sp in seen:
                continue
            seen.add(sp)
            names.append(sp)
        for sp in names:
            counts[sp] += 1
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                co[a][b] += 1
                co[b][a] += 1

    rows = []
    for rank, (sp, count) in enumerate(counts.most_common(), start=1):
        pct = round(100.0 * count / n, 1) if n else 0.0
        rows.append({'rank': rank, 'species': sp, 'count': count, 'pct': pct})

    top20 = rows[:20]
    top_chart = rows[:18]
    partners = {}
    for row in rows[:6]:
        sp = row['species']
        partners[sp] = [[b, c] for b, c in co[sp].most_common(4)]

    tiers = {'S': [], 'A': [], 'B': [], 'C': [], 'D': []}
    for row in rows:
        tiers[tier_of(row['pct'])].append(row)

    payload = {
        'n': n,
        'unique': len(rows),
        'slots': n * 6,
        'rows': rows,
        'top20': top20,
        'partners': partners,
        'tiers': {k: v for k, v in tiers.items() if k != 'D'},
    }
    OUT_DATA.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    max_pct = top_chart[0]['pct'] if top_chart else 1.0
    scale = max(max_pct, 70.0)  # barras relativas

    def bar_rows_html() -> str:
        parts = []
        for row in top_chart:
            width = 100.0 * row['pct'] / scale
            cls = 'bar-fill s' if row['pct'] >= 40 else 'bar-fill'
            parts.append(
                f'      <div class="bar-row">\n'
                f'        <div>{escape(row["species"])}</div>\n'
                f'        <div class="bar-track"><div class="{cls}" style="width:{width:.2f}%"></div></div>\n'
                f'        <div class="num">{row["pct"]:.1f}%</div>\n'
                f'      </div>'
            )
        return '\n'.join(parts)

    def tier_s_html() -> str:
        lines = []
        for row in tiers['S']:
            lines.append(
                f'        <div style="display:flex;justify-content:space-between;margin:4px 0">'
                f'<strong>{escape(row["species"])}</strong>'
                f'<span style="color:var(--muted)">{row["pct"]:.1f}%</span></div>'
            )
        return '\n'.join(lines) or '        <div style="color:var(--muted)">—</div>'

    def tier_pills(letter: str) -> str:
        pills = []
        for row in tiers[letter]:
            pills.append(
                f'<span class="pill {letter.lower()}">{escape(row["species"])} {row["pct"]:.1f}%</span>'
            )
        return ''.join(pills) or '<span style="color:var(--muted)">—</span>'

    def partners_html() -> str:
        blocks = []
        for sp, pairs in partners.items():
            rows_html = ''.join(
                f'<tr><td>{escape(b)}</td><td class="num">{c}</td></tr>' for b, c in pairs
            )
            blocks.append(
                f'      <div class="panel" style="margin:0">\n'
                f'        <h3>{escape(sp)}</h3>\n'
                f'        <table>\n'
                f'          <thead><tr><th>Compañero</th><th class="num">Equipos juntos</th></tr></thead>\n'
                f'          <tbody>\n'
                f'            {rows_html}\n'
                f'          </tbody>\n'
                f'        </table>\n'
                f'      </div>'
            )
        return '\n'.join(blocks)

    def table_rows_html() -> str:
        parts = []
        for row in rows:
            t = tier_of(row['pct'])
            tr_cls = ' class="tier-s"' if t == 'S' else ''
            parts.append(
                f'          <tr{tr_cls}><td class="num">{row["rank"]}</td>'
                f'<td>{escape(row["species"])}</td>'
                f'<td class="num">{row["count"]}</td>'
                f'<td class="num">{row["pct"]:.1f}%</td>'
                f'<td class="ctr"><span class="badge {t}">{t}</span></td></tr>'
            )
        return '\n'.join(parts)

    top_name = top_chart[0]['species'] if top_chart else '—'
    top_pct = f'{top_chart[0]["pct"]:.1f}%' if top_chart else '—'

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Meta Pokémon usage — Reg M-C ({n} equipos)</title>
  <style>
    :root {{
      --bg: #0f1115;
      --surface: #171a21;
      --surface2: #1e2330;
      --text: #e8eaed;
      --muted: #9aa3b2;
      --border: #2a3140;
      --accent: #6ea8fe;
      --s: #3dd68c;
      --a: #6ea8fe;
      --b: #e8b339;
      --c: #9aa3b2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{ font-size: 1.6rem; font-weight: 650; margin: 0 0 6px; }}
    h2 {{ font-size: 1.15rem; font-weight: 600; margin: 28px 0 12px; }}
    h3 {{ font-size: 0.95rem; font-weight: 600; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 22px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 22px;
    }}
    .stat {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .stat .v {{ font-size: 1.35rem; font-weight: 650; color: var(--accent); }}
    .stat .l {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .panel .cap {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 140px 1fr 48px;
      gap: 10px;
      align-items: center;
      margin: 5px 0;
      font-size: 0.85rem;
    }}
    .bar-track {{
      height: 18px;
      background: var(--surface2);
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent);
      border-radius: 4px;
      min-width: 2px;
    }}
    .bar-fill.s {{ background: var(--s); }}
    .tiers {{
      display: grid;
      grid-template-columns: 1fr 1.4fr 1.2fr;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .tier-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }}
    .tier-card .head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .pill {{
      display: inline-block;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      margin: 3px 4px 3px 0;
      background: var(--surface2);
    }}
    .pill.s {{ color: var(--s); border-color: #2a5a40; }}
    .pill.a {{ color: var(--a); border-color: #2a4060; }}
    .pill.b {{ color: var(--b); border-color: #5a4a20; }}
    .partners {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      color: var(--muted);
      font-weight: 560;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      position: sticky;
      top: 0;
      background: var(--surface);
    }}
    td.num, th.num {{ text-align: right; }}
    td.ctr, th.ctr {{ text-align: center; }}
    tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
    tr.tier-s td {{ background: rgba(61, 214, 140, 0.06); }}
    .table-wrap {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
    }}
    .badge {{
      font-size: 0.72rem;
      font-weight: 650;
      padding: 1px 7px;
      border-radius: 4px;
      background: var(--surface2);
    }}
    .badge.S {{ color: var(--s); }}
    .badge.A {{ color: var(--a); }}
    .badge.B {{ color: var(--b); }}
    .badge.C, .badge.D {{ color: var(--c); }}
    @media (max-width: 800px) {{
      .stats, .tiers, .partners {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 110px 1fr 40px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Meta Pokémon usage — Reg M-C</h1>
    <p class="sub">
      Uso en {n} equipos de <code>metavgc_regmc_teams.json</code>.
      Cada especie cuenta 1 vez por equipo (máx. 100%).
      Formas Mega agrupadas por especie base.
    </p>

    <div class="stats">
      <div class="stat"><div class="v">{n}</div><div class="l">Equipos</div></div>
      <div class="stat"><div class="v">{len(rows)}</div><div class="l">Especies distintas</div></div>
      <div class="stat"><div class="v">{escape(top_name)}</div><div class="l">Más usado</div></div>
      <div class="stat"><div class="v">{top_pct}</div><div class="l">Uso máximo</div></div>
    </div>

    <h2>Top 18 — % de equipos</h2>
    <div class="panel" id="chart">
{bar_rows_html()}
      <div class="cap">Fuente: MetaVGC Reg M-C · {n} equipos · línea S ≈ ≥40%</div>
    </div>

    <h2>Tiers de uso</h2>
    <div class="tiers" id="tiers">
      <div class="tier-card">
        <div class="head"><h3>Tier S</h3><span class="pill s">≥40%</span></div>
{tier_s_html()}
      </div>
      <div class="tier-card">
        <div class="head"><h3>Tier A</h3><span class="pill a">20–39%</span></div>
        <div>{tier_pills('A')}</div>
      </div>
      <div class="tier-card">
        <div class="head"><h3>Tier B</h3><span class="pill b">10–19%</span></div>
        <div>{tier_pills('B')}</div>
      </div>
    </div>

    <h2>Compañeros más frecuentes</h2>
    <p class="sub" style="margin-top:-4px">Co-aparición en el mismo equipo de 6.</p>
    <div class="partners" id="partners">
{partners_html()}
    </div>

    <h2>Tabla completa ({len(rows)} especies)</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="num">#</th>
            <th>Pokémon</th>
            <th class="num">Apariciones</th>
            <th class="num">% equipos</th>
            <th class="ctr">Tier</th>
          </tr>
        </thead>
        <tbody id="tbody">
{table_rows_html()}
        </tbody>
      </table>
    </div>
    <p class="sub">Tier S ≥40% · A 20–39% · B 10–19% · C 5–9% · D &lt;5%. Fuente: MetaVGC Reg M-C.</p>
  </main>
</body>
</html>
'''
    OUT_HTML.write_text(html, encoding='utf-8')
    print(f'Wrote {OUT_HTML} ({n} teams, {len(rows)} species)')


if __name__ == '__main__':
    main()
