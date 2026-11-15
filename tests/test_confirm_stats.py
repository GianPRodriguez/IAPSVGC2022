"""Estadística confirmatoria: binomial + cluster bootstrap, McNemar pareado y Holm."""
import sys

sys.path.insert(0, '.')

from showdown.inference_sim import confirm_stats as cs


def _log(tmp_path, name, rows):
    lines = ['BATCH robust teams=4 seed=0 A=x B=y']
    for i, (res, a, b) in enumerate(rows, 1):
        lines.append(f'[{i}/{len(rows)}] {res} g1:t4W c2=+1.0 | W=0 L=0 E=0 S=0 sync=OK inf=OK | {a} vs {b}')
    p = tmp_path / name
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(p)


def test_parse_and_binomial(tmp_path):
    rows = [('WIN', 'T1', 'T2'), ('WIN', 'T2', 'T1'), ('LOSS', 'T3', 'T4'), ('DRAW', 'T4', 'T3')]
    s = cs.parse_series([_log(tmp_path, 'X_9201.log', rows)])
    assert [r[0] for r in s] == ['9201'] * 4
    assert [r[3] for r in s] == [1.0, 1.0, 0.0, 0.5]
    rep = cs.binom_report(s)
    assert rep['n'] == 4 and rep['wins'] == 2.5


def test_binomial_p_values():
    assert abs(cs.binom_two_sided_p(5, 10) - 1.0) < 1e-9
    assert abs(cs.binom_two_sided_p(0, 10) - 2 / 1024) < 1e-9
    # 2730/5100 = 53,5 % → claramente significativo
    assert cs.binom_two_sided_p(2730, 5100) < 1e-5


def test_mcnemar_pairs_by_seed_and_teams(tmp_path):
    a = _log(tmp_path, 'A_9201.log', [('WIN', 'T1', 'T2'), ('WIN', 'T3', 'T4'), ('LOSS', 'T5', 'T6')])
    b = _log(tmp_path, 'B_9201.log', [('WIN', 'T1', 'T2'), ('LOSS', 'T3', 'T4'), ('LOSS', 'T5', 'T6')])
    r = cs.mcnemar(cs.parse_series([a]), cs.parse_series([b]))
    assert r['matched'] == 3 and r['a_better'] == 1 and r['b_better'] == 0


def test_holm():
    out = cs.holm([0.01, 0.04, 0.03])
    assert [round(a, 3) for _, a, _ in out] == [0.03, 0.06, 0.06]
    assert [rej for *_, rej in out] == [True, False, False]


def test_team_names_with_pipe(tmp_path):
    rows = [('WIN', 'T1 at Cup; champions M/C #5 |', '#5th - Other')]
    s = cs.parse_series([_log(tmp_path, 'X_9201.log', rows)])
    assert len(s) == 1 and s[0][3] == 1.0


def test_truncated_second_team(tmp_path):
    p = tmp_path / 'X_9201.log'
    p.write_text('[1/2] LOSS g1:t4L c2=-1 | W=0 L=1 E=0 S=0 sync=OK inf=OK | #2nd - Long Team Name vs\n',
                 encoding='utf-8')
    s = cs.parse_series([str(p)])
    assert len(s) == 1 and s[0][1] == '#2nd - Long Team Name' and s[0][3] == 0.0
