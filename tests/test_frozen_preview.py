"""Frozen family preview: pairing, fallback, save/load roundtrip."""
from __future__ import annotations

from pathlib import Path

from showdown.rl.frozen_preview import (
    FAMILY_PREVIEW_KIND,
    FIXED_EXPORT_ORDER,
    default_checkpoint_for,
    family_preview_for_combat,
    load_frozen_preview,
    resolve_preview_kind,
)


def test_family_pairing():
    assert FAMILY_PREVIEW_KIND['ddqn'] == 'team_preview_ppo'
    assert FAMILY_PREVIEW_KIND['ppo'] == 'team_preview_ppo'
    assert FAMILY_PREVIEW_KIND['mcts_puct'] == 'team_preview_nash'
    assert FAMILY_PREVIEW_KIND['mcts_puct_nash'] == 'team_preview_nash'
    assert FAMILY_PREVIEW_KIND['qd_ppo'] == 'team_preview_qd_pga'
    assert FAMILY_PREVIEW_KIND['qd_pga'] == 'team_preview_qd_pga'
    assert FAMILY_PREVIEW_KIND['qd_ppga'] == 'team_preview_qd_pga'


def test_resolve_and_defaults():
    assert resolve_preview_kind('ddqn') == 'team_preview_ppo'
    assert 'team_preview_nash' in str(default_checkpoint_for('mcts_puct'))


def test_missing_checkpoint_falls_back_to_fixed_export():
    pol = load_frozen_preview(
        kind='team_preview_ppo',
        checkpoint='logs/__no_such_preview__.pt',
        allow_missing=True,
    )
    assert not pol.available
    a, cmd = pol.choose_from_species(
        ['Pikachu'] * 6, ['Charmander'] * 6,
    )
    assert a == 0
    assert cmd == FIXED_EXPORT_ORDER


def test_preview_ppo_save_load_roundtrip(tmp_path: Path):
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent

    agent = PreviewPPOAgent()
    path = tmp_path / 'final.pt'
    agent.save(path)
    pol = family_preview_for_combat('ddqn', checkpoint=str(path))
    assert pol.available
    a, cmd = pol.choose_from_species(
        ['Pikachu', 'Raichu', 'Bulbasaur', 'Ivysaur', 'Venusaur', 'Squirtle'],
        ['Charmander', 'Charmeleon', 'Charizard', 'Wartortle', 'Blastoise', 'Caterpie'],
    )
    assert cmd.startswith('/team')
    assert 0 <= a < 360


def test_preview_kind_override_for_qd_family(tmp_path: Path, monkeypatch):
    """PREVIEW_KIND (26-11-2026): I/J/K 2.0 juegan con el preview team_preview_ppo de PPO v2."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from showdown.rl.frozen_preview import configure_preview
    from showdown.rl.team_preview_ppo.agent import PreviewPPOAgent

    path = tmp_path / 'best.pt'
    PreviewPPOAgent().save(path)
    # setenv antes de delenv: así monkeypatch restaura también lo que configure_preview escribe
    for k in ('PREVIEW_CHECKPOINT', 'PREVIEW_REQUIRED', 'PREVIEW_KIND'):
        monkeypatch.setenv(k, '')
        monkeypatch.delenv(k)
    # sin override, la familia QD lo resolvería como QD-PGA
    assert family_preview_for_combat('qd_ppo', checkpoint=str(path)).kind == 'team_preview_qd_pga'
    monkeypatch.setenv('PREVIEW_KIND', 'team_preview_ppo')
    for fam in ('qd_ppo', 'qd_pga', 'qd_ppga'):
        pol = family_preview_for_combat(fam, checkpoint=str(path))
        assert pol.kind == 'team_preview_ppo' and pol.available
    msg = configure_preview('qd_pga', checkpoint=str(path), required=True)
    assert 'preview=team_preview_ppo' in msg and 'cargado=sí' in msg
