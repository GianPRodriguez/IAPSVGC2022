"""ObsSpec 1024D dense vector — docs/rl_env_spec.md v1.1."""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np

from showdown.rl.constants import (
    OBS_DIM,
    SEG_BELIEF,
    SEG_FIELD,
    SEG_MOVE_DMG,
    SEG_SPECIES,
    SEG_STATS_VOL,
)
from showdown.rl.damage_heuristic import estimate_damage_pct, move_entry
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot


def _stable_unit(text: str) -> float:
    h = hashlib.md5(str(text or '').encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'little') / 2**32


def _pad(vec: List[float], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    take = min(n, len(vec))
    if take:
        out[:take] = np.asarray(vec[:take], dtype=np.float32)
    return out


def _species_block(mon: Optional[MonSnapshot], width: int = 16) -> List[float]:
    if mon is None:
        return [0.0] * width
    sp = mon.display_species()
    feats = [
        _stable_unit(sp),
        _stable_unit((mon.types[0] if mon.types else '')),
        _stable_unit((mon.types[1] if len(mon.types) > 1 else '')),
        1.0 if mon.revealed else 0.0,
        1.0 if mon.fainted else 0.0,
        float(mon.hp_pct) / 100.0,
        1.0 if mon.is_active else 0.0,
        1.0 if mon.can_mega else 0.0,
        1.0 if mon.can_tera else 0.0,
        _stable_unit(mon.tera_type),
    ]
    return feats + [0.0] * max(0, width - len(feats))


def _belief_block(mon: Optional[MonSnapshot], width: int = 32) -> List[float]:
    if mon is None:
        return [0.0] * width
    known_map = {'unknown': 0.0, 'suspected': 0.5, 'confirmed': 1.0}
    feats = [
        float(mon.spe_tier),
        float(mon.atk_belief),
        float(mon.spa_belief),
        known_map.get(mon.ability_known, 0.0),
        known_map.get(mon.item_known, 0.0),
        _stable_unit(mon.ability),
        _stable_unit(mon.item),
        float(mon.hp_pct) / 100.0,
        1.0 if mon.revealed else 0.0,
    ]
    return feats + [0.0] * max(0, width - len(feats))


def _stats_vol_block(mon: Optional[MonSnapshot], width: int = 16) -> List[float]:
    if mon is None:
        return [0.0] * width
    boosts = mon.boosts or {}

    def stage(k: str) -> float:
        return max(-1.0, min(1.0, float(boosts.get(k, 0)) / 6.0))

    vol = mon.volatiles or {}
    feats = [
        stage('atk'), stage('def'), stage('spa'), stage('spd'), stage('spe'),
        stage('accuracy'), stage('evasion'),
        1.0 if mon.status else 0.0,
        _stable_unit(mon.status),
        float(vol.get('yawn', 0.0)) / 2.0,
        1.0 if vol.get('taunt') else 0.0,
        1.0 if vol.get('disable') else 0.0,
        1.0 if vol.get('substitute') else 0.0,
        1.0 if vol.get('protect') else 0.0,
        1.0 if mon.just_switched else 0.0,
        float(mon.hp_pct) / 100.0,
    ]
    return feats[:width] + [0.0] * max(0, width - len(feats))


def _one_hot_match(value: str, ids: Sequence[str]) -> List[float]:
    norm = str(value or '').lower().replace('-', '').replace(' ', '')
    out = [0.0] * len(ids)
    if not norm:
        out[0] = 1.0
        return out
    for i, token in enumerate(ids):
        if token and token in norm:
            out[i] = 1.0
            return out
    out[0] = 1.0
    return out


def _field_block(battle: BattleSnapshot) -> np.ndarray:
    f = battle.field
    weather_ids = ('', 'raindance', 'sunnyday', 'sandstorm', 'snowscape', 'hail')
    terrain_ids = ('', 'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain')
    vec = _one_hot_match(f.weather, weather_ids)
    tvec = _one_hot_match(f.terrain, terrain_ids)
    extras = [
        float(f.weather_turns) / 8.0,
        float(f.terrain_turns) / 8.0,
        1.0 if f.trick_room else 0.0,
        float(f.trick_room_turns) / 5.0,
        1.0 if f.our_tailwind else 0.0,
        float(f.our_tailwind_turns) / 4.0,
        1.0 if f.opp_tailwind else 0.0,
        float(f.opp_tailwind_turns) / 4.0,
        1.0 if f.our_reflect else 0.0,
        1.0 if f.our_lightscreen else 0.0,
        1.0 if f.our_auroraveil else 0.0,
        1.0 if f.opp_reflect else 0.0,
        1.0 if f.opp_lightscreen else 0.0,
        1.0 if f.opp_auroraveil else 0.0,
        1.0 if f.gravity else 0.0,
        float(battle.turn) / 30.0,
    ]
    return _pad(vec + tvec + extras, SEG_FIELD)


def _move_matrix_block(battle: BattleSnapshot) -> np.ndarray:
    """384D: 2 active slots × 4 moves × 48 feats (priority, flags, dmg min/max, …)."""
    per_move = 48
    out = np.zeros(SEG_MOVE_DMG, dtype=np.float32)
    weather = battle.field.weather
    foes = [m for m in battle.foe.active if m is not None and not m.fainted]
    idx = 0
    for slot in range(2):
        mon = battle.our.active[slot] if slot < len(battle.our.active) else None
        for mi in range(4):
            feats = [0.0] * per_move
            if mon is not None and mi < len(mon.moves) and mon.moves[mi]:
                mv = mon.moves[mi]
                entry = move_entry(mv)
                prio = float(entry.get('priority') or 0) / 5.0
                bp = float(entry.get('basePower') or 0) / 250.0
                cat = str(entry.get('category') or '')
                feats[0] = prio
                feats[1] = bp
                feats[2] = 1.0 if cat.lower() == 'physical' else 0.0
                feats[3] = 1.0 if cat.lower() == 'special' else 0.0
                feats[4] = 1.0 if cat.lower() == 'status' else 0.0
                feats[5] = _stable_unit(str(entry.get('type') or ''))
                # Grounded / terrain hooks (coarse)
                terrain = str(battle.field.terrain or '').lower()
                feats[6] = 1.0 if 'electric' in terrain else 0.0
                feats[7] = 1.0 if 'grassy' in terrain else 0.0
                feats[8] = 1.0 if 'psychic' in terrain else 0.0
                feats[9] = 1.0 if 'misty' in terrain else 0.0
                # Damage vs up to 2 foes
                for fi, foe in enumerate(foes[:2]):
                    lo, hi = estimate_damage_pct(
                        mv,
                        attacker_types=mon.types,
                        defender_types=foe.types,
                        weather=weather,
                    )
                    feats[10 + fi * 2] = lo / 100.0
                    feats[11 + fi * 2] = hi / 100.0
                if mi < len(mon.move_pps):
                    feats[14] = float(mon.move_pps[mi]) / 40.0
            out[idx:idx + per_move] = np.asarray(feats, dtype=np.float32)
            idx += per_move
    return out


def build_observation(battle: BattleSnapshot) -> np.ndarray:
    party = battle.party_list()
    species = []
    belief = []
    stats = []
    for mon in party:
        species.extend(_species_block(mon, width=SEG_SPECIES // 8))
        belief.extend(_belief_block(mon, width=SEG_BELIEF // 8))
        stats.extend(_stats_vol_block(mon, width=SEG_STATS_VOL // 8))

    obs = np.concatenate([
        _pad(species, SEG_SPECIES),
        _pad(belief, SEG_BELIEF),
        _field_block(battle),
        _pad(stats, SEG_STATS_VOL),
        _move_matrix_block(battle),
    ]).astype(np.float32)
    if obs.shape[0] != OBS_DIM:
        # Hard guarantee
        return _pad(obs.tolist(), OBS_DIM)
    return obs


def build_observation_ddqn(battle: BattleSnapshot) -> np.ndarray:
    """ObsSpec 1024D with belief segment zeroed (DDQN / Masked PPO flat-belief)."""
    obs = build_observation(battle).copy()
    lo = SEG_SPECIES
    hi = SEG_SPECIES + SEG_BELIEF
    obs[lo:hi] = 0.0
    return obs


def build_observation_ppo(battle: BattleSnapshot) -> np.ndarray:
    """Alias: PPO v1.0 uses the same flat-belief obs as DDQN."""
    return build_observation_ddqn(battle)
