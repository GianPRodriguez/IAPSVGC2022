import sys
import math
import queue
import requests
import constants

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from API.PokemonData import getPkmnInfo

import json
from typing import Union, Dict, Any

from teams.team_converter import normalize_name
from showdown.stat_inference import (
    SP_BUDGET,
    SP_MAX,
    TeamStatBeliefs,
    calibrate_from_known_team,
    dex_base_stats,
)


def _fmt_mapping(mapping) -> str:
    """Compacta dicts/defaultdicts vacíos como '-' para reportes legibles."""
    if not mapping:
        return '-'
    data = dict(mapping)
    if not data:
        return '-'
    return ', '.join(f'{k}:{v}' for k, v in data.items())


def _fmt_stat_line(atk, df, spa, spd, spe) -> str:
    return f"Atk {atk} | Def {df} | SpA {spa} | SpD {spd} | Spe {spe}"


def _fmt_boosts_short(boosts) -> str:
    if not boosts:
        return '-'
    data = {k: v for k, v in dict(boosts).items() if v}
    if not data:
        return '-'
    # atk+1 spa-1 (legible en una línea)
    parts = []
    for k, v in data.items():
        short = {
            'attack': 'atk', 'defense': 'def', 'special-attack': 'spa',
            'special-defense': 'spd', 'speed': 'spe', 'accuracy': 'acc', 'evasion': 'eva',
        }.get(str(k), str(k)[:3])
        parts.append(f"{short}{v:+d}")
    return ' '.join(parts)


# Volátiles / flags que solo viven dentro del turno (o no aportan al snap de decisión).
_REPORT_HIDDEN_VOLATILES = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield',
    'obstruct', 'silktrap', 'burningbulwark', 'maxguard',
    'helpinghand', 'roost', 'focuspunch', 'beakblast', 'shelltrap',
    'charge', 'destinybond', 'grudge', 'yawn', 'moveyawn',
    'mustrecharge',
})


def _fmt_status_short(pkmn) -> str:
    bits = []
    nvs = dict(getattr(pkmn, 'non_volatile_statuses', {}) or {})
    for k, v in nvs.items():
        if v:
            bits.append(str(k))
    if getattr(pkmn, 'is_drowsy', False):
        bits.append('drowsy')
    pc = int(getattr(pkmn, 'perish_count', 0) or 0)
    if pc:
        bits.append(f'perish{pc}')
    # Destiny Bond sí puede cruzar |turn| hasta el próximo move: útil al agente.
    if getattr(pkmn, 'is_destiny_bonded', False):
        bits.append('destinybond')
    if getattr(pkmn, 'is_charged', False):
        bits.append('charge')
    if getattr(pkmn, 'must_recharge', False):
        bits.append('mustrecharge')
    vs = dict(getattr(pkmn, 'volatile_statuses', {}) or {})
    for k, v in vs.items():
        if not v:
            continue
        key = str(k)
        if key.startswith('perish'):
            continue
        if key in _REPORT_HIDDEN_VOLATILES:
            continue
        bits.append(key)
    return ','.join(bits) if bits else '-'


def _fmt_hp(pkmn) -> str:
    try:
        cur = int(float(str(pkmn.HPA_battle_actual).split()[0]))
        mx = int(float(str(pkmn.HPB_battle_base).split()[0]))
        if mx > 0:
            return f"{cur}/{mx} ({100 * cur / mx:.0f}%)"
        return f"{cur}/{mx}"
    except Exception:
        return f"{pkmn.HPA_battle_actual}/{pkmn.HPB_battle_base}"


def _stat_num(pkmn, actual_attr: str, base_attr: str):
    """Prefer actual (>0); else base; else None."""
    try:
        actual = int(getattr(pkmn, actual_attr, -1) or -1)
    except (TypeError, ValueError):
        actual = -1
    if actual > 0:
        return actual
    try:
        base = int(getattr(pkmn, base_attr, -1) or -1)
    except (TypeError, ValueError):
        base = -1
    return base if base > 0 else None


def _fmt_battle_stats_short(pkmn) -> str:
    """
    Stats de combate compactas (auditar Guard Split / etc.).
    Muestra actual si existe; si no, base de especie (rival).
    """
    parts = []
    for label, actual_a, base_a in (
        ('Atk', 'ATK_battle_actual', 'ATK_battle_base'),
        ('Def', 'DEF_battle_actual', 'DEF_battle_base'),
        ('SpA', 'SPA_battle_actual', 'SPA_battle_base'),
        ('SpD', 'SPD_battle_actual', 'SPD_battle_base'),
        ('Spe', 'VEL_battle_actual', 'VEL_battle_base'),
    ):
        val = _stat_num(pkmn, actual_a, base_a)
        parts.append(f'{label}={val if val is not None else "-"}')
    split = ' SPLIT' if getattr(pkmn, 'guard_split_active', False) else ''
    return ' '.join(parts) + split


def _fmt_flags_short(pkmn, is_opponent: bool) -> str:
    flags = []
    if getattr(pkmn, 'is_fainted', False):
        flags.append('FNT')
    if getattr(pkmn, 'is_ditto', False) and str(getattr(pkmn, 'name', '')).lower() == 'ditto':
        flags.append('DITTO')
    if getattr(pkmn, 'is_zoroark', False) and not getattr(pkmn, 'illusion_disguise', None):
        flags.append('ZORO')
    if getattr(pkmn, 'is_mega_evo', False):
        flags.append('MEGA')
    if getattr(pkmn, 'can_mega_evo', False):
        flags.append('canMega')
    if is_opponent and getattr(pkmn, 'is_opponent_confirmed', False):
        flags.append('OK')
    return ' '.join(flags) if flags else ''


def _fmt_pkmn_display_name(pkmn) -> str:
    """
    Nombres duales para depuración vs UI Showdown:
      - Ditto transformado: ditto -> dragonite
      - Illusion deducida (PS aún muestra disfraz): indeedee-f (ZORO/Illusion)
      - Illusion tras |replace| de PS: zoroark (Illusion was indeedee-f)
    """
    name = getattr(pkmn, 'name', '?') or '?'
    if getattr(pkmn, 'is_ditto', False) and str(name).lower() != 'ditto':
        return f'ditto -> {name}'
    disguise = getattr(pkmn, 'illusion_disguise', None)
    if disguise and getattr(pkmn, 'is_zoroark', False):
        d = str(disguise).lower()
        if d and d != 'zoroark' and not d.startswith('zoroark'):
            if getattr(pkmn, 'illusion_protocol_revealed', False):
                return f'{name} (Illusion was {disguise})'
            return f'{disguise} (ZORO/Illusion)'
    return str(name)


def _snap_kind(battle) -> str:
    if getattr(battle, 'force_switch', False):
        return 'FORCE'
    if getattr(battle, 'wait', False):
        return 'WAIT'
    if getattr(battle, 'error', False):
        return 'ERR'
    return 'DECISION'


def _append_report_file(text: str) -> None:
    import os
    os.makedirs('logs', exist_ok=True)
    with open('logs/reporte.txt', 'a', encoding='utf-8') as f:
        f.write(text if text.endswith('\n') else text + '\n')


def _write_stdout(text: str) -> None:
    import sys
    try:
        sys.stdout.write(text)
        if not text.endswith('\n'):
            sys.stdout.write('\n')
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
        sys.stdout.buffer.write(b'\n')
        sys.stdout.buffer.flush()

############################################################################################################################################################
# OK - CLASE BATTLE                                                                                                                                      ###
############################################################################################################################################################
class Battle(ABC):
    def __init__(self):

        ''' Usuario y Oponente '''
        self.user = Battler()
        self.opponent = Battler()

        ''' Tag Globales '''
        self.battleBO3_tag = None
        self.battleBO3_count = 1
        self.botMode = None
        self.battle_tag = None
        self.q = queue.Queue()

        ''' Características del Entorno '''
        self.weather = None                     # ['None','raindance','sunnyday','sandstorm','hail']
        self.weatherTurn = 0
        self.terrain = defaultdict(lambda: 0)   # ['electricterrain','grassyterrain','mistyterrain','psychicterrain', 'wonderroom', 'trickroom']

        ''' Estados e Info del Turno '''
        self.turn = 0
        self.active = False
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.rqid = None
        # OTS en reporte solo una vez por partida individual
        self._report_ots_dumped = False
        self._report_game_started = False

        ''' Estimacion de Stat Points del rival (se acumula durante todo el BO3) '''
        self.stat_beliefs = TeamStatBeliefs()
        # Orden de accion del turno en curso, para deducir velocidad
        self.turn_actions = []
        self.turn_speed_snapshot = None
        # Golpe pendiente: |move| -> |-damage| para inferir Atk/Def/SpA/SpD
        self.pending_damage_ctx = None
        # Objetivo del próximo |-damage| marcado por |-crit| (None = sin crit pendiente).
        self.pending_crit_target = None
        # Focus Sash / Sturdy: el próximo daño está truncado.
        self.pending_damage_sash = False

        ''' Recompensas '''
        self.reward = 0
        self.score = 0
        # Spec v1.1 (Candidate 2 dense) — online reporte.txt
        self.spec_c2_return = 0.0
        self.spec_c1_return = 0.0
        self.last_c2_reward = 0.0
        self.last_c1_reward = 0.0
        self._spec_prev_our = None
        self._spec_prev_foe = None


############################################################################################################################################################
# OK - Completa las clases Battler con los datos de los 6 PKMN usuario y 6 oponente en el TEAM PREVIEW, se ejecuta 1 vez por Batalla Individual          ###
############################################################################################################################################################
    async def initialize_team_preview(self, user_json: str, opponent_pokemon: list, natures: list) -> None:
        # Lado Usuario
        first_turn = True
        self.user.updatePkmnUser(user_json, first_turn, natures) # Inicializa solo los 6 PKMN del Usuario
        self.rqid = user_json[constants.RQID] # The rqid is obtained

        # Lado Oponente
        for pkmn_string in opponent_pokemon:
            # pkmn_string = ['Swampert', 'Swampertite', 'Torrent', ['WaveCrash', 'Surf', 'BulkUp', 'FlipTurn'], 'Brave']
            name = pkmn_string[0]
            objectPkmn = normalize_name(pkmn_string[1] if len(pkmn_string) > 1 else None)
            abilityPkmn = normalize_name(pkmn_string[2] if len(pkmn_string) > 2 else None)
            nature = normalize_name(pkmn_string[4] if len(pkmn_string) > 4 else None)

            moves = pkmn_string[3] if len(pkmn_string) > 3 and isinstance(pkmn_string[3], list) else []

            # Extraer hasta 4 movimientos
            move1 = moves[0] if len(moves) > 0 else None
            move2 = moves[1] if len(moves) > 1 else None
            move3 = moves[2] if len(moves) > 2 else None
            move4 = moves[3] if len(moves) > 3 else None

            # Normalizar nombre
            name_lower = 'mimikyu-disguised' if name.lower() == 'mimikyu' else name.lower()
            namePkmn = constants.standardName.get(name_lower, name_lower)

            self.opponent.opponents.append(
                Pokemon(namePkmn, -1, True, objectPkmn, abilityPkmn, move1, move2, move3, move4, nature) # Dentro del Constructor se Autocompleta los datos restantes del PKMN Oponente
            )

        # Champions-only: fija topes 32/66 (no ensancha aunque el equipo propio
        # traiga stats ilegales / EV clásicos).
        calibrate_from_known_team(self.user.reserved or [])


############################################################################################################################################################
# OK - Método Abstracto BOT TEAM PREVIEW                                                                                                                 ###
############################################################################################################################################################
    @abstractmethod
    async def action_team_preview(self, battle):
        ...


############################################################################################################################################################
# OK - Método Abstracto BOT TURNO                                                                                                                        ###
############################################################################################################################################################
    @abstractmethod
    async def action_battle(self, battle):
        ...        


############################################################################################################################################################
# OK - Limpiar Nueva Batalla                                                                                                                             ###       
############################################################################################################################################################
    def toClean(self) -> None:
        """
        Limpia y reinicia el estado de la batalla para una nueva ronda (por ejemplo, en un BO3),
        reseteando al usuario, oponente, entorno, turnos y recompensas.
        Mantiene intactos los Tag Globales y los nombres de cuenta (BO3 / |player|).
        """
        # Conservar identidades de cuenta entre partidas del BO3
        user_account = getattr(self.user, 'account_name', None)
        opp_account = getattr(self.opponent, 'account_name', None)

        # =====================================================================
        # 1. Reiniciar Usuario y Oponente
        # =====================================================================
        self.user = Battler()
        self.opponent = Battler()
        self.user.account_name = user_account
        self.opponent.account_name = opp_account

        # =====================================================================
        # 2. Reiniciar Características del Entorno
        # =====================================================================
        self.weather = None
        self.weatherTurn = 0
        self.terrain = defaultdict(lambda: 0)

        # =====================================================================
        # 3. Reiniciar Estados e Info del Turno
        # =====================================================================
        self.turn = 0
        self.force_switch = False
        self.wait = False
        self.error = False
        self.request_json = None
        self.rqid = None
        self._report_ots_dumped = False
        self._report_game_started = False

        # Lo aprendido de los Stat Points del rival SI se conserva entre juegos
        # del BO3: es el mismo equipo, igual que las notas de un jugador humano.
        self.turn_actions = []
        self.turn_speed_snapshot = None
        self.pending_damage_ctx = None
        # Objetivo del próximo |-damage| marcado por |-crit| (None = sin crit pendiente).
        self.pending_crit_target = None
        self.pending_damage_sash = False

        # =====================================================================
        # 4. Reiniciar Recompensas
        # =====================================================================
        self.reward = 0
        self.score = 0
        self.spec_c2_return = 0.0
        self.spec_c1_return = 0.0
        self.last_c2_reward = 0.0
        self.last_c1_reward = 0.0
        self._spec_prev_our = None
        self._spec_prev_foe = None
        self.last_expectimax_meta = None
        self.last_preview_meta = None

        # Nota: Los 'Tag Globales' (self.battleBO3_tag, self.battleBO3_count, 
        # self.botMode, self.battle_tag) NO se tocan y conservan sus valores.

    def append_game_banner(self, kind: str, winner: str | None = None, to_console: bool = True) -> None:
        """Separador entre partidas del BO3 (START al preview, END al |win|)."""
        game_n = getattr(self, 'battleBO3_count', '?')
        user_n = getattr(self.user, 'account_name', None) or '?'
        opp_n = getattr(self.opponent, 'account_name', None) or '?'
        lines = ['#' * 72]
        if kind.upper() == 'START':
            lines.append(f' BO3 GAME {game_n} START  |  {user_n} vs {opp_n}')
        else:
            win_s = winner or '?'
            lines.append(f' BO3 GAME {game_n} END  |  win: {win_s}  |  {user_n} vs {opp_n}')
        lines.append('#' * 72)
        text = '\n'.join(lines) + '\n'
        _append_report_file('\n' + text)
        if to_console:
            _write_stdout(text)

############################################################################################################################################################
# OK - Impresión de TODA la Batalla (CMD = fuente principal turno a turno)                                                                               ###       
############################################################################################################################################################
    def imprimir(self, to_console: bool = True, accion=None):
        """
        Reporte de estado:
        - Siempre a logs/latest_state.txt y logs/reporte.txt
        - Por defecto también a CMD (revisión turno a turno)
        - accion: se incluye dentro del bloque (CMD + archivo)
        """
        import os
        import io
        from datetime import datetime

        os.makedirs('logs', exist_ok=True)
        stamp = datetime.now().strftime('%H:%M:%S')
        user_n = getattr(self.user, 'account_name', None) or '?'
        opp_n = getattr(self.opponent, 'account_name', None) or '?'
        kind = _snap_kind(self)

        buf = io.StringIO()

        def w(line: str = ''):
            print(line, file=buf)

        # Banner START en team preview (turno 0)
        if int(getattr(self, 'turn', 0) or 0) == 0 and not getattr(self, '_report_game_started', False):
            self._report_game_started = True
            # Se escribe aparte para no duplicar en latest_state; lo hacemos aquí embebido
            w('#' * 72)
            w(f' BO3 GAME {getattr(self, "battleBO3_count", "?")} START  |  {user_n} vs {opp_n}')
            w('#' * 72)

        w('=' * 72)
        w(
            f' TURNO {self.turn} [{kind}]  |  rqid {self.rqid}  |  '
            f'BO3 game {getattr(self, "battleBO3_count", "?")}  |  {stamp}'
        )
        w(f' {user_n}  vs  {opp_n}')
        w(
            f' clima={self.weather or "-"}({self.weatherTurn})  '
            f'campo={_fmt_mapping(self.terrain)}'
        )
        w('-' * 72)
        w(f' YO [{self.user.id}]  side={_fmt_mapping(self.user.side_conditions)}')
        self.user.imprimir_cmd(buf)
        w('-' * 72)
        w(
            f' RIVAL [{self.opponent.id}]  side={_fmt_mapping(self.opponent.side_conditions)}  '
            f'confirmados={self.opponent.opponentPkmnConfirmed}'
        )
        self.opponent.imprimir_cmd(buf)

        # Estimación de Stat Points del rival (evolución turno a turno).
        # El reporte siempre usa reglas Champions (32/66), aunque el cálculo
        # interno tuviera candidatos más anchos por tests u otros callers.
        beliefs = getattr(self, 'stat_beliefs', None)
        if beliefs is not None and beliefs.beliefs:
            w('-' * 72)
            w(f' STAT POINTS ESTIMADOS (rival)  |  tope {SP_MAX}/stat, {SP_BUDGET} total:')
            for species, belief in sorted(beliefs.beliefs.items()):
                forme = (getattr(belief, 'current_forme', None) or species).lower()
                forme_tag = f'  forme={forme}' if forme != species else ''
                w(f'  {species}  nat={belief.nature or "-"}{forme_tag}')
                w(f'      SP:   {belief.sp_summary_line_for_report()}')
                w(f'      est:  {belief.summary_line_for_report()}')
                # Mega/Blade: mismos SP, otras bases → el final "base" (p.ej. Atk 92)
                # no tiene por qué caer en el rango mega (Atk 110-139).
                if forme != species:
                    dex = dex_base_stats(species)
                    if dex:
                        w(
                            f'      base: {belief.summary_line_with_bases_for_report(dex)}'
                            f'  (forma {species})'
                        )
                w(f'      conv: {belief.convergence_line()}')

        # OTS solo una vez por partida (team preview / primer dump)
        dump_ots = not getattr(self, '_report_ots_dumped', False)
        if dump_ots and getattr(self.opponent, 'opponents', None):
            w('-' * 72)
            w(' OTS (plantilla showteam; una vez por partida):')
            for i, pkmn in enumerate(self.opponent.opponents, start=1):
                if pkmn is None:
                    continue
                moves = ','.join(m for m in (pkmn.moves or []) if m) or '-'
                w(
                    f'  {i}. {pkmn.name}  item={pkmn.objectPkmn or "-"}  '
                    f'abi={pkmn.abilityPkmn or "-"}  moves={moves}'
                )
            self._report_ots_dumped = True

        if accion is not None:
            w('-' * 72)
            w(f' ACCION -> {accion}')

        # Spec Candidate 2: reward del turno + retorno acumulado (archivo + CMD).
        if (
            hasattr(self, 'last_c2_reward')
            or hasattr(self, 'spec_c2_return')
            or hasattr(self, 'lastReward')
        ):
            w('-' * 72)
            c2_t = float(getattr(self, 'last_c2_reward', 0.0) or 0.0)
            c2_ep = float(getattr(self, 'spec_c2_return', 0.0) or 0.0)
            c1_t = float(getattr(self, 'last_c1_reward', 0.0) or 0.0)
            c1_ep = float(getattr(self, 'spec_c1_return', 0.0) or 0.0)
            legacy_d = float(getattr(self, 'lastReward', 0.0) or 0.0)
            legacy_r = float(getattr(self, 'reward', 0.0) or 0.0)
            w(' REWARD (Spec v1.1 Candidate 2 dense / turno):')
            w(f'  c2_turn={c2_t:+.4f}  c2_episode={c2_ep:+.4f}')
            w(f'  c1_turn={c1_t:+.0f}  c1_episode={c1_ep:+.0f}')
            w(f'  ui_delta={legacy_d:.0f}  ui_total={legacy_r:.0f}')

        # Expectimax Top-K / preview: resumen corto en archivo; detalle solo CMD.
        emeta = getattr(self, 'last_expectimax_meta', None)
        pmeta = getattr(self, 'last_preview_meta', None)
        if emeta is not None:
            w('-' * 72)
            w(' EXPECTIMAX: ver bloque Top-K en CMD (decision + why)')
            summary = str(getattr(emeta, 'decision_summary', '') or '').strip()
            if summary:
                first = summary.splitlines()[0]
                w(f'  {first}')
        if pmeta is not None and int(getattr(self, 'turn', 0) or 0) == 0:
            w('-' * 72)
            w(' TEAM PREVIEW: ver bloque ranking en CMD (4 de 6 + why)')

        w('=' * 72)

        report = buf.getvalue()

        with open('logs/latest_state.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        _append_report_file('\n' + report)

        if to_console:
            _write_stdout(report)
            # Solo CMD: ranking K opciones + por qué se eligió esa combinación.
            if emeta is not None:
                cmd = []
                cmd.append('-' * 72)
                cmd.append(' EXPECTIMAX DECISION (CMD) — Top-K pares por V bayesiana')
                summary = str(getattr(emeta, 'decision_summary', '') or '').strip()
                if summary:
                    for line in summary.splitlines():
                        cmd.append(f'  {line}')
                lines = list(getattr(emeta, 'topk_lines', None) or [])
                if lines:
                    cmd.append('  ranked ( > = elegido ):')
                    for line in lines:
                        cmd.append(f'  {line}')
                else:
                    cmd.append('  ranked: (sin candidatos phase2; floor phase1)')
                why = str(getattr(emeta, 'decision_why', '') or '').strip()
                if why:
                    cmd.append('  por qué:')
                    for line in why.splitlines():
                        cmd.append(f'    {line}')
                cmd.append('=' * 72)
                _write_stdout('\n'.join(cmd) + '\n')
            if pmeta is not None and int(getattr(self, 'turn', 0) or 0) == 0:
                cmd = []
                cmd.append('-' * 72)
                cmd.append(' TEAM PREVIEW DECISION (CMD) — 4 de 6 + lead')
                cmd.append(
                    f'  scored={getattr(pmeta, "n_scored", 0)}  '
                    f'C(6,4)={getattr(pmeta, "n_combos_4", 0)}  '
                    f'lead_orders_total={getattr(pmeta, "n_lead_orders", 0)}'
                )
                lines = list(getattr(pmeta, 'ranked_lines', None) or [])
                if lines:
                    cmd.append('  ranked ( > = elegido ):')
                    for line in lines:
                        cmd.append(f'  {line}')
                why_lines = list(getattr(pmeta, 'why_lines', None) or [])
                if why_lines:
                    cmd.append('  por qué:')
                    for line in why_lines:
                        cmd.append(f'    {line}')
                cmd.append('=' * 72)
                _write_stdout('\n'.join(cmd) + '\n')

        return report.strip().splitlines()[0] if report else ''


############################################################################################################################################################
# PENDIENTE - Cálculos de las Recompensas                                                                                                                ###
############################################################################################################################################################
    def calculateRewards(self):
        """UI rewards: Spec C2 delta / episode return (×100 for legacy UI scale)."""
        c2_t = float(getattr(self, 'last_c2_reward', 0.0) or 0.0)
        c2_ep = float(getattr(self, 'spec_c2_return', 0.0) or 0.0)
        # Escala suave a la UI legacy (~centenas). No floor aquí: los bots
        # deben usar round(..., 1) al empujar a la cola (chips pequeños).
        self.lastReward = c2_t * 100.0
        self.reward = c2_ep * 100.0
        return self.lastReward, self.reward


############################################################################################################################################################
# OK - CLASE BATTLER                                                                                                                                     ###
############################################################################################################################################################
class Battler:
    def __init__(self):
        self.id = None                                  # Ejm: p1
        self.account_name = None                        # String
        self.active1 = None                             # class Pokemon or None
        self.active2 = None                             # class Pokemon or None
        self.reserved = []                              # class Pokemon
        self.opponents = []                             # class Pokemon
        self.side_conditions = defaultdict(lambda: 0)   # Tailwind, Stealth Rock, Reflect
        self.isUser = False                             # Se actualiza en True solo cuando es Usuario
        self.opponentPkmnConfirmed = 0                  # Contabilizador de reservados


###############################################################################################################################################################
# OK - Actualiza TODOS los datos de los PKMN del usuario en base al JSON recibido por cada tipo de |request|                                                ###
###############################################################################################################################################################
    def updatePkmnUser(self, JSONdict: Union[str, Dict[str, Any]], first_turn: bool, natures: list) -> None:
        if (first_turn): # TEAM PREVIEW - 6 PKMN
            # Datos Básicos
            self.id = JSONdict["side"]["id"]
            self.account_name = JSONdict["side"]["name"]
            self.isUser = True

            for i, pkmn_data in enumerate(JSONdict["side"]["pokemon"]):
                # Normalizar nombre
                raw_name = pkmn_data["details"].partition(",")[0].lower()
                namePkmn = constants.standardName.get(raw_name, raw_name)

                # Extraer datos básicos
                max_hp = pkmn_data["condition"].partition("/")[2]
                item = pkmn_data.get("item", "")
                ability = pkmn_data.get("ability", "")
                
                # Extraer hasta 4 movimientos
                moves = pkmn_data.get("moves", [])
                m1 = moves[0] if len(moves) > 0 else None
                m2 = moves[1] if len(moves) > 1 else None
                m3 = moves[2] if len(moves) > 2 else None
                m4 = moves[3] if len(moves) > 3 else None

                # Crear instancia
                pkmn = Pokemon(namePkmn, max_hp, False, item, ability, m1, m2, m3, m4, natures[i])

                # Asignar stats en combate
                stats = pkmn_data["stats"]
                pkmn.ATK_battle_actual = int(stats["atk"])
                pkmn.DEF_battle_actual = int(stats["def"])
                pkmn.SPA_battle_actual = int(stats["spa"])
                pkmn.SPD_battle_actual = int(stats["spd"])
                pkmn.VEL_battle_actual = int(stats["spe"])

                self.reserved.append(pkmn)
        else:
            request_pokemon = JSONdict["side"].get("pokemon", [])[:4]
            current_pokemon = [self.active1, self.active2, *self.reserved]
            ordered_pokemon = []
            # Conservar boosts/volatiles por identidad de objeto (el request no
            # los trae). Fallback por especie: el dict con mayor |stage| sumado
            # (evita que un shell vacío / reserved pise un active ya unboosted).
            _carry_boosts_by_id = {}
            _carry_volatile_by_id = {}
            _carry_boosts_by_species = {}
            _carry_volatile_by_species = {}
            _carry_score_by_species = {}

            def _party_key(name: str) -> str:
                raw = (name or '').lower().replace(' ', '-')
                raw = constants.standardName.get(raw, raw)
                compact = raw.replace('-', '').replace('_', '')
                for suf in (
                    'megax', 'megay', 'mega', 'eternal', 'blade', 'shield',
                    'busted', 'disguised', 'crowned',
                ):
                    if compact.endswith(suf) and len(compact) > len(suf) + 2:
                        compact = compact[: -len(suf)]
                        break
                if compact.startswith('floette'):
                    return 'floette'
                return compact

            def _boost_score(boosts) -> int:
                if not boosts:
                    return 0
                seen = {}
                for k, v in dict(boosts).items():
                    try:
                        iv = int(v or 0)
                    except Exception:
                        continue
                    if not iv:
                        continue
                    sk = str(k).strip().lower().replace('_', '').replace('-', '')
                    aliases = {
                        'attack': 'atk', 'defense': 'def',
                        'specialattack': 'spa', 'spatk': 'spa',
                        'specialdefense': 'spd', 'spdef': 'spd',
                        'speed': 'spe',
                    }
                    short = aliases.get(sk, sk if sk in (
                        'atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion',
                    ) else sk)
                    prev = seen.get(short, 0)
                    if abs(iv) >= abs(prev):
                        seen[short] = iv
                return sum(abs(v) for v in seen.values())

            for p in current_pokemon:
                if p is None:
                    continue
                _carry_boosts_by_id[id(p)] = deepcopy(getattr(p, 'boosts', None))
                _carry_volatile_by_id[id(p)] = deepcopy(getattr(p, 'volatile_statuses', None))
                compact = _party_key(getattr(p, 'name', '') or '')
                score = _boost_score(getattr(p, 'boosts', None))
                if score >= _carry_score_by_species.get(compact, -1):
                    _carry_score_by_species[compact] = score
                    _carry_boosts_by_species[compact] = deepcopy(getattr(p, 'boosts', None))
                    _carry_volatile_by_species[compact] = deepcopy(
                        getattr(p, 'volatile_statuses', None)
                    )

            def request_pokemon_name(pkmn_data):
                """Forme completa del request (autoridad de especie en slot)."""
                raw_name = pkmn_data["details"].partition(",")[0].lower().replace(" ", "-")
                return constants.standardName.get(raw_name, raw_name)

            def create_pokemon(pkmn_data, name):
                condition = pkmn_data.get("condition", "0/0")
                max_hp = condition.partition("/")[2].partition(" ")[0]
                moves = pkmn_data.get("moves", [])[:4]
                pkmn = Pokemon(
                    name,
                    max_hp,
                    False,
                    pkmn_data.get("item", ""),
                    pkmn_data.get("ability", ""),
                    *(moves + [None] * (4 - len(moves))),
                    None,
                )
                return pkmn

            for pkmn_data in request_pokemon:
                target_name = request_pokemon_name(pkmn_data)
                target_key = _party_key(target_name)
                found_pkmn = next(
                    (
                        pkmn for pkmn in current_pokemon
                        if pkmn is not None and pkmn not in ordered_pokemon
                        and (
                            _party_key(getattr(pkmn, 'name', '')) == target_key
                            or (pkmn.is_ditto and target_key == 'ditto')
                        )
                    ),
                    None,
                )
                if found_pkmn is None:
                    found_pkmn = create_pokemon(pkmn_data, target_name)
                else:
                    # Alinear forme del request (Mega-Y vs base, etc.).
                    # Un Ditto transformado conserva la identidad copiada: el request
                    # sigue diciendo details=Ditto y no debe pisar el nombre.
                    if not (found_pkmn.is_ditto and target_key == 'ditto'):
                        found_pkmn.name = target_name
                    if 'mega' in target_name.replace('-', ''):
                        found_pkmn.is_mega_evo = True
                if found_pkmn is not None:
                    ordered_pokemon.append(found_pkmn)

            if len(ordered_pokemon) >= 2:
                self.active1, self.active2 = ordered_pokemon[:2]
                self.reserved = ordered_pokemon[2:4]
                for mon in ordered_pokemon:
                    if mon is None:
                        continue
                    key = _party_key(getattr(mon, 'name', '') or '')
                    # Merge: no reemplazar un dict rico (p.ej. Intimidate) por uno
                    # vacío/parcial del carry al reordenar el |request|.
                    # Importante: un carry-by-id vacío NO debe bloquear el carry por
                    # especie (pasa tras rol p1/p2 invertido o shells fabricados).
                    candidates = []
                    if id(mon) in _carry_boosts_by_id:
                        candidates.append(_carry_boosts_by_id[id(mon)])
                    if key in _carry_boosts_by_species:
                        candidates.append(_carry_boosts_by_species[key])
                    # Fusionar todos los carries (no quedarse con el de mayor
                    # score): id vacío + especie con Intimidate debe conservar atk.
                    merged = defaultdict(lambda: 0)
                    for src in (getattr(mon, 'boosts', None), *candidates):
                        if not src:
                            continue
                        for bk, bv in dict(src).items():
                            try:
                                iv = int(bv or 0)
                            except Exception:
                                continue
                            if not iv:
                                continue
                            sk = str(bk).strip().lower().replace('_', '').replace('-', '')
                            aliases = {
                                'attack': 'atk', 'defense': 'def',
                                'specialattack': 'spa', 'spatk': 'spa',
                                'specialdefense': 'spd', 'spdef': 'spd',
                                'speed': 'spe',
                            }
                            short = aliases.get(sk, sk if sk in (
                                'atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion',
                            ) else sk)
                            prev = int(merged.get(short, 0) or 0)
                            if abs(iv) >= abs(prev):
                                merged[short] = iv
                    if any(merged.values()):
                        mon.boosts = merged
                    vol_id = _carry_volatile_by_id.get(id(mon))
                    vol_sp = _carry_volatile_by_species.get(key)
                    pick_v = vol_id if vol_id is not None else vol_sp
                    if pick_v is not None:
                        cur_v = getattr(mon, 'volatile_statuses', None) or {}
                        if not any(dict(cur_v).values()):
                            mon.volatile_statuses = deepcopy(pick_v)

            # =========================================================================
            # UPDATE POKÉMON ACTIVO 1 (Izquierda - Índice 0 del JSON)
            # =========================================================================
            pokemon0 = JSONdict["side"]["pokemon"][0]
            cond0 = pokemon0["condition"]
            if "/" in cond0:
                self.active1.HPA_battle_actual = int(cond0.partition("/")[0].partition(" ")[0])
                self.active1.HPB_battle_base = int(cond0.partition("/")[2].partition(" ")[0])
            else:
                self.active1.HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")

            # Estadísticas de combate (Activo 1)
            stats1 = pokemon0["stats"]
            if not (self.active1.is_ditto and self.active1.name != "ditto"):
                self.active1.ATK_battle_actual = int(stats1["atk"])
                self.active1.SPA_battle_actual = int(stats1["spa"])
                self.active1.VEL_battle_actual = int(stats1["spe"])
                # Guard Split altera Def/SpD en el sim; el request no lo refleja
                if not getattr(self.active1, 'guard_split_active', False):
                    self.active1.DEF_battle_actual = int(stats1["def"])
                    self.active1.SPD_battle_actual = int(stats1["spd"])

            # Item / ability / faint desde el request (autoridad del protocolo)
            self.active1.objectPkmn = normalize_name(pokemon0.get("item", "") or "") or None
            req_ability1 = normalize_name(pokemon0.get("ability", "") or "")
            if req_ability1:
                self.active1.abilityPkmn = req_ability1
            if self.active1.HPA_battle_actual <= 0 or "fnt" in str(cond0).lower():
                self.active1.is_fainted = True
            # Showdown mantiene details=Ditto aunque esté transformado; solo
            # forzamos identidad base al debilitarse.
            if self.active1.is_ditto and self.active1.is_fainted:
                self.active1.name = "ditto"
                self.active1.is_zoroark = False
                self.active1.abilityPkmn = "imposter"
                self.active1.moves = ["transform", None, None, None]

            # =========================================================================
            # UPDATE POKÉMON ACTIVO 2 (Derecha - Índice 1 del JSON)
            # =========================================================================
            pokemon1 = JSONdict["side"]["pokemon"][1]
            cond1 = pokemon1["condition"]
            if "/" in cond1:
                self.active2.HPA_battle_actual = int(cond1.partition("/")[0].partition(" ")[0])
                self.active2.HPB_battle_base = int(cond1.partition("/")[2].partition(" ")[0])
            else:
                self.active2.HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Activo 2)
            stats2 = pokemon1["stats"]
            if not (self.active2.is_ditto and self.active2.name != "ditto"):
                self.active2.ATK_battle_actual = int(stats2["atk"])
                self.active2.SPA_battle_actual = int(stats2["spa"])
                self.active2.VEL_battle_actual = int(stats2["spe"])
                if not getattr(self.active2, 'guard_split_active', False):
                    self.active2.DEF_battle_actual = int(stats2["def"])
                    self.active2.SPD_battle_actual = int(stats2["spd"])

            self.active2.objectPkmn = normalize_name(pokemon1.get("item", "") or "") or None
            req_ability2 = normalize_name(pokemon1.get("ability", "") or "")
            if req_ability2:
                self.active2.abilityPkmn = req_ability2
            if self.active2.HPA_battle_actual <= 0 or "fnt" in str(cond1).lower():
                self.active2.is_fainted = True
            if self.active2.is_ditto and self.active2.is_fainted:
                self.active2.name = "ditto"
                self.active2.is_zoroark = False
                self.active2.abilityPkmn = "imposter"
                self.active2.moves = ["transform", None, None, None]

            # Asegurándonos de que 'active' tenga al menos 2 elementos en la lista
            request_active = JSONdict.get("active", [])
            
            # Para el Pokémon de la izquierda (active1 - índice 0)
            active1_info = request_active[0] if len(request_active) > 0 else {}
            self.active1.can_mega_evo = bool(active1_info.get("canMegaEvo", False))
            
            # Para el Pokémon de la derecha (active2 - índice 1)
            active2_info = request_active[1] if len(request_active) > 1 else {}
            self.active2.can_mega_evo = bool(active2_info.get("canMegaEvo", False))

            # Update Pkmn - Turno 2 en adelante... Moves y PP
            request_active = JSONdict.get("active") or []
            for index, active_pkmn in enumerate((self.active1, self.active2)):
                if active_pkmn is None or index >= len(request_active):
                    continue

                move_data_list = request_active[index].get("moves", [])
                active_pkmn.moves = [
                    move_data.get("id")
                    for move_data in move_data_list[:4]
                ]
                active_pkmn.moves.extend([None] * (4 - len(active_pkmn.moves)))
                active_pkmn.movesPP = [
                    1 if move_data.get("id") == "struggle" else (
                        int(move_data["pp"]) if move_data.get("pp") is not None else -1
                    )
                    for move_data in move_data_list[:4]
                ]
                active_pkmn.movesPP.extend([-1] * (4 - len(active_pkmn.movesPP)))

            # =========================================================================
            # UPDATE POKÉMON RESERVA 1 (Index 2 del JSON)
            # =========================================================================
            pokemon2 = JSONdict["side"]["pokemon"][2]
            cond2 = pokemon2["condition"]
            if "/" in cond2:
                self.reserved[0].HPA_battle_actual = int(cond2.partition("/")[0].partition(" ")[0])
                self.reserved[0].HPB_battle_base = int(cond2.partition("/")[2].partition(" ")[0])
            else:
                self.reserved[0].HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Reserva 1)
            stats3 = pokemon2["stats"]
            self.reserved[0].ATK_battle_actual = int(stats3["atk"])
            self.reserved[0].DEF_battle_actual = int(stats3["def"])
            self.reserved[0].SPA_battle_actual = int(stats3["spa"])
            self.reserved[0].SPD_battle_actual = int(stats3["spd"])
            self.reserved[0].VEL_battle_actual = int(stats3["spe"])

            # =========================================================================
            # UPDATE POKÉMON RESERVA 2 (Index 3 del JSON)
            # =========================================================================
            pokemon3 = JSONdict["side"]["pokemon"][3]
            cond3 = pokemon3["condition"]
            if "/" in cond3:
                self.reserved[1].HPA_battle_actual = int(cond3.partition("/")[0].partition(" ")[0])
                self.reserved[1].HPB_battle_base = int(cond3.partition("/")[2].partition(" ")[0])
            else:
                self.reserved[1].HPA_battle_actual = 0  # Debilitado (ej. "0 fnt")
            
            # Estadísticas de combate (Reserva 2)
            stats4 = pokemon3["stats"]
            self.reserved[1].ATK_battle_actual = int(stats4["atk"])
            self.reserved[1].DEF_battle_actual = int(stats4["def"])
            self.reserved[1].SPA_battle_actual = int(stats4["spa"])
            self.reserved[1].SPD_battle_actual = int(stats4["spd"])
            self.reserved[1].VEL_battle_actual = int(stats4["spe"])


###############################################################################################################################################################
# OK - Impresión de la clase Battler (CMD denso + legacy archivo detallado)                                                                                ###
###############################################################################################################################################################
    def imprimir_cmd(self, f) -> None:
        """Bloque compacto para consola: activos + reserva en pocas líneas."""
        is_opp = not self.isUser

        def slot_block(label: str, pkmn):
            if pkmn is None:
                print(f'  {label}  (vacio)', file=f)
                return
            types = f"{pkmn.type1 or '?'}" + (f"/{pkmn.type2}" if pkmn.type2 else '')
            flags = _fmt_flags_short(pkmn, is_opp)
            flag_s = f'  [{flags}]' if flags else ''
            print(
                f'  {label}  {_fmt_pkmn_display_name(pkmn)} [{types}]  {_fmt_hp(pkmn)}{flag_s}',
                file=f,
            )
            print(
                f'       item={pkmn.objectPkmn or "-"}  abi={pkmn.abilityPkmn or "-"}  '
                f'status={_fmt_status_short(pkmn)}  boosts={_fmt_boosts_short(pkmn.boosts)}',
                file=f,
            )
            print(f'       stats: {_fmt_battle_stats_short(pkmn)}', file=f)
            moves = [f'{m}({pp})' for m, pp in zip(pkmn.moves or [], pkmn.movesPP or []) if m]
            print(f'       moves: {", ".join(moves) if moves else "-"}', file=f)

        slot_block('A1', self.active1)
        slot_block('A2', self.active2)

        if self.reserved:
            parts = []
            for p in self.reserved:
                if p is None:
                    continue
                tags = []
                if getattr(p, 'is_fainted', False):
                    tags.append('FNT')
                if getattr(p, 'is_zoroark', False) and not getattr(p, 'illusion_disguise', None):
                    tags.append('ZORO')
                if getattr(p, 'is_ditto', False) and str(getattr(p, 'name', '')).lower() == 'ditto':
                    tags.append('DITTO')
                tag_s = f"[{','.join(tags)}]" if tags else ''
                parts.append(f'{_fmt_pkmn_display_name(p)}{tag_s} {_fmt_hp(p)}')
            print(f'  RSV  {" | ".join(parts) if parts else "(vacia)"}', file=f)
        else:
            print('  RSV  (vacia)', file=f)

    def imprimir(self, f):
        """Formato detallado legacy (por si se llama fuera de imprimir_cmd)."""
        print(f"ID: {self.id} | Cuenta: {self.account_name}", file=f)
        if not self.isUser:
            print(f"Confirmados oponente: {self.opponentPkmnConfirmed}", file=f)
        print(f"Side conditions: {_fmt_mapping(self.side_conditions)}", file=f)
        self.imprimir_cmd(f)


############################################################################################################################################################
# OK - CLASE POKEMON                                                                                                                                     ###
############################################################################################################################################################
class Pokemon:
    def __init__(
        self, 
        name: str, 
        HPB_battle_base: int, 
        is_opponent: bool, 
        object_pkmn: str, 
        ability_pkmn: str, 
        move1: str, 
        move2: str, 
        move3: str, 
        move4: str, 
        nature: str
    ):
        # Identidad y Configuración Básica
        self.name = name
        self.nature = nature
        self.is_opponent = is_opponent
        self.objectPkmn = object_pkmn
        self.abilityPkmn = ability_pkmn
        self.is_mega_evo = False
        self.can_mega_evo = False
        self.is_opponent_confirmed = False
        self.is_ditto = True if str(name).lower() == 'ditto' else False
        _name_l = str(name).lower()
        self.is_zoroark = _name_l == 'zoroark' or _name_l.startswith('zoroark-')
        # Especie disfraz vista en PS mientras Illusion no se revela / se acaba de deducir
        self.illusion_disguise = None
        # True solo tras |replace| real del protocolo (PS ya muestra Zoroark)
        self.illusion_protocol_revealed = False
        # Guard Split activo: no pisar Def/SpD con el |request| (PS no las refleja ahí)
        self.guard_split_active = False
        self.is_fainted = False
        # Yawn: True desde |-start|move: Yawn hasta dormir / salir / |-end|Yawn.
        # Durante el turno de decisión previo al slp el mon sigue despierto y puede actuar.
        self.is_drowsy = False
        # Perish Song: 0 = sin efecto; 3/2/1 = contador visible de Showdown (| -start|perishN).
        self.perish_count = 0
        # Efectos efímeros (no se imprimen en el snap de decisión; se limpian a tiempo).
        # Protect/Detect/…: solo el turno actual (| -singleturn| → clear en |turn|).
        self.is_protected = False
        # Helping Hand recibido: solo el turno actual.
        self.is_helping_hand = False
        # Destiny Bond: hasta el próximo |move| propio, faint o switch.
        self.is_destiny_bonded = False
        # Charge: hasta move Eléctrico / |-end| / switch.
        self.is_charged = False
        # Must recharge (Hyper Beam…): hasta el próximo |move| propio o switch.
        self.must_recharge = False
        # Follow Me / Rage Powder / Spotlight: redirige ataques single-target.
        self.is_redirecting = False
        # abilityOn del calc (@smogon): Flash Fire absorbed, Slow Start activo, etc.
        self.ability_on = False
        # Entró este turno (Stakeout del rival atacante).
        self.just_switched = False
        # Terastallize (Champions / gen9).
        self.tera_type = None
        self.is_terastallized = False

        # Tipos (Se asignan posteriormente)
        self.type1 = None
        self.type2 = None

        # Estados y Modificadores de Combate
        self.non_volatile_statuses = defaultdict(lambda: 0)  # Ej: ['slp', 'brn', 'frz', 'par', 'psn']
        self.volatile_statuses = defaultdict(lambda: 0)      # Ej: 'yawn', 'confusion', 'substitute'
        self.boosts = defaultdict(lambda: 0)                 # Modificadores de estadísticas (-6 a +6)

        # Estadísticas Base - Solo se llena una vez para el Oponente, -1 para el Usuario
        self.HPB_battle_base = HPB_battle_base
        self.ATK_battle_base = -1
        self.DEF_battle_base = -1
        self.SPA_battle_base = -1
        self.SPD_battle_base = -1
        self.VEL_battle_base = -1

        # Estadísticas Actuales - Se actualiza por cada turno, -1 para el Oponente
        self.HPA_battle_actual = HPB_battle_base
        self.ATK_battle_actual = -1
        self.DEF_battle_actual = -1
        self.SPA_battle_actual = -1
        self.SPD_battle_actual = -1
        self.VEL_battle_actual = -1

        # Movimientos y PP
        self.moves = [move1, move2, move3, move4]
        self.movesPP = [-1, -1, -1, -1]

        # Actualizar atributos
        self.getStatsBattle(is_opponent)


############################################################################################################################################################
# OK - Actualización Otros Atributos POKEAPI                                                                                                             ###
############################################################################################################################################################
    def getStatsBattle(self, isOpponent) -> None:
        info = getPkmnInfo(self.name)
        if not info:
            return
        type1, type2, HPB_battle_base, ATK_battle_base, DEF_battle_base, SPA_battle_base, SPD_battle_base, VEL_battle_base = info
        if isOpponent:
            self.type1, self.type2, self.HPB_battle_base, self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base = type1, type2, HPB_battle_base, ATK_battle_base, DEF_battle_base, SPA_battle_base, SPD_battle_base, VEL_battle_base
            self.HPA_battle_actual = 100
        else:
            self.type1, self.type2 = type1, type2


############################################################################################################################################################
# OK - Impresión de la clase Pokemon                                                                                                                     ###
############################################################################################################################################################
    def imprimir(self, f, isOpponent) -> None:
        """Impresión compacta orientada a pruebas / depuración de estado."""
        flags = []
        if self.is_fainted:
            flags.append('FAINTED')
        if getattr(self, 'is_drowsy', False):
            flags.append('DROWSY')
        if getattr(self, 'perish_count', 0):
            flags.append(f'PERISH{self.perish_count}')
        if self.is_ditto:
            flags.append('DITTO')
        if self.is_zoroark:
            flags.append('ZOROARK')
        if self.is_mega_evo:
            flags.append('MEGA')
        if self.can_mega_evo:
            flags.append('CAN_MEGA')
        if isOpponent and self.is_opponent_confirmed:
            flags.append('CONFIRMED')

        types = f"{self.type1}" + (f"/{self.type2}" if self.type2 else "")
        hp_pct = ''
        try:
            max_hp = int(float(str(self.HPB_battle_base).split()[0]))
            cur_hp = int(float(str(self.HPA_battle_actual).split()[0]))
            if max_hp > 0:
                hp_pct = f" ({100 * cur_hp / max_hp:.0f}%)"
        except Exception:
            hp_pct = ''

        print(f"    {self.name} [{types}]  HP {self.HPA_battle_actual}/{self.HPB_battle_base}{hp_pct}", file=f)
        print(
            f"      item={self.objectPkmn or '-'} | abi={self.abilityPkmn or '-'} | nat={self.nature or '-'}",
            file=f,
        )
        if flags:
            print(f"      flags: {', '.join(flags)}", file=f)
        print(
            f"      status={_fmt_mapping(self.non_volatile_statuses)} | "
            f"volatile={_fmt_mapping(self.volatile_statuses)} | "
            f"boosts={_fmt_mapping(self.boosts)}",
            file=f,
        )
        print(
            f"      stats: {_fmt_stat_line(self.ATK_battle_actual, self.DEF_battle_actual, self.SPA_battle_actual, self.SPD_battle_actual, self.VEL_battle_actual)}",
            file=f,
        )
        if any(v != -1 for v in (self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base)):
            print(
                f"      base:  {_fmt_stat_line(self.ATK_battle_base, self.DEF_battle_base, self.SPA_battle_base, self.SPD_battle_base, self.VEL_battle_base)}",
                file=f,
            )
        moves = [f"{move}({pp})" for move, pp in zip(self.moves, self.movesPP) if move]
        print(f"      moves: {', '.join(moves) if moves else '-'}", file=f)
