#!/usr/bin/env node
/**
 * Arranca pokemon-showdown con semillas de batalla deterministas.
 *
 * Showdown crea cada batalla con una semilla aleatoria, así que dos corridas con
 * el mismo `--seed` no dan el mismo resultado. Este lanzador envuelve
 * `Rooms.createBattle` y fija `options.seed` (campo que Showdown ya soporta).
 *
 * La semilla NO depende del orden de llegada (con varios workers por servidor el
 * orden varía entre corridas). Se calcula con:
 *
 *     sha256(PS_BATTLE_SEED : equipo del jugador 1 : equipo del jugador 2 : n)
 *
 * donde n es cuántas batallas con ese mismo emparejamiento ordenado se han
 * creado ya en este servidor (juego 1, 2 y 3 de un BO3 usan semillas distintas,
 * pero siempre las mismas entre corridas).
 *
 * Uso (cwd = pokemon-showdown, igual que `pokemon-showdown start`):
 *     PS_BATTLE_SEED=42 PSPORT=8000 node ps_seeded_start.js --no-security 8000
 *
 * Variables:
 *     PS_BATTLE_SEED  semilla base (obligatoria; si falta, arranque normal)
 *     PS_SEED_LOG     archivo opcional: una línea JSON por batalla {room, seed, key, n}
 */
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const root = process.env.POKEMON_SHOWDOWN_ROOT || process.cwd();
process.chdir(root);

const base = (process.env.PS_BATTLE_SEED || '').trim();
const logPath = (process.env.PS_SEED_LOG || '').trim();

// Arranca el servidor (mismo punto de entrada que `pokemon-showdown start`).
require(path.join(root, 'dist', 'server', 'index.js'));

if (!base) {
  process.stderr.write('[ps_seeded_start] PS_BATTLE_SEED no definido: semillas aleatorias\n');
} else {
  const {Rooms} = require(path.join(root, 'dist', 'server', 'rooms'));
  const original = Rooms.createBattle;
  const seen = new Map();

  const teamKey = (p) => String((p && (p.team || (p.user && p.user.battleSettings && p.user.battleSettings.team))) || '');

  Rooms.createBattle = function patchedCreateBattle(options) {
    try {
      if (options && !options.seed) {
        const players = options.players || [];
        const t1 = teamKey(players[0]);
        const t2 = teamKey(players[1]);
        const pairKey = crypto.createHash('sha256').update(`${t1}\u0000${t2}`).digest('hex').slice(0, 16);
        const n = seen.get(pairKey) || 0;
        seen.set(pairKey, n + 1);
        const digest = crypto.createHash('sha256')
          .update(`${base}:${t1}:${t2}:${n}`).digest('hex').slice(0, 32);
        options.seed = `sodium,${digest}`;
        options.__psSeedInfo = {key: pairKey, n};
      }
    } catch (err) {
      process.stderr.write(`[ps_seeded_start] no se pudo fijar la semilla: ${err && err.message}\n`);
    }
    const room = original.call(this, options);
    if (logPath && room && options && options.seed) {
      try {
        const info = options.__psSeedInfo || {};
        fs.appendFileSync(logPath, JSON.stringify({
          room: room.roomid, seed: options.seed, key: info.key, n: info.n,
        }) + '\n');
      } catch (err) { /* el log nunca debe romper una batalla */ }
    }
    return room;
  };
  process.stderr.write(`[ps_seeded_start] semillas deterministas activas (base=${base})\n`);
}
