#!/usr/bin/env node
/**
 * Nucleo compartido del calc Champions (gen 0).
 * Usado por calc_worker.js (stdio) y calc_server.js (TCP).
 */
'use strict';

const path = require('path');

const CALC_ROOT = process.env.DAMAGE_CALC_DIST
  || path.resolve(__dirname, '..', '..', '..', 'damage-calc', 'calc');

let calculate, Generations, Pokemon, Move, Field;
try {
  ({calculate, Generations, Pokemon, Move, Field} = require(CALC_ROOT));
} catch (err) {
  process.stderr.write(`[calc_core] failed to load calc from ${CALC_ROOT}: ${err}\n`);
  process.exit(1);
}

const GEN = Generations.get(0);

function titleCaseItem(id) {
  if (!id) return undefined;
  const raw = String(id).trim();
  if (!raw) return undefined;
  if (raw.includes(' ') || raw.includes('-')) {
    return raw.split(/[\s-]+/).map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
  }
  return raw;
}

function applyStats(poke, stats, curHP) {
  if (!stats) return;
  for (const key of ['hp', 'atk', 'def', 'spa', 'spd', 'spe']) {
    const v = stats[key];
    if (typeof v === 'number' && v > 0) {
      poke.rawStats[key] = v;
      poke.stats[key] = v;
    }
  }
  const maxHp = (stats && typeof stats.hp === 'number' && stats.hp > 0) ? stats.hp : poke.maxHP();
  if (typeof curHP === 'number' && curHP > 0) {
    poke.originalCurHP = Math.min(curHP, maxHp);
  } else if (typeof stats.hp === 'number' && stats.hp > 0) {
    poke.originalCurHP = stats.hp;
  }
}

function buildPokemon(spec) {
  const boosts = Object.assign(
    {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    spec.boosts || {}
  );
  // Champions: en @smogon/calc los 'evs' son Stat Points. Si el caller manda
  // evs/statPoints los usamos; si manda stats finales, los aplicamos después.
  const evs = Object.assign(
    {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
    spec.evs || spec.statPoints || {}
  );
  const poke = new Pokemon(GEN, spec.species, {
    level: 50,
    nature: spec.nature || 'Serious',
    item: titleCaseItem(spec.item),
    ability: spec.ability || undefined,
    status: spec.status || '',
    boosts,
    evs,
  });
  applyStats(poke, spec.stats, spec.curHP);

  // IMPORTANTE: calculate() hace attacker.clone()/defender.clone(), y clone()
  // reconstruye el Pokemon desde EVs (SP), descartando rawStats mutados.
  // Congelamos los stats finales en un clone propio para que la inferencia SP
  // vea rolls distintos por cada valor de Atk/SpA/Def/SpD.
  if (spec.stats) {
    const frozen = {
      hp: poke.rawStats.hp,
      atk: poke.rawStats.atk,
      def: poke.rawStats.def,
      spa: poke.rawStats.spa,
      spd: poke.rawStats.spd,
      spe: poke.rawStats.spe,
    };
    const frozenCur = poke.originalCurHP;
    const cloneFrozen = function cloneFrozen() {
      const copy = Pokemon.prototype.clone.call(this);
      applyStats(copy, frozen, frozenCur);
      copy.clone = cloneFrozen;
      return copy;
    };
    poke.clone = cloneFrozen;
  }
  return poke;
}

function handleOne(msg) {
  const attacker = buildPokemon(msg.attacker);
  const defender = buildPokemon(msg.defender);
  const moveOpts = {};
  if (msg.field && msg.field.isSpread) moveOpts.isSpread = true;
  if (msg.isCrit) moveOpts.isCrit = true;
  const move = new Move(GEN, msg.move, moveOpts);

  const fieldOpts = {
    gameType: (msg.field && msg.field.gameType) || 'Doubles',
    weather: (msg.field && msg.field.weather) || undefined,
    terrain: (msg.field && msg.field.terrain) || undefined,
    isWonderRoom: !!(msg.field && msg.field.isWonderRoom),
    attackerSide: (msg.field && msg.field.attackerSide) || {},
    defenderSide: (msg.field && msg.field.defenderSide) || {},
  };
  const field = new Field(fieldOpts);
  const result = calculate(GEN, attacker, defender, move, field);
  const range = result.range();
  return {ok: true, min: range[0], max: range[1]};
}

function handleMessage(msg) {
  if (msg.cmd === 'ping') {
    return {ok: true, pong: true};
  }
  if (msg.cmd === 'batch' && Array.isArray(msg.items)) {
    const results = [];
    for (const item of msg.items) {
      try {
        results.push(handleOne(item));
      } catch (err) {
        results.push({ok: false, error: String(err && err.message ? err.message : err)});
      }
    }
    return {ok: true, results};
  }
  return handleOne(msg);
}

module.exports = {handleMessage, CALC_ROOT};
