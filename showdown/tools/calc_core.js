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

const GEN9 = Generations.get(9);

function _dexLookupName(collection, toId) {
  if (!toId) return undefined;
  try {
    const fromChampions = collection.get(toId);
    if (fromChampions && fromChampions.name) return fromChampions.name;
  } catch (_) { /* fall through */ }
  return undefined;
}

function titleCaseItem(id) {
  if (!id) return undefined;
  const raw = String(id).trim();
  if (!raw) return undefined;
  // Preferir el nombre canónico del dex Champions (acepta id compacto).
  const toId = raw.toLowerCase().replace(/[^a-z0-9]+/g, '');
  const fromDex = _dexLookupName(GEN.items, toId)
    || _dexLookupName(GEN9.items, toId);
  if (fromDex) return fromDex;
  if (raw.includes(' ') || raw.includes('-')) {
    return raw.split(/[\s-]+/).map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
  }
  return raw;
}

function titleCaseAbility(id) {
  if (!id) return undefined;
  const raw = String(id).trim();
  if (!raw) return undefined;
  const toId = raw.toLowerCase().replace(/[^a-z0-9]+/g, '');
  // Gen 0 Champions omite varias abilities (p.ej. Grassy Surge); gen 9 las tiene.
  const fromDex = _dexLookupName(GEN.abilities, toId)
    || _dexLookupName(GEN9.abilities, toId);
  if (fromDex) return fromDex;
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

function speciesId(name) {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');
}

function speciesExists(name) {
  if (!name) return false;
  try {
    const id = speciesId(name);
    return !!(id && GEN.species.get(id));
  } catch (_) {
    return false;
  }
}

function resolveSpeciesName(name) {
  const id = speciesId(name);
  if (!id) return null;
  try {
    const s = GEN.species.get(id);
    return s && s.name ? s.name : null;
  } catch (_) {
    return null;
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

  // El dex indexa por id en minúsculas (kingambit), no por "Kingambit".
  const resolved = resolveSpeciesName(spec.species);
  const known = !!resolved;
  // Magikarp puede no estar en el dex Champions; Pikachu sí.
  const shellName = known ? resolved : (resolveSpeciesName('Pikachu') || 'Pikachu');
  const overrides = {};
  if (!known) {
    const bs = spec.stats || {};
    overrides.baseStats = {
      hp: bs.hp || 100,
      atk: bs.atk || 100,
      def: bs.def || 100,
      spa: bs.spa || 100,
      spd: bs.spd || 100,
      spe: bs.spe || 100,
    };
    if (Array.isArray(spec.types) && spec.types.length) {
      overrides.types = spec.types.filter(Boolean);
    }
    overrides.name = spec.species;
  }

  const pokeOpts = {
    level: 50,
    nature: spec.nature || 'Serious',
    item: titleCaseItem(spec.item),
    ability: titleCaseAbility(spec.ability) || spec.ability || undefined,
    status: spec.status || '',
    boosts,
    evs,
    overrides,
  };
  // Megastones: @smogon/calc intenta megaevolucionar y falla si la especie no
  // está bien en el dex Champions. Ya aplicamos stats/forma finales.
  if (pokeOpts.item) {
    const itemId = String(pokeOpts.item).toLowerCase().replace(/[^a-z0-9]+/g, '');
    if (itemId.endsWith('ite') && itemId !== 'eviolite') {
      pokeOpts.item = undefined;
    }
  }
  if (spec.teraType) pokeOpts.teraType = spec.teraType;
  if (spec.abilityOn) pokeOpts.abilityOn = true;
  // Supreme Overlord / (payload) Last Respects helpers
  if (typeof spec.alliesFainted === 'number' && spec.alliesFainted > 0) {
    pokeOpts.alliesFainted = Math.max(0, Math.floor(spec.alliesFainted));
  }
  const poke = new Pokemon(GEN, shellName, pokeOpts);
  if (!known) {
    poke.name = spec.species || poke.name;
    if (Array.isArray(spec.types) && spec.types.length) {
      poke.types = spec.types.filter(Boolean);
    }
  }
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
    const frozenTypes = poke.types && poke.types.slice ? poke.types.slice() : poke.types;
    const frozenName = poke.name;
    const cloneFrozen = function cloneFrozen() {
      const copy = Pokemon.prototype.clone.call(this);
      applyStats(copy, frozen, frozenCur);
      copy.types = frozenTypes;
      copy.name = frozenName;
      copy.clone = cloneFrozen;
      return copy;
    };
    poke.clone = cloneFrozen;
  }
  return poke;
}

function _moveId(name) {
  return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function handleOne(msg) {
  const attacker = buildPokemon(msg.attacker);
  const defender = buildPokemon(msg.defender);
  const moveOpts = {};
  if (msg.field && msg.field.isSpread) moveOpts.isSpread = true;
  if (msg.isCrit) moveOpts.isCrit = true;
  // @smogon/calc no escala Last Respects solo; PS: BP = 50 + 50 * side.totalFainted
  const fainted = (msg.attacker && typeof msg.attacker.alliesFainted === 'number')
    ? Math.max(0, Math.floor(msg.attacker.alliesFainted))
    : 0;
  if (_moveId(msg.move) === 'lastrespects' && fainted > 0) {
    moveOpts.overrides = {basePower: 50 + 50 * fainted};
  }
  const move = new Move(GEN, msg.move, moveOpts);

  const fieldOpts = {
    gameType: (msg.field && msg.field.gameType) || 'Doubles',
    weather: (msg.field && msg.field.weather) || undefined,
    terrain: (msg.field && msg.field.terrain) || undefined,
    isWonderRoom: !!(msg.field && msg.field.isWonderRoom),
    isMagicRoom: !!(msg.field && msg.field.isMagicRoom),
    isGravity: !!(msg.field && msg.field.isGravity),
    isBeadsOfRuin: !!(msg.field && msg.field.isBeadsOfRuin),
    isSwordOfRuin: !!(msg.field && msg.field.isSwordOfRuin),
    isTabletsOfRuin: !!(msg.field && msg.field.isTabletsOfRuin),
    isVesselOfRuin: !!(msg.field && msg.field.isVesselOfRuin),
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
  try {
    return handleOne(msg);
  } catch (err) {
    return {ok: false, error: String(err && err.message ? err.message : err)};
  }
}

module.exports = {handleMessage, CALC_ROOT};
