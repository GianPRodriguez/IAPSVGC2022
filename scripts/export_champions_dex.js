// Exporta un snapshot congelado del dex Champions de Pokémon Showdown.
//
// Uso (desde la raíz del repo, con ../pokemon-showdown compilado):
//   node scripts/export_champions_dex.js [ruta_ps] [salida]
//
// Default: ruta_ps = ../pokemon-showdown, salida = API/JSONs/champions_dex_snapshot.json
// El snapshot lo consume showdown/rl/team_preview_obs.py (observación de Team Preview).
// Regenerarlo solo al cambiar el commit fijado de Showdown, y registrar ese commit.
'use strict';

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const psRoot = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', 'pokemon-showdown'));
const outPath = path.resolve(process.argv[3] || path.join(__dirname, '..', 'API', 'JSONs', 'champions_dex_snapshot.json'));
const { Dex } = require(path.join(psRoot, 'dist', 'sim'));
const dex = Dex.mod('champions');

function pickBoosts(b) {
  if (!b) return undefined;
  const out = {};
  for (const [k, v] of Object.entries(b)) if (v) out[k] = v;
  return Object.keys(out).length ? out : undefined;
}

function secondaryList(m) {
  const raw = m.secondaries || (m.secondary ? [m.secondary] : []);
  return raw.map(s => ({
    chance: s.chance || 100,
    boosts: pickBoosts(s.boosts),
    status: s.status || undefined,
    volatileStatus: s.volatileStatus || undefined,
    selfBoosts: s.self ? pickBoosts(s.self.boosts) : undefined,
  }));
}

const moves = {};
for (const m of dex.moves.all()) {
  if (m.isZ || m.isMax) continue;
  moves[m.id] = {
    name: m.name,
    type: m.type,
    category: m.category,
    basePower: m.basePower,
    accuracy: m.accuracy === true ? 101 : m.accuracy,
    priority: m.priority,
    target: m.target,
    flags: Object.keys(m.flags || {}).filter(k => m.flags[k]),
    multihit: m.multihit || undefined,
    critRatio: m.critRatio || undefined,
    willCrit: m.willCrit || undefined,
    drain: m.drain || undefined,
    recoil: m.recoil || undefined,
    heal: m.heal || undefined,
    selfSwitch: m.selfSwitch || undefined,
    stallingMove: m.stallingMove || undefined,
    status: m.status || undefined,
    volatileStatus: m.volatileStatus || undefined,
    sideCondition: m.sideCondition || undefined,
    pseudoWeather: m.pseudoWeather || undefined,
    weather: m.weather || undefined,
    terrain: m.terrain || undefined,
    boosts: pickBoosts(m.boosts),
    selfBoosts: pickBoosts((m.self && m.self.boosts) || (m.selfBoost && m.selfBoost.boosts)),
    secondaries: secondaryList(m),
    overrideOffensiveStat: m.overrideOffensiveStat || undefined,
    overrideOffensivePokemon: m.overrideOffensivePokemon || undefined,
    overrideDefensiveStat: m.overrideDefensiveStat || undefined,
    damage: m.damage || undefined,
    ohko: m.ohko || undefined,
    forceSwitch: m.forceSwitch || undefined,
    breaksProtect: m.breaksProtect || undefined,
    isNonstandard: m.isNonstandard || undefined,
  };
}

const species = {};
for (const s of dex.species.all()) {
  species[s.id] = {
    name: s.name,
    types: s.types,
    baseStats: s.baseStats,
    abilities: s.abilities,
    isMega: !!s.isMega || undefined,
    baseSpecies: s.baseSpecies,
    requiredItem: s.requiredItem || undefined,
  };
}

const items = {};
for (const it of dex.items.all()) {
  items[it.id] = {
    name: it.name,
    megaStone: it.megaStone || undefined,
    isBerry: it.isBerry || undefined,
    naturalGift: it.naturalGift ? it.naturalGift.type : undefined,
    isNonstandard: it.isNonstandard || undefined,
  };
}

const abilities = {};
for (const a of dex.abilities.all()) {
  abilities[a.id] = { name: a.name, isNonstandard: a.isNonstandard || undefined };
}

let commit = 'unknown';
try {
  commit = execSync('git rev-parse --short HEAD', { cwd: psRoot }).toString().trim();
} catch (e) { /* sin git */ }

const out = {
  meta: { source: 'pokemon-showdown Dex.mod("champions")', ps_commit: commit },
  moves, species, items, abilities,
};
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(`OK ${outPath} moves=${Object.keys(moves).length} species=${Object.keys(species).length} ` +
            `items=${Object.keys(items).length} abilities=${Object.keys(abilities).length} ps=${commit}`);
