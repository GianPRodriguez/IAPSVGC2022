// Auditoría (P5 completo, 19-11-2026): habilidades, movimientos y objetos del dex Champions que cambian
// el daño, el tipo o la efectividad, según los manejadores que declaran en el simulador de Showdown.
//
// Uso (desde la raíz del repo, con ../pokemon-showdown compilado):
//   node scripts/analysis/audit_damage_effects.js [ruta_ps] [salida.json]
// Salida: API/JSONs/damage_effects_audit.json (lo lee scripts/analysis/audit_damage_effects_report.py).
'use strict';
const fs = require('fs');
const path = require('path');

const psRoot = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', '..', 'pokemon-showdown'));
const outPath = path.resolve(process.argv[3] || path.join(__dirname, '..', '..', 'API', 'JSONs', 'damage_effects_audit.json'));
const { Dex } = require(path.join(psRoot, 'dist', 'sim'));
const dex = Dex.mod('champions');

// Manejadores del simulador que afectan al daño de un ataque (propio, del aliado, del rival o de campo).
const HOOKS = [
  'onModifyType', 'onModifyMove', 'onBasePower', 'onAllyBasePower', 'onAnyBasePower', 'onFoeBasePower',
  'onSourceBasePower', 'basePowerCallback', 'damageCallback', 'onModifyAtk', 'onModifySpA', 'onModifyDef',
  'onModifySpD', 'onAllyModifyAtk', 'onAllyModifySpA', 'onAllyModifyDef', 'onAllyModifySpD',
  'onAnyModifyAtk', 'onAnyModifySpA', 'onAnyModifyDef', 'onAnyModifySpD', 'onSourceModifyAtk',
  'onSourceModifySpA', 'onSourceModifyDamage', 'onModifyDamage', 'onAnyModifyDamage', 'onAllyModifyDamage',
  'onEffectiveness', 'onAnyEffectiveness', 'onTryHit', 'onAllyTryHitSide', 'onImmunity', 'onModifyCritRatio',
  'onModifyAccuracy', 'onSourceModifyAccuracy', 'onAnyAccuracy', 'onModifySTAB', 'onModifyPriority',
  'onFractionalPriority', 'onModifySpe', 'onAnyRedirectTarget', 'onFoeRedirectTarget',
  'onWeather', 'onStart', 'onAnyModifyBoost', 'onSourceModifyMove',
];

function hooksOf(e) {
  const out = [];
  for (const h of HOOKS) if (typeof e[h] === 'function' || (h === 'onModifyMove' && e[h])) out.push(h);
  if (e.condition) for (const h of HOOKS) if (typeof e.condition[h] === 'function') out.push('condition.' + h);
  return out;
}

const legal = (e) => e.exists && !e.isNonstandard;
// Habilidades presentes en especies legales (incluidas las formas Mega).
const abilitiesInUse = new Set();
for (const s of dex.species.all()) {
  if (!legal(s)) continue;
  for (const a of Object.values(s.abilities || {})) if (a) abilitiesInUse.add(dex.abilities.get(a).id);
}

const abilities = [];
for (const a of dex.abilities.all()) {
  if (!a.exists || !abilitiesInUse.has(a.id)) continue;
  const h = hooksOf(a);
  if (h.length) abilities.push({ id: a.id, name: a.name, hooks: h, desc: a.shortDesc || a.desc || '' });
}
const moves = [];
for (const m of dex.moves.all()) {
  if (!legal(m) || m.isZ || m.isMax) continue;
  const h = hooksOf(m);
  if (m.category === 'Status' && !h.length) continue;
  const special = h.length || m.overrideOffensiveStat || m.overrideDefensiveStat || m.overrideOffensivePokemon
    || m.ignoreDefensive || m.ignoreImmunity || m.multihit || m.willCrit || m.selfdestruct || m.recoil
    || m.drain || m.weather || m.terrain || m.pseudoWeather || m.sideCondition;
  if (!special) continue;
  moves.push({
    id: m.id, name: m.name, type: m.type, category: m.category, basePower: m.basePower, hooks: h,
    override: [m.overrideOffensiveStat, m.overrideDefensiveStat, m.overrideOffensivePokemon].filter(Boolean),
    multihit: m.multihit || null, desc: m.shortDesc || m.desc || '',
  });
}
const items = [];
for (const it of dex.items.all()) {
  if (!legal(it)) continue;
  const h = hooksOf(it);
  if (h.length) items.push({ id: it.id, name: it.name, hooks: h, desc: it.shortDesc || it.desc || '' });
}
const conditions = [];
for (const id of ['sunnyday', 'raindance', 'sandstorm', 'snowscape', 'electricterrain', 'grassyterrain',
  'psychicterrain', 'mistyterrain', 'reflect', 'lightscreen', 'auroraveil', 'helpinghand', 'charge',
  'tailwind', 'trickroom', 'gravity', 'friendguard']) {
  const c = dex.conditions.get(id);
  if (c && c.exists) conditions.push({ id, hooks: hooksOf(c) });
}
fs.writeFileSync(outPath, JSON.stringify({ abilities, moves, items, conditions }, null, 1));
console.log(`habilidades ${abilities.length} | movimientos ${moves.length} | objetos ${items.length} -> ${outPath}`);
