/**
 * Genera API/JSONs/champions_move_meta.json a partir de @smogon/calc.
 *
 * Solo se necesita la metadata que interviene en el orden de accion:
 *   p = prioridad del movimiento (0 si es normal)
 *   c = categoria abreviada (p = Physical, s = Special, t = Status)
 *
 * La categoria hace falta porque Prankster solo sube la prioridad de los
 * movimientos de estado.
 *
 * Uso (el paquete @smogon/calc vive en el repo damage-calc):
 *   node tools/gen_move_meta.js <ruta-a-damage-calc> <ruta-de-salida>
 */
'use strict';

const fs = require('fs');
const path = require('path');

const calcRepo = process.argv[2];
const outPath = process.argv[3];

if (!calcRepo || !outPath) {
  console.error('Uso: node tools/gen_move_meta.js <ruta-damage-calc> <salida.json>');
  process.exit(1);
}

const { Generations } = require(path.join(calcRepo, 'node_modules', '@smogon', 'calc'));
const gen = Generations.get(9);

const CATEGORY = { Physical: 'p', Special: 's', Status: 't' };

const meta = {};
for (const move of gen.moves) {
  const id = move.name.toLowerCase().replace(/[^a-z0-9]/g, '');
  // ?? y no || : prioridades negativas (Trick Room -7) deben preservarse.
  meta[id] = {
    p: move.priority ?? 0,
    c: CATEGORY[move.category] || 't',
  };
}

// @smogon/calc a veces omite prioridades negativas; alinear con PS Gen 9.
const PS_PRIORITY_OVERRIDES = {
  trickroom: -7,
  roar: -6,
  whirlwind: -6,
  teleport: -6,
  circlethrow: -6,
  dragontail: -6,
  counter: -5,
  mirrorcoat: -5,
  revenge: -4,
  avalanche: -4,
  focuspunch: -3,
  beakblast: -3,
  shelltrap: -3,
  vitalthrow: -1,
};
for (const [id, priority] of Object.entries(PS_PRIORITY_OVERRIDES)) {
  if (meta[id]) meta[id].p = priority;
}

fs.writeFileSync(outPath, JSON.stringify(meta, null, 1));

const prioritized = Object.values(meta).filter((m) => m.p !== 0).length;
console.log(`Movimientos: ${Object.keys(meta).length} (con prioridad != 0: ${prioritized})`);
