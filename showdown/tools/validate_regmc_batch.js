#!/usr/bin/env node
/**
 * Valida equipos contra un formato de Pokémon Showdown (Reg M-C por defecto).
 *
 * stdin:  JSON [{name, text}, ...]   (text = export Showdown del equipo)
 * argv:   [formatId]                 (default gen9championsvgc2026regmcbo3)
 * stdout: JSON [{name, ok, problems: [...]}, ...]
 *
 * Usa el TeamValidator compilado de pokemon-showdown (dist/sim). La raíz se toma
 * de POKEMON_SHOWDOWN_ROOT o del directorio de trabajo (el llamador ejecuta con
 * cwd = pokemon-showdown).
 */
'use strict';

const path = require('path');

const psRoot = process.env.POKEMON_SHOWDOWN_ROOT || process.cwd();
const formatId = process.argv[2] || 'gen9championsvgc2026regmcbo3';

function fail(message) {
  process.stderr.write(`[validate_regmc_batch] ${message}\n`);
  process.exit(2);
}

let sim;
try {
  sim = require(path.join(psRoot, 'dist', 'sim'));
} catch (err) {
  fail(`no se pudo cargar ${path.join(psRoot, 'dist', 'sim')}: ${err && err.message}`);
}
const {Teams, TeamValidator} = sim;
if (!Teams || !TeamValidator) fail('dist/sim no exporta Teams/TeamValidator');

let validator;
try {
  validator = TeamValidator.get(formatId);
} catch (err) {
  fail(`formato inválido ${formatId}: ${err && err.message}`);
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  let rows;
  try {
    rows = JSON.parse(input || '[]');
  } catch (err) {
    fail(`JSON de entrada inválido: ${err && err.message}`);
  }
  if (!Array.isArray(rows)) fail('la entrada debe ser una lista');

  const out = rows.map((row) => {
    const name = row && row.name;
    try {
      const team = Teams.import(String((row && row.text) || ''));
      if (!team || !team.length) {
        return {name, ok: false, problems: ['equipo vacío o no se pudo leer el export']};
      }
      const problems = validator.validateTeam(team) || [];
      return {name, ok: problems.length === 0, problems};
    } catch (err) {
      return {name, ok: false, problems: [`error al validar: ${err && err.message}`]};
    }
  });
  process.stdout.write(JSON.stringify(out));
});
