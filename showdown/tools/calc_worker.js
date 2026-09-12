#!/usr/bin/env node
/**
 * Worker stdio: una linea JSON -> una linea JSON.
 * Soporta cmd ping | batch | calculo simple.
 */
'use strict';

const {handleMessage} = require('./calc_core');
const readline = require('readline');
const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let out;
  try {
    out = handleMessage(JSON.parse(trimmed));
  } catch (err) {
    out = {ok: false, error: String(err && err.message ? err.message : err)};
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
