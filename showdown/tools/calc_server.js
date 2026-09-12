#!/usr/bin/env node
/**
 * Servidor TCP compartido para N agentes Python.
 *
 *   DAMAGE_CALC_HOST=127.0.0.1 DAMAGE_CALC_PORT=8765 node calc_server.js
 *
 * Protocolo: una linea JSON por request/response (igual que el worker stdio).
 * Un unico proceso Node atiende a todos los agentes -> ~1 proceso en vez de 50+.
 */
'use strict';

const net = require('net');
const {handleMessage, CALC_ROOT} = require('./calc_core');

const HOST = process.env.DAMAGE_CALC_HOST || '127.0.0.1';
const PORT = parseInt(process.env.DAMAGE_CALC_PORT || '8765', 10);

const server = net.createServer((socket) => {
  socket.setEncoding('utf8');
  let buf = '';
  socket.on('data', (chunk) => {
    buf += chunk;
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      let out;
      try {
        out = handleMessage(JSON.parse(line));
      } catch (err) {
        out = {ok: false, error: String(err && err.message ? err.message : err)};
      }
      try {
        socket.write(JSON.stringify(out) + '\n');
      } catch (_) {
        // cliente desconectado
      }
    }
  });
  socket.on('error', () => {});
});

server.on('error', (err) => {
  if (err && err.code === 'EADDRINUSE') {
    process.stderr.write(`[calc_server] puerto ${PORT} ya en uso (otro server activo)\n`);
    process.exit(0);
  }
  process.stderr.write(`[calc_server] error: ${err}\n`);
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  process.stderr.write(`[calc_server] listening ${HOST}:${PORT} calc=${CALC_ROOT}\n`);
});
