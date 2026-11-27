# Partidas de la web de retos BO3 contra personas

Las series se juegan en la web pública (`scripts/online/bo3_web.py`, EC2 del Learner Lab,
https://teamofabrexuxa.duckdns.org) con la cuenta `teamofabrexuxa`. Los datos crudos (un JSON por serie y los
replays HTML) están en `logs/online_web/` y se traen con `scripts/online/aws/collect_battles.sh`. La tabla
generada es `logs/online_web/resumen.md`.

Con pocas series no hay conclusión estadística (protocolo de evaluación: la conclusión de un agente sale de la
evaluación contra el set `eval`). Sirven como diagnóstico cualitativo, para ver qué se hace mal contra personas
y contra arquetipos que no aparecen en el entrenamiento.

Equipo del agente (fijo, `showdown/rl/data/ddqn_team.txt`): Raichu-Mega-Y (Lightning Rod), Rillaboom,
Gholdengo, Arcanine-Hisui (Focus Sash), Staraptor-Mega y Primarina.

## Registro

| Fecha (UTC) | Rival | Agente | Resultado | Nota |
|---|---|---|---|---|
| 2026-11-27 04:17 | Deep1ODDQN | PPO v2 | 2–0 | **Prueba técnica**: jugador aleatorio de poke-env, no una persona |
| 2026-11-27 04:25 | adojs | PPO v2 | 0–2 | Primera serie real (ver análisis) |

## 2026-11-27: PPO v2 contra adojs (0–2)

**Rival:** lluvia con Trick Room: Pelipper (Drizzle, Focus Sash), Basculegion (Swift Swim, Wave Crash,
Last Respects, Flip Turn), Farigiraf (Armor Tail, Trick Room, Psychic), Archaludon (Stamina, Electro Shot) y
Golisopod/Sneasler (no salieron). Ganó las dos partidas sin perder ningún Pokémon en la primera.

**Datos técnicos:** 25 decisiones, 0 excepciones, 0 órdenes de respaldo. El bot aceptó en 28 s y la serie
duró 5,3 min. El fallo es de juego, no de infraestructura.

### Partida 1 (4 turnos, 0–4 en Pokémon)

Preview: Gholdengo + Staraptor delante, Rillaboom + Arcanine detrás.

1. Staraptor megaevoluciona y usa **Protect**. Gholdengo no protege y cae de un golpe antes de moverse:
   Helping Hand + Wave Crash de Basculegion, que con Swift Swim es más rápido que Gholdengo (aun con el
   −1 de Intimidate).
2. Rillaboom usa **Grassy Glide contra Farigiraf**: Armor Tail bloquea la prioridad y el turno se pierde.
   Pelipper le hace KO con Hurricane, que en lluvia no falla. Staraptor usa Brave Bird contra Farigiraf
   (53 %, que se cura con la Sitrus Berry).
3. Arcanine rompe el Focus Sash de Pelipper con Head Smash y Staraptor repite Brave Bird contra Farigiraf.
   Pelipper le hace KO a Staraptor con Hurricane y Farigiraf activa Trick Room.
4. Arcanine cae con Psychic + Weather Ball.

### Partida 2 (8 turnos, 0–4 en Pokémon)

Preview: Gholdengo + Staraptor delante otra vez, Arcanine + Primarina detrás.

1. Otra vez Mega Staraptor + **Protect**. Flip Turn de Basculegion mete a Archaludon, Gholdengo usa
   Shadow Ball contra él y Farigiraf activa **Trick Room** sin oposición.
2. Con Trick Room, Staraptor cae ante Psychic + Dragon Pulse antes de moverse.
3. Doble Protect (Gholdengo y Arcanine). Farigiraf pone lluvia y el Electro Shot de Archaludon choca con
   Protect.
4. Arcanine cae sin haber atacado en toda la partida. Gholdengo le hace KO a Archaludon.
5. Primarina protege (bien leído: Psychic iba a por ella). Pelipper pone Tailwind.
6. Termina Trick Room. Weather Ball le hace KO a Gholdengo y Hyper Voice de Primarina le hace KO a Pelipper.
7. Primarina protege frente a Last Respects (200 de potencia con 3 caídos): bien.
8. Primarina **vuelve a usar Protect**, falla (Protect consecutivo) y cae con Last Respects.

### Patrones

1. **Preview: Raichu se queda fuera las dos veces.** Raichu-Mega-Y con Lightning Rod absorbe Electro Shot y
   Zap Cannon castiga a Pelipper, Basculegion y Golisopod. Contra lluvia era la pieza clave. El preview juega
   con sorteo T = 1 contra personas y las dos veces salió la misma formación sin Raichu.
2. **Mega Staraptor + Protect en el turno 1, las dos veces.** Megaevolucionar protegido es habitual, pero aquí
   regala el turno: el compañero queda solo frente a la amenaza (partida 1) o el rival activa Trick Room
   libremente (partida 2). Tailwind no se usó nunca.
3. **No se reconoce la velocidad bajo lluvia y Trick Room.** Gholdengo no protege frente a Basculegion con
   Swift Swim y Helping Hand. Bajo Trick Room, Staraptor y Arcanine caen sin moverse. La observación P5 usa la
   calculadora para el daño (con el clima), pero el orden de movimiento con Swift Swim o Trick Room no es
   explícito.
4. **Habilidades del rival que bloquean.** Grassy Glide contra Armor Tail. Farigiraf puede tener Armor Tail,
   Cud Chew o Sap Sipper, y la habilidad no se revela hasta que actúa. El agente no considera esa posibilidad.
5. **Protect consecutivo** (partida 2, turno 8): es legal en Showdown, pero falla casi siempre. Con el combate
   muestreado a T = 0,5 puede ser el sorteo. Las máscaras no lo penalizan.
6. **Arquetipo fuera de la distribución de entrenamiento.** Lluvia + Trick Room + Last Respects no es el estilo
   de los heurísticos ni del pool. Es el mismo diagnóstico que el 0–2 de PPO v1 contra una persona
   (memorizacion_equipos §16).

### Ideas que salen de aquí (sin evaluar)

- Penalizar o enmascarar el Protect consecutivo cuando la probabilidad de éxito es ≤ 1/3.
- Añadir a la observación el orden de velocidad efectivo (Trick Room, Tailwind, Swift Swim/Chlorophyll,
  parálisis) además del daño.
- Incluir arquetipos de lluvia y Trick Room en el pool de rivales del entrenamiento.
- Revisar si el preview determinista (o con temperatura menor) trae a Raichu contra equipos de lluvia.
