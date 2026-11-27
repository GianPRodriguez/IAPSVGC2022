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
| 2026-11-27 04:36 | Sam smithy | PPO v2 | 0–2 | Sol + Tailwind (ver análisis) |

## 2026-11-27: PPO v2 contra adojs (0–2)

**Rival:** lluvia con Trick Room: Pelipper (Drizzle, Focus Sash), Basculegion (Swift Swim, Wave Crash,
Last Respects, Flip Turn), Farigiraf (Armor Tail, Trick Room, Psychic), Archaludon (Stamina, Electro Shot) y
Golisopod/Sneasler (no salieron). Ganó las dos partidas sin perder ningún Pokémon en la primera.

**Datos técnicos:** 25 decisiones, 0 excepciones, 0 órdenes de respaldo. El bot aceptó en 28 s y la serie
duró 5,3 min. El fallo es de juego, no de infraestructura.

**Hojas abiertas (corrección del 27-11-2026):** la serie se jugó **con hojas abiertas**, según la constancia que
envió el rival. El primer diagnóstico decía "cerradas" porque los replays no tenían `|showteam|`, pero poke-env
procesa esas líneas sin pasarlas a `parse_message` y nunca las escribe en el replay guardado. Desde ese día
`bo3_web.py` las añade al replay y marca `hojas_abiertas` en el JSON de cada partida.

### Partida 1 (4 turnos, 0–4 en Pokémon)

Preview: Gholdengo + Staraptor delante, Rillaboom + Arcanine detrás.

1. Staraptor megaevoluciona y usa **Protect**. Gholdengo no protege y cae de un golpe antes de moverse:
   Helping Hand + Wave Crash de Basculegion, que con Swift Swim es más rápido que Gholdengo (aun con el
   −1 de Intimidate).
2. Rillaboom elige Grassy Glide contra **Basculegion** (68 %), pero el rival **cambia a Farigiraf** en ese
   hueco y su Armor Tail bloquea la prioridad. Fue una buena lectura del humano, no un error de información:
   al decidir, Farigiraf no estaba en el campo.
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
   Swift Swim y Helping Hand. Bajo Trick Room, Staraptor y Arcanine caen sin moverse. *Corrección (27-11-2026):*
   la observación **sí** tiene el orden de velocidad efectivo (matrices de velocidad con Tailwind, Swift Swim,
   Chlorophyll y similares, e inversión por Trick Room; `observation.effective_speed` / `speed_relation`) y marca
   los movimientos rivales de Trick Room y Tailwind. La información estaba; falla la respuesta de la política.
4. ~~Habilidades del rival que bloquean~~ (retirado): el Grassy Glide contra Armor Tail fue un cambio del rival,
   y con hojas abiertas la habilidad se conoce (la observación marca la prioridad bloqueada por Armor Tail).
5. **Protect consecutivo** (partida 2, turno 8): es legal en Showdown, pero falla casi siempre. Con el combate
   muestreado a T = 0,5 puede ser el sorteo. Las máscaras no lo penalizan.
6. **Arquetipo fuera de la distribución de entrenamiento.** Lluvia + Trick Room + Last Respects no es el estilo
   de los heurísticos ni del pool. Es el mismo diagnóstico que el 0–2 de PPO v1 contra una persona
   (memorizacion_equipos §16).

## 2026-11-27: PPO v2 contra Sam smithy (0–2)

**Rival:** sol con Tailwind: Whimsicott (Prankster, Tailwind, Moonblast, Focus Sash), Kingambit (Defiant, Sucker
Punch, Kowtow Cleave), Charizard-Mega-Y (Drought, Heat Wave, Weather Ball) y Garchomp (Rough Skin, Earthquake,
Dragon Claw, Sitrus Berry). Floette-Eternal y Basculegion no salieron. Abrió con Whimsicott + Kingambit en las
dos partidas.

**Datos técnicos:** 42 decisiones, 0 excepciones, 0 órdenes de respaldo. Tardó 273 s en retar al bot y la serie
duró 6,2 min. No consta si hubo hojas abiertas: la serie es anterior al registro de `|showteam|`.

### Partida 1 (8 turnos, 2–4 en Pokémon)

Preview: Gholdengo + Staraptor delante, Rillaboom + Arcanine detrás.

1. **Intimidate de Staraptor contra Kingambit con Defiant**: Kingambit queda con +1 neto. Gholdengo usa
   Protect y Staraptor (Mega) cae ante Sucker Punch + Moonblast antes de moverse.
2. Rillaboom usa Fake Out contra Charizard (bien) y Gholdengo Make It Rain (Whimsicott protege).
3. Charizard-Mega-Y pone sol y Heat Wave le hace KO a Rillaboom. Make It Rain rompe el Focus Sash de
   Whimsicott (1 %).
4. Arcanine usa **Head Smash contra Garchomp**, que resiste Roca (64 %) y recibe Rough Skin, con lo que Arcanine
   pierde el Focus Sash. Gholdengo usa Shadow Ball contra Whimsicott, que protege. Extreme Speed contra
   Whimsicott (12 %) no se podía usar ese turno por el Protect, pero en el turno 5 sí.
5. Earthquake de Garchomp: KO a Arcanine (Fuego/Roca, débil ×4) y a su propio Whimsicott.
6–8. Gholdengo, solo contra Garchomp + Kingambit: **Protect tres turnos seguidos**. El primero tiene sentido
   (agota el último turno de Tailwind), el segundo sale por suerte (1/3) y el tercero falla. Cae con
   Kowtow Cleave.

### Partida 2 (5 turnos, 2–4 en Pokémon)

Preview: Staraptor + Primarina delante, Arcanine + Gholdengo detrás.

1. El mejor turno de la serie: **Close Combat de Staraptor le hace KO a Kingambit** (Defiant le había subido el
   ataque, pero Close Combat es ×4) y Moonblast de Primarina (crítico) deja a Whimsicott al 1 %.
2. Heat Wave en sol le hace KO a Staraptor (2/185). Whimsicott pone Tailwind y Primarina lo remata.
3. Earthquake: **KO a Arcanine** (×4) sin que actúe, por ir más lento bajo Tailwind. Moonblast deja a
   Garchomp al 18 %, que se cura con la Sitrus Berry.
4. Doble Protect (Gholdengo, recién entrado, y Primarina): agota un turno de Tailwind, pero no progresa.
5. Weather Ball en sol le hace KO a Gholdengo y Earthquake a Primarina.

## Patrones en las 4 partidas contra personas (2 series, 0–4)

| Partida | Trae | Abre |
|---|---|---|
| adojs 1 | Gholdengo, Staraptor, Rillaboom, Arcanine | Gholdengo + Staraptor |
| adojs 2 | Gholdengo, Staraptor, Primarina, Arcanine | Gholdengo + Staraptor |
| Sam smithy 1 | Gholdengo, Staraptor, Rillaboom, Arcanine | Gholdengo + Staraptor |
| Sam smithy 2 | Gholdengo, Staraptor, Primarina, Arcanine | Staraptor + Primarina |

1. **El preview casi no depende del rival.** Raichu no entró en ninguna de las 4 partidas, y Gholdengo,
   Staraptor y Arcanine-Hisui entraron en las 4. Las 4 elecciones son casi iguales aun con sorteo T = 1,
   contra lluvia y contra sol con Garchomp. Arcanine-Hisui (débil ×4 a Agua y Tierra) murió pronto en las 4:
   dos veces por Earthquake, una por Weather Ball en lluvia y otra por Psychic + Weather Ball. **Es el hallazgo
   más fuerte:** conviene medir la distribución del preview v2.1b (¿está casi colapsada?, ¿cambia con el equipo
   rival?) antes de pensar en memoria entre partidas.
2. **Protect consecutivo, de nuevo:** Gholdengo tres seguidos (`protect_counter` ya está en la observación). Con
   dos casos en dos series pasa de anécdota a patrón. Se arregla en la máscara.
3. **El control de velocidad del rival decide las dos series** (Trick Room y Tailwind) y el agente no responde:
   Staraptor no ha usado Tailwind en ninguna de las 4 partidas.
4. **Interacciones de habilidades rivales:** Intimidate contra Defiant (Kingambit) en las dos partidas contra Sam
   smithy. Con hojas abiertas el agente conocía Defiant.
5. **Elección de objetivos:** Head Smash contra un Garchomp que resiste Roca y tiene Rough Skin, con lo que
   Arcanine pierde su Focus Sash.

### Memoria entre partidas del Bo3 (idea del usuario, 27-11-2026)

Las dos partidas son contra el mismo equipo, y el agente empezó la segunda igual que la primera (misma
formación, Mega Staraptor + Protect). Con hojas abiertas, lo que la partida 1 aporta de verdad es **qué 4 trajo
el rival, quién abrió y cómo jugó** (aquí: lluvia con Basculegion delante y luego Trick Room). Lo que no aporta
mucho son los SP: el oráculo de SP exactos solo da unos +1 punto a los agentes greedy (sp-information, 14-11-2026),
y el motor de inferencia actual no acota nada dentro de una partida. Prioridad propuesta: preview condicionado
a lo que el rival trajo en la partida 1, luego patrones de juego (probabilísticos, porque el humano también se
adapta: adojs cambió el compañero de apertura). Pendiente de pre-registro.

**Seguimiento (27-11-2026):** el usuario decidió llevar estas ideas a PPO v3 y DDQN v2 (los dos con los mismos
cambios): máscara del Protect consecutivo, memoria de serie BO3, entrenamiento en series y preview condicionado
como prueba aparte. Pre-registro y medición del preview (M1) en `protocolo_evaluacion.md` §4.16. Lo que se pudo
comprobar con los replays: con `protect_counter` en 1 y 2 en la observación, Gholdengo volvió a usar Protect, así
que no faltaba información, sino que la política no aprendió a evitarlo.

### Ideas que salen de aquí (sin evaluar)

- Penalizar o enmascarar el Protect consecutivo cuando la probabilidad de éxito es ≤ 1/3.
- ~~Añadir a la observación el orden de velocidad efectivo~~ (retirada: ya está en la observación).
- ~~Incluir arquetipos de lluvia y Trick Room en el pool~~: ya están entre los equipos (`train`: Trick Room 24 %,
  Tailwind 68 %, lluvia 17 %, sol 12 %, Defiant/Competitive 38 %). Lo que falta medir es si los rivales
  heurísticos *ejecutan* esos planes.
- Medir la distribución del preview v2.1b contra estos equipos: entropía, frecuencia de cada Pokémon y si la
  elección cambia con el rival (Raichu 0/4, Arcanine 4/4).
