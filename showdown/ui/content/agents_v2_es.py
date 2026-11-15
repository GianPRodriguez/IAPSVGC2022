"""Textos educativos v2 (español): los 11 agentes A–K como pipelines preview → combate.

Solo información básica para usuarios que quieren entender cómo funciona cada
agente. No cambia qué agentes se pueden ejecutar en Jugar / Laboratorio (eso
sigue en ``agents_es.AGENTS_V1``).

Fuente de verdad de los pares: ``showdown/inference_sim/conclusion_eval/pipelines.py``
y ``docs/SPECS_INDEX.md`` («Pipelines familiares»). Detalle técnico: specs en ``docs/``.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Familias
# ---------------------------------------------------------------------------

FAMILIES: Dict[str, Dict[str, str]] = {
    'heuristica': {
        'nombre': 'Heurísticos',
        'resumen': (
            'Reglas diseñadas a mano. No entrenan: su calidad depende de lo bien '
            'pensadas que estén las reglas. Cada uno trae su propio Team Preview.'
        ),
    },
    'aprendizaje': {
        'nombre': 'Aprendizaje por refuerzo',
        'resumen': (
            'Redes neuronales que aprenden jugando miles de partidas. Lo aprendido '
            'queda guardado en pesos; al jugar, deciden rápido.'
        ),
    },
    'planificacion': {
        'nombre': 'Planificación',
        'resumen': (
            'No guardan experiencia en pesos de combate: en cada turno imaginan '
            'muchas continuaciones y eligen la que mejor resiste. Piensan más por turno.'
        ),
    },
    'diversidad': {
        'nombre': 'Calidad–Diversidad (QD)',
        'resumen': (
            'En vez de un único campeón, entrenan un «museo» de estilos distintos y '
            'guardan el mejor de cada estilo. Al jugar se usa el mejor del museo.'
        ),
    },
}

# ---------------------------------------------------------------------------
# Team Preview (elegir 4 de 6 y los 2 que salen primero)
# ---------------------------------------------------------------------------

PREVIEWS: Dict[str, Dict[str, str]] = {
    'random': {
        'nombre': 'Preview aleatorio',
        'como': 'Elige 4 de los 6 Pokémon al azar, sin mirar tipos ni roles.',
        'entrena': 'No',
    },
    'greedy_solo': {
        'nombre': 'Preview por tipos (greedy)',
        'como': (
            'Prueba cada combinación de 4 y cada pareja de leads, y se queda con la '
            'que mejor cubre al rival en la tabla de tipos y mejor resiste sus '
            'ataques (con un pequeño desempate por estadísticas base).'
        ),
        'entrena': 'No',
    },
    'expectimax_bayesian': {
        # Histórico (≤ v1.6). Desde v1.7 Expectimax usa el preview por tipos (greedy_solo).
        'nombre': 'Preview por tipos + roles (descartado)',
        'como': (
            'Versión anterior del preview de Expectimax: además de los tipos premiaba '
            'roles de campo (Tailwind, Trick Room, Fake Out, redirección) y la calidad '
            'de los leads. Se descartó porque en las pruebas elegía peores equipos '
            '(−3,8 puntos frente al preview por tipos).'
        ),
        'entrena': 'No',
    },
    'team_preview_ppo': {
        'nombre': 'Preview Masked PPO',
        'como': (
            'Una red mira las dos hojas de equipo abiertas (OTS) y elige una de las '
            '360 formaciones posibles. Aprende por refuerzo: si la partida se gana, '
            'esa elección se vuelve más probable.'
        ),
        'entrena': 'Sí (refuerzo, PPO)',
    },
    'team_preview_nash': {
        'nombre': 'Preview Payoff Nash',
        'como': (
            'Una red estima, para cada formación nuestra contra cada formación '
            'rival, qué tan favorable es el cruce. Luego un solver busca una mezcla '
            'de formaciones difícil de explotar (equilibrio de Nash), en vez de '
            'repetir siempre la misma.'
        ),
        'entrena': 'Sí (red supervisada + solver)',
    },
    'team_preview_qd_pga': {
        'nombre': 'Preview QD-PGA',
        'como': (
            'Entrena un archivo de 100 «salas», cada una con la mejor política de '
            'preview para un estilo distinto. Las nuevas variantes salen de mutar '
            'una política o de mejorarla con PPO.'
        ),
        'entrena': 'Sí (calidad–diversidad)',
    },
}

# ---------------------------------------------------------------------------
# Los 11 agentes (letra → pipeline)
# ---------------------------------------------------------------------------

AGENT_LETTERS: List[str] = list('ABCDEFGHIJK')

AGENTS_V2: Dict[str, Dict[str, object]] = {
    'A': {
        'id': 'random',
        'nombre': 'Aleatorio',
        'familia': 'heuristica',
        'preview': 'random',
        'combate': 'Aleatorio',
        'subtitulo': 'Como tirar un dado entre jugadas legales',
        'analogia': (
            'En ajedrez: listas solo las jugadas permitidas y eliges una al azar. '
            'No miras si ganas material ni si dejas al rey expuesto.'
        ),
        'como_juega': (
            'Cada turno consulta qué movimientos y cambios permite Showdown y sortea '
            'uno. No aprende ni mejora con el tiempo.'
        ),
        'entrena': 'No',
        'velocidad': 'Instantáneo',
        'fuerte': 'Sirve de piso: si un agente no le gana con claridad, algo falla.',
        'debil': 'No tiene ningún plan.',
        'nota_tesis': 'Línea base uniforme sobre acciones legales; sin modelo de rival.',
        'spec': 'docs/random_spec.md',
    },
    'B': {
        'id': 'greedy_solo',
        'nombre': 'Greedy por Pokémon',
        'familia': 'heuristica',
        'preview': 'greedy_solo',
        'combate': 'Greedy solo',
        'subtitulo': 'Capturar lo más valioso… mirando una sola pieza',
        'analogia': (
            'En ajedrez: con la pieza que estás mirando, capturas lo que más puntos '
            'dé ahora, sin pensar en el resto del tablero.'
        ),
        'como_juega': (
            'Cada Pokémon activo elige por separado el ataque que, según un cálculo '
            'rápido de daño, más perjudica al rival en ese turno.'
        ),
        'entrena': 'No',
        'velocidad': 'Muy rápido',
        'fuerte': 'Agresivo y simple; castiga errores obvios.',
        'debil': 'No coordina a sus dos Pokémon ni piensa en turnos futuros.',
        'nota_tesis': 'Heurística de daño por slot (factored); greedy T=1.',
        'spec': 'docs/greedy_solo_spec.md',
    },
    'C': {
        'id': 'greedy_joint',
        'nombre': 'Greedy en pareja',
        'familia': 'heuristica',
        'preview': 'greedy_solo',
        'combate': 'Greedy joint',
        'subtitulo': 'La mejor jugada de dos piezas a la vez',
        'analogia': (
            'En ajedrez: miras el alfil y el caballo juntos y eliges la pareja de '
            'jugadas que más puntos da ahora.'
        ),
        'como_juega': (
            'Prueba combinaciones de lo que hacen sus dos Pokémon activos y se queda '
            'con la que más daño y utilidad suma en este turno. Comparte el preview '
            'por tipos del agente B.'
        ),
        'entrena': 'No',
        'velocidad': 'Rápido',
        'fuerte': 'Coordina mejor que B: foco en un objetivo, menos fuego amigo.',
        'debil': 'Sigue siendo «corto de vista»: solo mira el turno actual.',
        'nota_tesis': 'Argmax sobre pares de acciones legales (sinergia 2v2).',
        'spec': 'docs/greedy_joint_spec.md',
    },
    'D': {
        'id': 'expectimax_bayesian',
        'nombre': 'Expectimax',
        'familia': 'heuristica',
        'preview': 'greedy_solo',
        'combate': 'Expectimax bayesiano',
        'subtitulo': 'Revisar las mejores ideas bajo incertidumbre',
        'analogia': (
            'En ajedrez: un compañero propone sus mejores jugadas y tú compruebas '
            'cuál sigue siendo buena si el rival responde de una u otra forma.'
        ),
        'como_juega': (
            'Toma las mejores parejas de jugadas del greedy en pareja y las '
            're-evalúa promediando el azar (rangos de daño, precisión), los sets '
            'posibles del rival y cómo podría responder. Comparte el preview por '
            'tipos de los agentes B y C, así que su diferencia está solo en el combate.'
        ),
        'entrena': 'No (solo se ajustan sus parámetros)',
        'velocidad': 'Lento',
        'fuerte': 'Más prudente: considera protecciones, velocidad y la suerte.',
        'debil': 'Sigue mirando un solo turno; cuesta más tiempo de cálculo.',
        'nota_tesis': 'Expectimax 1-ply anclado a Top-K de Joint + creencias sobre sets.',
        'spec': 'docs/expectimax_bayesian_spec.md',
    },
    'E': {
        'id': 'ddqn',
        'nombre': 'DDQN',
        'familia': 'aprendizaje',
        'preview': 'team_preview_ppo',
        'combate': 'Double DQN',
        'subtitulo': 'Una tabla de valores aprendida jugando',
        'analogia': (
            'En ajedrez: un jugador que, tras miles de partidas, tiene una '
            'intuición de cuánto vale cada jugada en cada posición, y elige la de '
            'mayor valor.'
        ),
        'como_juega': (
            'Una red estima cuánto conviene cada orden conjunta de los dos Pokémon '
            '(576 combinaciones) y elige la mejor legal. Aprende reutilizando '
            'partidas pasadas guardadas en memoria; una segunda red «lenta» evita '
            'que sobrevalore jugadas.'
        ),
        'entrena': 'Sí (refuerzo, basado en valor)',
        'velocidad': 'Rápido al jugar',
        'fuerte': 'Aprovecha mucha experiencia vieja; decide al instante.',
        'debil': 'Depende de lo que vio entrenando; puede fallar en situaciones raras.',
        'nota_tesis': 'Q-learning off-policy con replay, red online/target, salida joint 576.',
        'spec': 'docs/ddqn_spec.md',
    },
    'F': {
        'id': 'ppo',
        'nombre': 'PPO',
        'familia': 'aprendizaje',
        'preview': 'team_preview_ppo',
        'combate': 'Masked PPO',
        'subtitulo': 'Una política que aprende qué jugar directamente',
        'analogia': (
            'En ajedrez: en vez de puntuar cada jugada, el jugador desarrolla un '
            'estilo — «en esta posición suelo jugar esto» — y lo corrige poco a poco.'
        ),
        'como_juega': (
            'La red decide primero la orden del Pokémon 1 y luego la del Pokémon 2 '
            'teniendo en cuenta la primera. Las jugadas ilegales se tapan antes de '
            'elegir. Aprende de partidas recientes y limita cuánto cambia en cada paso.'
        ),
        'entrena': 'Sí (refuerzo, basado en política)',
        'velocidad': 'Rápido al jugar',
        'fuerte': 'Aprendizaje estable; coordina las dos órdenes.',
        'debil': 'Necesita muchas partidas nuevas (no reutiliza las antiguas).',
        'nota_tesis': 'PPO on-policy auto-regresivo slot1→slot2 con IAM y GAE.',
        'spec': 'docs/ppo_spec.md',
    },
    'G': {
        'id': 'mcts_puct',
        'nombre': 'MCTS + PUCT',
        'familia': 'planificacion',
        'preview': 'team_preview_nash',
        'combate': 'MCTS + PUCT',
        'subtitulo': 'Calcular muchas variantes con el reloj en marcha',
        'analogia': (
            'En ajedrez: antes de mover juegas mentalmente muchas variantes; cuando '
            'se acaba el tiempo, eliges la línea que más veces salió bien en tu análisis.'
        ),
        'como_juega': (
            'En cada turno simula muchas continuaciones, reparte la búsqueda entre '
            'probar jugadas nuevas y profundizar en las prometedoras, y juega la más '
            'visitada. El rival se imagina eligiendo al azar.'
        ),
        'entrena': 'No (se fija el presupuesto de simulaciones)',
        'velocidad': 'Lento (depende del número de simulaciones)',
        'fuerte': 'Mira varios turnos adelante; fácil de interpretar.',
        'debil': 'Más simulaciones = más tiempo; trata al rival como aleatorio.',
        'nota_tesis': 'MCTS open-loop con selección PUCT; sin gradientes en combate.',
        'spec': 'docs/mcts_puct_spec.md',
    },
    'H': {
        'id': 'mcts_puct_nash',
        'nombre': 'MCTS–Nash',
        'familia': 'planificacion',
        'preview': 'team_preview_nash',
        'combate': 'MCTS simultáneo + Nash',
        'subtitulo': 'Calcular variantes sabiendo que ambos mueven a la vez',
        'analogia': (
            'Como piedra-papel-tijera: no basta con «qué hago yo»; preguntas «si '
            'mezclo A y B y él responde bien, ¿quedo cubierto?».'
        ),
        'como_juega': (
            'Durante la búsqueda mantiene una tabla «mi jugada × su jugada» con '
            'las mejores opciones de cada lado, la rellena con simulaciones y la '
            'resume con un equilibrio aproximado, porque en VGC ambos eligen a la vez.'
        ),
        'entrena': 'No (se fija el presupuesto de simulaciones)',
        'velocidad': 'Lento',
        'fuerte': 'Modela al rival como alguien que también piensa.',
        'debil': 'Aún más costoso por simulación; solo considera las mejores opciones.',
        'nota_tesis': 'SM-MCTS con matrices Top-K por nodo y backup del valor de Nash/FP.',
        'spec': 'docs/mcts_puct_nash_spec.md',
    },
    'I': {
        'id': 'qd_ppo',
        'nombre': 'QD-PPO',
        'familia': 'diversidad',
        'preview': 'team_preview_qd_pga',
        'combate': 'QD-PPO',
        'subtitulo': 'Un museo de estilos, mejorados con PPO',
        'analogia': (
            'En ajedrez: un club que guarda al mejor jugador de cada estilo '
            '(agresivo, posicional, defensivo…) y entrena copias de ellos.'
        ),
        'como_juega': (
            'Mantiene 100 casillas de estilo. Toma un campeón, entrena una copia con '
            'PPO y la guarda si es mejor que el ocupante de su casilla. Al jugar usa '
            'el mejor de todo el archivo.'
        ),
        'entrena': 'Sí (calidad–diversidad + PPO)',
        'velocidad': 'Rápido al jugar',
        'fuerte': 'Explora muchas formas de ganar, no solo una.',
        'debil': 'Entrenar es costoso; al final solo despliega un campeón.',
        'nota_tesis': 'MAP-Elites 10×10 con PPO como operador de mejora.',
        'spec': 'docs/qd_ppo_spec.md',
    },
    'J': {
        'id': 'qd_pga',
        'nombre': 'QD-PGA',
        'familia': 'diversidad',
        'preview': 'team_preview_qd_pga',
        'combate': 'QD-PGA',
        'subtitulo': 'Museo de estilos: saltos al azar + mejora dirigida',
        'analogia': (
            'En ajedrez: a veces pruebas una apertura rara por curiosidad; otras, '
            'pules con estudio la que ya juegas.'
        ),
        'como_juega': (
            'Igual que QD-PPO, pero cada copia nueva sale, mitad y mitad, de una '
            'mutación aleatoria (exploración) o de una mejora con PPO. Todas '
            'comparten un mismo «crítico» que evalúa posiciones.'
        ),
        'entrena': 'Sí (calidad–diversidad híbrida)',
        'velocidad': 'Rápido al jugar',
        'fuerte': 'La mutación encuentra estilos que el gradiente no alcanza.',
        'debil': 'Las mutaciones a veces desperdician entrenamiento.',
        'nota_tesis': 'PGA-MAP-Elites: mutación gaussiana / PPO, crítico compartido.',
        'spec': 'docs/qd_pga_map_elites_spec.md',
    },
    'K': {
        'id': 'qd_ppga',
        'nombre': 'QD-PPGA',
        'familia': 'diversidad',
        'preview': 'team_preview_qd_pga',
        'combate': 'QD-PPGA',
        'subtitulo': 'Museo de estilos que busca llenar huecos',
        'analogia': (
            'En ajedrez: el club detecta qué estilos le faltan y pone a entrenar al '
            'jugador más parecido para cubrirlos.'
        ),
        'como_juega': (
            'Sin mutaciones: todas las copias se mejoran con PPO. La mitad de las '
            'veces elige como padre al campeón más cercano a una casilla vacía o '
            'débil, para cubrir estilos que faltan.'
        ),
        'entrena': 'Sí (calidad–diversidad, solo PPO)',
        'velocidad': 'Rápido al jugar',
        'fuerte': 'Mejora estable con presión explícita por cubrir estilos.',
        'debil': 'Sin saltos aleatorios, puede explorar menos lejos.',
        'nota_tesis': 'PPGA: emits solo PPO, padres por arborescencia hacia la frontera.',
        'spec': 'docs/qd_ppga_spec.md',
    },
}


def agent_title(letter: str) -> str:
    """Etiqueta corta, p. ej. «E · DDQN»."""
    return f'{letter} · {AGENTS_V2[letter]["nombre"]}'


def letters_by_family(family: str | None = None) -> List[str]:
    """Letras en orden A–K, filtradas por familia si se indica."""
    if not family:
        return list(AGENT_LETTERS)
    return [k for k in AGENT_LETTERS if AGENTS_V2[k]['familia'] == family]
