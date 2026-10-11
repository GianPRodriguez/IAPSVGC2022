"""Textos educativos (español, analogía ajedrez) — agentes v1."""
from __future__ import annotations

from typing import Dict, List

# Clave = carpeta battle_bots / BATTLE_BOT
AGENTS_V1: List[str] = [
    'random',
    'greedy_solo',
    'greedy_joint',
    'expectimax_bayesian',
]

AGENT_LABELS: Dict[str, str] = {
    'random': 'Aleatorio',
    'greedy_solo': 'Greedy (por Pokémon)',
    'greedy_joint': 'Greedy (en pareja)',
    'expectimax_bayesian': 'Expectimax',
}

AGENT_CHESS: Dict[str, Dict[str, str]] = {
    'random': {
        'titulo': 'Agente aleatorio',
        'subtitulo': 'Como tirar un dado entre jugadas legales',
        'parrafos': [
            'En ajedrez, imagina que miras el tablero, listas solo las jugadas '
            'permitidas y eliges una al azar. No miras si ganas material ni si '
            'dejas al rey expuesto: solo evitas mover “ilegal”.',
            'En VGC es igual: el bot consulta qué movimientos y cambios admite '
            'Showdown en ese turno y sortea uno. Sirve de rival fácil y de '
            'línea base: si otro agente no le gana con claridad, aún le falta práctica.',
            'No “aprende” ni mejora con el tiempo. Cada partida vuelve a empezar '
            'desde cero, como un jugador que siempre juega a ciegas.',
        ],
        'usa_cuando': 'Para calentar, probar tu equipo o ver si la conexión con Showdown funciona.',
    },
    'greedy_solo': {
        'titulo': 'Greedy por Pokémon (solo)',
        'subtitulo': 'Como capturar la pieza más valiosa… mirando solo una pieza tuya',
        'parrafos': [
            'En ajedrez, un estilo “avaricioso” sería: en cada turno, con el alfil '
            '(o la torre) que estás mirando, capturar lo que más puntos dé ahora, '
            'sin pensar mucho en el resto del tablero.',
            'Este bot mira cada Pokémon activo por separado y elige el ataque o '
            'cambio que, según un cálculo rápido de daño, más perjudica al rival '
            'en ese instante. No coordina bien los dos Pokémon a la vez.',
            'Es rápido e intuitivo —como un aficionado que siempre captura— pero '
            'puede ignorar jugadas de equipo (protección, apoyo, turnos futuros).',
        ],
        'usa_cuando': 'Si quieres un rival agresivo y sencillo, sin esperas largas.',
    },
    'greedy_joint': {
        'titulo': 'Greedy en pareja (joint)',
        'subtitulo': 'Como elegir la mejor jugada de dos piezas a la vez',
        'parrafos': [
            'Siguiendo la analogía del ajedrez: en vez de decidir solo con el alfil, '
            'miras el alfil y el caballo juntos y eliges la pareja de jugadas que '
            'más “puntos” da ahora (por ejemplo, dos capturas fuertes a la vez).',
            'En VGC el bot prueba combinaciones legales de lo que hacen tus dos '
            'Pokémon activos y se queda con la que más daño estimado suma al rival.',
            'Sigue siendo “corto de vista” (casi no simula muchos turnos adelante), '
            'pero suele coordinar mejor que el greedy por separado: menos fuego amigo '
            'absurdo y mejor foco en el mismo objetivo cuando conviene.',
        ],
        'usa_cuando': 'Buen rival intermedio: más listo que el aleatorio, más rápido que el Expectimax.',
    },
    'expectimax_bayesian': {
        'titulo': 'Expectimax (con creencias)',
        'subtitulo': 'Como calcular variantes y promediar lo que puede pasar',
        'parrafos': [
            'En ajedrez avanzado no solo miras la captura inmediata: calculas '
            'variantes (“si yo hago A, puede responder B o C”) y eliges la línea '
            'que, en promedio, te deja mejor. Eso se parece al expectimax: no asume '
            'que el rival siempre juega perfecto, sino que sopesa posibilidades.',
            'Este bot simula turnos hacia adelante, tiene en cuenta la suerte del '
            'juego (daños, precisión) y mantiene “creencias” sobre lo que el rival '
            'podría tener (como intuiciones que se actualizan al ver jugadas).',
            'Por eso piensa más y a veces tarda. A cambio, suele jugar más “de '
            'entrenador”: prepárate para protecciones, cambios y planes a medio plazo.',
        ],
        'usa_cuando': 'Para un desafío serio o para estudiar cómo se siente un rival que calcula.',
    },
}


def agent_options() -> List[tuple]:
    """Lista (clave, etiqueta) para selectores."""
    return [(k, AGENT_LABELS[k]) for k in AGENTS_V1]
