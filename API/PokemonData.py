import json
from pathlib import Path


_STATS_PATH = Path(__file__).resolve().parent / "JSONs" / "champions_info_stats.json"


def getPkmnInfo(name_pkmn: str):
    try:
        with _STATS_PATH.open("r", encoding="utf-8") as stats_file:
            stats_data = json.load(stats_file)

        matched_key = next(
            (key for key in stats_data if key.lower() == name_pkmn.strip().lower()),
            None,
        )

        if matched_key:
            pkmn_info = stats_data[matched_key]
            return (
                pkmn_info.get("tipo1"),
                pkmn_info.get("tipo2"),
                pkmn_info.get("hp", 0),
                pkmn_info.get("ataque", 0),
                pkmn_info.get("defensa", 0),
                pkmn_info.get("ataque_especial", 0),
                pkmn_info.get("defensa_especial", 0),
                pkmn_info.get("velocidad", 0),
            )

        print(f"[AVISO] Pokémon '{name_pkmn}' no encontrado en champions_info_stats.json")
    except Exception as error:
        print(f"[ERROR] Falló la lectura de champions_info_stats.json para '{name_pkmn}': {error}")

    return None, None, 0, 0, 0, 0, 0, 0
