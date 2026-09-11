import json
from pathlib import Path

import constants


_MOVES_PATH = Path(__file__).resolve().parent / "JSONs" / "moves.json"


async def getColorMove(move: str) -> str:
    try:
        with _MOVES_PATH.open("r", encoding="utf-8") as moves_file:
            moves_data = json.load(moves_file)

        move_data = moves_data.get(move.strip().lower().replace(" ", "-"), {})
        move_type = move_data.get("type", "").lower()
        return constants.COLOR_TYPE.get(move_type, "#FFFFFF")
    except (OSError, json.JSONDecodeError) as error:
        print(f"[ERROR] Falló la lectura de moves.json para '{move}': {error}")
        return "#FFFFFF"
