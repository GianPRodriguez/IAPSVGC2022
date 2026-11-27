"""Arranque en caliente con ampliación de la entrada (PPO v3 / DDQN v2, protocolo §4.16).

La memoria de serie añade ``MEM_DIM`` features al final de la obs. El codificador estructurado las pasa
como segmento crudo al final de su salida, así que solo crece la entrada de la primera capa del torso
(PPO ``fc1``, DDQN ``net.0``). ``expand_state_dict`` copia todos los pesos del checkpoint de origen y, en
esas capas, rellena las columnas nuevas (las últimas) con **cero**: la red ampliada calcula exactamente
lo mismo que la de origen hasta que el entrenamiento mueva esos pesos.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple


def expand_state_dict(src: Dict, dst: Dict, *, first_layers: Sequence[str]) -> Tuple[Dict, Dict[str, object]]:
    """Devuelve (state_dict para ``dst``, resumen). Falla si algo que no es primera capa no encaja."""
    import torch

    out = {}
    expanded, copied = [], 0
    for k, v_dst in dst.items():
        if k not in src:
            raise KeyError(f'arranque en caliente: falta {k!r} en el checkpoint de origen')
        v_src = src[k]
        if tuple(v_src.shape) == tuple(v_dst.shape):
            out[k] = v_src.detach().clone().to(dtype=v_dst.dtype)
            copied += 1
            continue
        if k in first_layers and v_src.dim() == 2 and v_dst.dim() == 2 \
                and v_src.shape[0] == v_dst.shape[0] and v_src.shape[1] < v_dst.shape[1]:
            w = torch.zeros_like(v_dst)
            w[:, :v_src.shape[1]] = v_src.to(dtype=v_dst.dtype)
            out[k] = w
            expanded.append(f'{k}: {tuple(v_src.shape)} -> {tuple(v_dst.shape)} (columnas nuevas en 0)')
            continue
        raise ValueError(f'arranque en caliente: {k!r} no encaja {tuple(v_src.shape)} -> {tuple(v_dst.shape)}')
    extra = sorted(set(src) - set(dst))
    if extra:
        raise KeyError(f'arranque en caliente: el origen tiene claves que el destino no: {extra[:5]}')
    return out, {'copiados': copied, 'ampliados': expanded}
