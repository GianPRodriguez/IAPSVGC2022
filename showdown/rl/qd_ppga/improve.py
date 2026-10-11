"""Proximal PG emit — reuses QD-PGA pg_improve (no mutation)."""
from showdown.rl.qd_pga.improve import G_C_DEFAULT, K_PG_DEFAULT, pg_improve

__all__ = ['G_C_DEFAULT', 'K_PG_DEFAULT', 'pg_improve']
