from showdown.inference_sim import analyze_speed as a


def _row(before, after, true_sp, **kw):
    r = {'kind': 'speed', 'species': 'garchomp', 'before': before, 'after': after,
         'true_spe_sp': true_sp, 'true_in_before': true_sp in before,
         'true_in_after': true_sp in after, 'opponent_first': True, 'user_speed': 100,
         'pid': 1, 'series': 1, 'seq': 0}
    r.update(kw)
    return r


def test_summarize_counts_risk_and_prune():
    rows = [_row(list(range(10)), list(range(5, 10)), 7),
            _row(list(range(10)), list(range(5, 10)), 2)]
    s = a.summarize(rows)
    assert s['observaciones'] == 2 and s['verdadero_excluido'] == 1
    assert s['con_poda'] == 2 and abs(s['poda_media'] - 0.5) < 1e-9


def test_report_flags_context_and_failures():
    rows = [_row(list(range(10)), list(range(5, 10)), 2, opp_item='choicescarf')]
    text = a.format_report(rows)
    assert 'item' in text and 'FALLO garchomp' in text


def test_speed_sink_is_optional_and_clearable():
    from showdown import stat_inference as si
    si.set_speed_sink(lambda ev: None)
    si.set_speed_sink(None)
    assert si._SPEED_SINK is None


def test_mega_family_bases_cover_pre_and_post_mega():
    from showdown.stat_inference import mega_family_spe_bases
    bases = mega_family_spe_bases('floette-eternal', 'floettite')
    assert 92 in bases and 102 in bases
    assert 112 in mega_family_spe_bases('lucario', 'lucarionite')
    assert mega_family_spe_bases('lucario', 'leftovers') == ()
    assert mega_family_spe_bases('incineroar', 'lifeorb') == ()


def test_speed_order_ok_directions():
    from showdown.stat_inference import speed_order_ok as ok
    assert ok(120, 100, True, False, 0) and not ok(90, 100, True, False, 0)
    assert ok(90, 100, False, False, 0) and not ok(120, 100, False, False, 0)
    assert ok(90, 100, True, True, 0) and not ok(120, 100, True, True, 0)
    assert ok(90, 100, True, False, 12)
