import sqp_constants as C
import sqp_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert C.V0_MIN <= preset["V0"] <= C.V0_MAX
        assert C.H_MAX_MIN <= preset["h_max"] <= C.H_MAX_MAX


def test_preset_hoehenlimit_nicht_bindend_gives_inactive_case():
    p = C.PRESETS["hoehenlimit_nicht_bindend"]
    out = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"]))
    assert out["solution"].case == "inactive"


def test_preset_hoehenlimit_bindet_gives_active_case():
    p = C.PRESETS["hoehenlimit_bindet"]
    out = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"]))
    assert out["solution"].case == "active"
    assert out["solution"].mu > 0
