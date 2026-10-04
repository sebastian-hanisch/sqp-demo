from streamlit.testing.v1 import AppTest

_APP_TIMEOUT = 60


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_preset_hoehenlimit_nicht_bindend():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Höhenlimit nicht bindend"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Fall"] == "Höhenlimit inaktiv"


def test_preset_hoehenlimit_bindet():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Höhenlimit bindet"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Fall"] == "Höhenlimit bindet"


def test_sliders_extreme_values_do_not_crash():
    at = _fresh()
    v0_slider = [s for s in at.slider if s.label == "Volumen V₀"][0]
    v0_slider.set_value(v0_slider.max).run()
    assert not at.exception
    h_slider = [s for s in at.slider if s.label == "Höhenlimit h_max"][0]
    h_slider.set_value(h_slider.min).run()
    assert not at.exception


def test_small_volume_does_not_show_none_in_comparison_text():
    """Regression: für V0 <= 5 erreicht das Strafverfahren die Toleranz nicht; die App darf
    nicht „brauchen None äußere Stufen“ schreiben."""
    at = _fresh()
    v0_slider = [s for s in at.slider if s.label == "Volumen V₀"][0]
    v0_slider.set_value(v0_slider.min).run()
    assert not at.exception
    texts = [m.value for m in at.markdown if "SQP braucht" in m.value]
    assert texts, "Vergleichstext fehlt"
    assert all("None" not in t for t in texts)
    assert any("gar nicht" in t for t in texts)
