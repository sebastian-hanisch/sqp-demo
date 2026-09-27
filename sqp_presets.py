"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import sqp_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None


SETTING_SPECS = [
    SettingSpec("V0", "v0", C.V0_DEFAULT, float, (C.V0_MIN, C.V0_MAX)),
    SettingSpec("h_max", "hmax", C.H_MAX_DEFAULT, float, (C.H_MAX_MIN, C.H_MAX_MAX)),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            raw = params[spec.param]
            try:
                value = spec.cast(raw)
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    for field in ("V0", "h_max"):
        if field in preset:
            st.session_state[field] = preset[field]
            st.session_state[f"widget_{field}"] = preset[field]
