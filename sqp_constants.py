"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

V0_MIN, V0_MAX, V0_DEFAULT = 1.0, 100.0, 10.0
H_MAX_MIN, H_MAX_MAX, H_MAX_DEFAULT = 0.5, 6.0, 5.0

PRESETS = {
    "hoehenlimit_nicht_bindend": dict(
        label="Höhenlimit nicht bindend",
        V0=10.0, h_max=5.0,
        help="Nur die Volumen-Gleichung ist am Optimum aktiv — SQP konvergiert in 5 Iterationen "
             "zur klassischen 'Höhe=Durchmesser'-Lösung.",
    ),
    "hoehenlimit_bindet": dict(
        label="Höhenlimit bindet",
        V0=10.0, h_max=1.5,
        help="Beide Nebenbedingungen sind am Optimum aktiv — SQP identifiziert die aktive Menge "
             "sofort korrekt und konvergiert ebenfalls in 5 Iterationen.",
    ),
}
