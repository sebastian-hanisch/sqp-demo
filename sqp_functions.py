"""Vehikel B (Kopie aus lagrange-kkt-demo/straf-barriere-demo, kein Import): zylindrischer
Transportbehaelter. Zielfunktion, Gleichungs-/Ungleichungsnebenbedingung je mit Gradient UND
Hesse-Matrix von Hand hergeleitet. x = (r, h)."""
import numpy as np


def surface_area(x: np.ndarray) -> float:
    r, h = x
    return 2 * np.pi * r ** 2 + 2 * np.pi * r * h


def grad_surface_area(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([4 * np.pi * r + 2 * np.pi * h, 2 * np.pi * r])


def hess_surface_area() -> np.ndarray:
    return np.array([[4 * np.pi, 2 * np.pi], [2 * np.pi, 0.0]])


def volume_constraint(x: np.ndarray, V0: float) -> float:
    """g1(x) = 0 bei festem Volumen V0."""
    r, h = x
    return np.pi * r ** 2 * h - V0


def grad_volume_constraint(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([2 * np.pi * r * h, np.pi * r ** 2])


def hess_volume_constraint(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([[2 * np.pi * h, 2 * np.pi * r], [2 * np.pi * r, 0.0]])


def height_constraint(x: np.ndarray, h_max: float) -> float:
    """g2(x) <= 0 beim Hoehenlimit h_max."""
    r, h = x
    return h - h_max


def grad_height_constraint() -> np.ndarray:
    return np.array([0.0, 1.0])
