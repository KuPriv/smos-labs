"""Приближённое вычисление непрерывного преобразования Фурье (Theory_Lab1).

Прямое ПФ:   P(ω) = ∫ p(t) e^{-iωt} dt,  ω = 2πf
  1) формула прямоугольников:   P(ω) ≈ Δt Σ_j p_j e^{-iω t_j}
  2) сплайн-«шапочки» φ_j(t):  P(ω) ≈ Σ_j p_j Φ_j(ω),
        Φ_j(ω) = ∫ φ_j(t) e^{-iωt} dt = Δt e^{-iω t_j} [sin(ωΔt/2) / (ωΔt/2)]^2

Обратное ПФ: p(t) = (1/π) Re ∫_0^Ω P(ω) e^{iωt} dω
  1) прямоугольники:  p1(t) ≈ (Δω/π) Re Σ_i w_i P_i e^{iω_i t}
  2) «шапочки» по ω:  p1(t) ≈ (1/π) Re Σ_i w_i P_i Ψ_i(t),
        Ψ_i(t) = Δω e^{iω_i t} [sin(tΔω/2) / (tΔω/2)]^2
  Узел ω = 0 входит с весом w_0 = 1/2: от «шапочки» в нуле на [0, Ω] лежит только половина
  (то же даёт и симметричное продолжение P(-ω) = P*(ω) вещественного сигнала).

Частоты задаются в размерных единицах f (кГц при времени в мс, Гц при времени в с).
"""
import numpy as np

TWO_PI = 2 * np.pi
_MAX_ELEMS = 4_000_000          # размер блока матрицы экспонент (память)


def _sinc2(z):
    """[sin(z)/z]^2 (np.sinc(x) = sin(πx)/(πx))."""
    return np.sinc(z / np.pi) ** 2


def forward_ft(t, p, f, method="rect"):
    """Прямое ПФ на частотах f (массив). method: 'rect' | 'spline'."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    f = np.asarray(f, float)
    dt = t[1] - t[0]
    P = np.empty(f.size, complex)
    step = max(1, _MAX_ELEMS // t.size)
    for s in range(0, f.size, step):
        w = TWO_PI * f[s:s + step]
        P[s:s + step] = np.exp(-1j * np.outer(w, t)) @ p          # Σ p_j e^{-iω t_j}
    P *= dt
    if method == "spline":
        P *= _sinc2(TWO_PI * f * dt / 2)                              # Φ_j = Δt e^{-iωt_j} sinc²(ωΔt/2)
    elif method != "rect":
        raise ValueError(method)
    return P


def inverse_ft(f, P, t, method="rect", half_first=True):
    """Обратное ПФ по значениям P на равномерной сетке частот f = 0, Δf, ..., F."""
    f = np.asarray(f, float)
    t = np.asarray(t, float)
    df = f[1] - f[0]
    dw = TWO_PI * df
    w = np.ones(f.size)
    if half_first and f[0] == 0.0:
        w[0] = 0.5
    PW = P * w
    out = np.empty(t.size)
    step = max(1, _MAX_ELEMS // f.size)
    for s in range(0, t.size, step):
        tt = t[s:s + step]
        out[s:s + step] = (np.exp(1j * TWO_PI * np.outer(tt, f)) @ PW).real
    out *= dw / np.pi
    if method == "spline":
        out *= _sinc2(t * dw / 2)                                     # Ψ_i = Δω e^{iω_i t} sinc²(tΔω/2)
    elif method != "rect":
        raise ValueError(method)
    return out
