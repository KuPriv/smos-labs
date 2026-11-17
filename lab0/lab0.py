"""Лабораторная работа № 0.

1) Аналитические преобразования Фурье  Q(α) = ∫ q(x) e^{iαx} dx  функций 1)-14)
   (формулы – в функциях Q1...Q14, проверка – сравнением с квадратурой Гаусса).
2) Численное обращение  q̃(x) = 1/(2π) ∫_{-A}^{A} Q(α) e^{-iαx} dα  (интеграл (2),
   усечённый до [-A, A]) четырьмя квадратурными формулами на равномерной сетке
   с шагом h:  прямоугольники, трапеции, Симпсон, «сплайн-шапочки» (метод Филона:
   кусочно-линейная интерполяция Q и точное интегрирование экспоненты).
3) Сопоставление q и q̃, исследование влияния A, h и метода.

Запуск:  python lab0/lab0.py [--a 1.0]
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import j0, j1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.plotting import plt, save, save_results  # noqa: E402

LAB = os.path.dirname(os.path.abspath(__file__))
PI = np.pi


# ----------------------------------------------------------------------------
# 1. Исходные функции q(x) и их аналитические преобразования Фурье Q(α)
# ----------------------------------------------------------------------------
def box(x, lo, hi, closed=True):
    return ((x >= lo) & (x <= hi)) if closed else ((x > lo) & (x < hi))


def G(g):
    """G(γ) = ∫_0^2 e^{iγx} dx = 2 e^{iγ} sin(γ)/γ  (без особенности в γ = 0)."""
    return 2.0 * np.exp(1j * g) * np.sinc(g / PI)


def cos_term(al, c, a):
    """∫_{-a}^{a} cos(cx) e^{iαx} dx = sin((α-c)a)/(α-c) + sin((α+c)a)/(α+c)."""
    return a * (np.sinc((al - c) * a / PI) + np.sinc((al + c) * a / PI))


def make_functions(a):
    """Список задач: q(x), Q(α) (аналитически), носитель, точки разрыва q, особые точки формулы Q."""
    b4, b5, c = PI / (2 * a), PI / a, 2.0 / a
    F = []

    def add(q, Q, supp, jumps, sing):
        F.append(dict(q=q, Q=Q, supp=supp, jumps=jumps, sing=sing))

    # 1) x/a
    add(lambda x: np.where(box(x, -a, a), x / a, 0.0),
        lambda al: 2j * (np.sin(al * a) - al * a * np.cos(al * a)) / (a * al ** 2),
        (-a, a), [-a, a], [0.0])
    # 2) a x^3
    add(lambda x: np.where(box(x, -a, a), a * x ** 3, 0.0),
        lambda al: 2j * a * ((3 * a ** 2 / al ** 2 - 6 / al ** 4) * np.sin(al * a)
                             + (6 * a / al ** 3 - a ** 3 / al) * np.cos(al * a)),
        (-a, a), [-a, a], [0.0])
    # 3) 1 - x^2/a^2
    add(lambda x: np.where(box(x, -a, a), 1 - x ** 2 / a ** 2, 0.0),
        lambda al: 4 * (np.sin(al * a) - al * a * np.cos(al * a)) / (a ** 2 * al ** 3),
        (-a, a), [], [0.0])
    # 4) cos(pi x / 2a)
    add(lambda x: np.where(box(x, -a, a), np.cos(PI * x / (2 * a)), 0.0),
        lambda al: 4 * PI * a * np.cos(al * a) / (PI ** 2 - 4 * a ** 2 * al ** 2),
        (-a, a), [], [-b4, b4])
    # 5) sin(pi x / a)
    add(lambda x: np.where(box(x, -a, a), np.sin(PI * x / a), 0.0),
        lambda al: 2j * PI * a * np.sin(al * a) / (PI ** 2 - a ** 2 * al ** 2),
        (-a, a), [], [-b5, b5])
    # 6) cos^2(x/a) = (1 + cos(2x/a))/2
    add(lambda x: np.where(box(x, -a, a), np.cos(x / a) ** 2, 0.0),
        lambda al: a * np.sinc(al * a / PI) + 0.5 * cos_term(al, c, a),
        (-a, a), [-a, a], [])
    # 7) sin^2(x/a) = (1 - cos(2x/a))/2
    add(lambda x: np.where(box(x, -a, a), np.sin(x / a) ** 2, 0.0),
        lambda al: a * np.sinc(al * a / PI) - 0.5 * cos_term(al, c, a),
        (-a, a), [-a, a], [])
    # 8) |x|/a
    add(lambda x: np.where(box(x, -a, a), np.abs(x) / a, 0.0),
        lambda al: 2 * np.sin(al * a) / al + 2 * (np.cos(al * a) - 1) / (a * al ** 2),
        (-a, a), [-a, a], [0.0])
    # 9) exp(-|x|)
    add(lambda x: np.where(box(x, -a, a), np.exp(-np.abs(x)), 0.0),
        lambda al: 2 * (1 - np.exp(-a) * (np.cos(al * a) - al * np.sin(al * a))) / (1 + al ** 2),
        (-a, a), [-a, a], [])
    # 10) x + x^2
    add(lambda x: np.where(box(x, -a, a), x + x ** 2, 0.0),
        lambda al: (2j * (np.sin(al * a) - al * a * np.cos(al * a)) / al ** 2
                    + 2 * (a ** 2 * np.sin(al * a) / al + 2 * a * np.cos(al * a) / al ** 2
                           - 2 * np.sin(al * a) / al ** 3)),
        (-a, a), [-a, a], [0.0])
    # 11) sin(2πx) sin(πx/2) = [cos(3πx/2) - cos(5πx/2)]/2 на (0, 2)
    add(lambda x: np.where(box(x, 0, 2, False), np.sin(2 * PI * x) * np.sin(PI * x / 2), 0.0),
        lambda al: 0.25 * (G(al + 1.5 * PI) + G(al - 1.5 * PI) - G(al + 2.5 * PI) - G(al - 2.5 * PI)),
        (0, 2), [], [])
    # 12) ½ sin(2πx)(1 - cos πx) = ½ sin 2πx - ¼ sin 3πx - ¼ sin πx на (0, 2)
    S = lambda al, be: (G(al + be) - G(al - be)) / 2j  # noqa: E731  ∫_0^2 sin(βx) e^{iαx} dx
    add(lambda x: np.where(box(x, 0, 2, False), 0.5 * np.sin(2 * PI * x) * (1 - np.cos(PI * x)), 0.0),
        lambda al: 0.5 * S(al, 2 * PI) - 0.25 * S(al, 3 * PI) - 0.25 * S(al, PI),
        (0, 2), [], [])
    # 13) 1/sqrt(1-x^2)
    add(lambda x: np.where(box(x, -1, 1, False), 1 / np.sqrt(np.clip(1 - x ** 2, 1e-300, None)), 0.0),
        lambda al: PI * j0(al) + 0j,
        (-1, 1), [-1, 1], [])
    # 14) x/sqrt(1-x^2)
    add(lambda x: np.where(box(x, -1, 1, False), x / np.sqrt(np.clip(1 - x ** 2, 1e-300, None)), 0.0),
        lambda al: 1j * PI * j1(al),
        (-1, 1), [-1, 1], [])
    for k, f in enumerate(F, 1):
        f["id"] = k
    return F


def Q_quad(f, al, n=400):
    """Q(α) квадратурой Гаусса–Лежандра по определению (1).
    Для 13), 14) – после замены x = sin θ, снимающей особенность на концах."""
    t, w = np.polynomial.legendre.leggauss(n)
    al = np.atleast_1d(al).astype(float)
    if f["id"] in (13, 14):
        th = 0.5 * PI * t
        x, wx = np.sin(th), 0.5 * PI * w
        g = np.ones_like(x) if f["id"] == 13 else x
        return (g * wx * np.exp(1j * np.outer(al, x))).sum(axis=1)
    lo, hi = f["supp"]
    # отрезок делится в точке x = 0 (излом |x| у функций 8, 9), чтобы сохранить точность
    parts = [(lo, 0.0), (0.0, hi)] if lo < 0 < hi else [(lo, hi)]
    res = 0
    for l, r in parts:
        x = 0.5 * (r - l) * t + 0.5 * (r + l)
        wx = 0.5 * (r - l) * w
        res = res + (f["q"](x) * wx * np.exp(1j * np.outer(al, x))).sum(axis=1)
    return res


def Q_exact(f, al, tol=1e-2):
    """Аналитическая формула; в малой окрестности устранимых особенностей формулы
    (0/0 при α = 0, α = ±π/(2a) и т.п.) значение берётся по квадратуре."""
    al = np.asarray(al, dtype=float)
    with np.errstate(all="ignore"):
        Q = np.asarray(f["Q"](al), dtype=complex)
    near = np.zeros(al.shape, bool)
    for s in f["sing"]:
        near |= np.abs(al - s) < tol
    if near.any():
        Q[near] = Q_quad(f, al[near])
    return Q


# ----------------------------------------------------------------------------
# 2. Численное обращение преобразования Фурье (формула (2))
# ----------------------------------------------------------------------------
def quad_weights(n, h, method):
    """Веса квадратурных формул на равномерной сетке из n узлов с шагом h."""
    w = np.full(n, h)
    if method == "rect":                    # левые прямоугольники
        w[-1] = 0.0
    elif method == "trap":                  # трапеции
        w[0] = w[-1] = h / 2
    elif method == "simpson":               # Симпсон (n нечётное)
        assert n % 2 == 1
        w[1:-1:2], w[2:-1:2] = 4 * h / 3, 2 * h / 3
        w[0] = w[-1] = h / 3
    return w


def inverse_ft(Qk, alpha, x, method="trap", chunk=2000):
    """q̃(x) = 1/(2π) ∫ Q(α) e^{-iαx} dα по узлам alpha (равномерная сетка) и значениям Qk.

    method: 'rect' | 'trap' | 'simpson' – классические квадратуры;
            'spline' – Q(α) интерполируется «шапочками» φ_k(α) (сплайн 1-го порядка),
            интегралы ∫ φ_k(α) e^{-iαx} dα берутся точно:
               внутренний узел:  h e^{-iα_k x} [sin(xh/2)/(xh/2)]^2,
               концевой узел:    половина «шапочки» – формула (e^{uh}-1-uh)/(u^2 h), u = ∓ix.
    """
    h = alpha[1] - alpha[0]
    n = len(alpha)
    x = np.asarray(x, float)
    out = np.empty(x.shape, complex)
    if method == "spline":
        w = np.full(n, 1.0 + 0j)
    else:
        w = quad_weights(n, h, method).astype(complex)
    for s in range(0, x.size, chunk):
        xs = x.ravel()[s:s + chunk]
        E = np.exp(-1j * np.outer(xs, alpha))          # e^{-i α_k x}
        if method == "spline":
            sinc2 = np.sinc(xs * h / (2 * PI)) ** 2     # [sin(xh/2)/(xh/2)]^2
            ww = np.tile(h * sinc2[:, None], (1, n)).astype(complex)
            ww[:, 0] = half_hat(-1j * xs, h)            # левый конец: ∫_0^h (1-s/h) e^{-i(α0+s)x} ds
            ww[:, -1] = half_hat(1j * xs, h)            # правый конец (зеркально)
            out.ravel()[s:s + chunk] = (E * ww * Qk[None, :]).sum(axis=1) / (2 * PI)
        else:
            out.ravel()[s:s + chunk] = E @ (w * Qk) / (2 * PI)
    return out


def half_hat(u, h):
    """∫_0^h (1 - s/h) e^{u s} ds = (e^{uh} - 1 - uh) / (u^2 h)  (ряд Тейлора при малых |uh|)."""
    u = np.asarray(u, complex)
    z = u * h
    res = np.empty_like(z)
    small = np.abs(z) < 1e-3
    zs = z[small]
    res[small] = h * (0.5 + zs / 6 + zs ** 2 / 24)
    zl = z[~small]
    res[~small] = h * (np.exp(zl) - 1 - zl) / zl ** 2
    return res


def reconstruct(f, x, A, h, method="trap"):
    n = int(round(2 * A / h))
    if method == "simpson" and n % 2:
        n += 1
    alpha = np.linspace(-A, A, n + 1)
    return inverse_ft(Q_exact(f, alpha), alpha, x, method).real


def errors(f, x, qt, delta=0.05):
    """Макс. ошибка вне δ-окрестностей разрывов и среднеквадратичная ошибка на всей сетке."""
    q = f["q"](x)
    mask = np.ones_like(x, bool)
    for xj in f["jumps"]:
        mask &= np.abs(x - xj) > delta
    if f["id"] in (13, 14):          # q неограничена у x = ±1
        mask &= np.abs(np.abs(x) - 1) > 0.1
    e = np.abs(q - qt)
    return float(e[mask].max()), float(np.sqrt(np.mean(e[mask] ** 2)))


# ----------------------------------------------------------------------------
# 3. Эксперименты и графики
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=1.0, help="параметр a в заданиях 1-10")
    args = ap.parse_args()
    a = args.a
    F = make_functions(a)
    res = {"a": a}
    A0, h0 = 200.0, 0.05
    res["A0"], res["h0"] = A0, h0

    # --- проверка аналитических формул квадратурой --------------------------------
    al = np.linspace(-60, 60, 2401)
    check = {}
    for aa in (0.5, 1.0, 2.0):
        for f in make_functions(aa):
            err = np.abs(Q_exact(f, al) - Q_quad(f, al, 800)).max()
            check.setdefault(str(f["id"]), {})[str(aa)] = float(err)
    res["check"] = check
    print("max|Q_formula - Q_quad|:", {k: max(v.values()) for k, v in check.items()})

    # --- q и q̃ для всех 14 функций --------------------------------------------------
    x = np.linspace(-2.5 * max(a, 1), 2.5 * max(a, 1), 1001)
    table = {}
    for part, ids in ((1, range(0, 7)), (2, range(7, 14))):
        fig, axs = plt.subplots(7, 2, figsize=(7.2, 12.5))
        for r, k in enumerate(ids):
            f = F[k]
            alp = np.linspace(-40, 40, 3001)
            Qv = Q_exact(f, alp)
            ax = axs[r, 0]
            ax.plot(alp, Qv.real, label="Re Q")
            ax.plot(alp, Qv.imag, "--", label="Im Q")
            ax.set_title(f"{f['id']}) Q(α)")
            ax.set_xlabel("α")
            if r == 0:
                ax.legend(loc="upper right")
            qt = reconstruct(f, x, A0, h0, "trap")
            ax = axs[r, 1]
            ax.plot(x, f["q"](x), "k", lw=1.6, label="q(x)")
            ax.plot(x, qt, "r--", lw=1, label="q̃(x)")
            if f["id"] in (13, 14):
                ax.set_ylim(-4, 4)
            ax.set_title(f"{f['id']}) q(x) и q̃(x), A={A0:g}, h={h0:g}")
            ax.set_xlabel("x")
            if r == 0:
                ax.legend(loc="upper right")
            table[str(f["id"])] = {}
            for A in (20.0, 100.0, 500.0):
                table[str(f["id"])][str(int(A))] = errors(f, x, reconstruct(f, x, A, h0, "trap"))
        save(fig, LAB, f"fig_all_{part}.png")
    res["table"] = table

    # --- влияние верхней границы A ----------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, k in zip(axs.ravel(), (0, 2, 8, 12)):
        f = F[k]
        ax.plot(x, f["q"](x), "k", lw=1.8, label="q(x)")
        for A in (5, 15, 50):
            ax.plot(x, reconstruct(f, x, A, 0.05), lw=0.9, label=f"A={A}")
        ax.set_title(f"Функция {f['id']}: влияние A")
        ax.set_xlim(-1.6 * max(a, 1), 1.6 * max(a, 1)) if f["id"] != 11 else ax.set_xlim(-0.5, 2.5)
        if f["id"] == 13:
            ax.set_ylim(-1, 6)
        ax.legend(fontsize=7)
    save(fig, LAB, "fig_A_influence.png")

    # --- зависимость ошибки от A (log-log) ----------------------------------------
    As = np.array([5, 10, 20, 40, 80, 160, 320, 640, 1280])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    slopes = {}
    x = np.linspace(-2.5, 2.5, 401)
    for k in (0, 2, 3, 8, 10, 12):
        f = F[k]
        e = [errors(f, x, reconstruct(f, x, A, 0.05))[1] for A in As]
        ax.loglog(As, e, "o-", ms=3, label=f"функция {f['id']}")
        p = np.polyfit(np.log(As[3:]), np.log(e[3:]), 1)[0]
        slopes[str(f["id"])] = float(p)
    ax.set_xlabel("A (верхняя граница по α)")
    ax.set_ylabel("СКО |q - q̃| (вне окрестн. разрывов)")
    ax.legend(ncol=2)
    save(fig, LAB, "fig_err_vs_A.png")
    res["slopes"] = slopes

    # --- влияние шага h: периодизация q̃ с периодом 2π/h ---------------------------
    f = F[8]                                  # функция 9
    xw = np.linspace(-14, 14, 2001)
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, h in zip(axs, (0.5, 1.0, 2.0)):
        ax.plot(xw, f["q"](xw), "k", lw=1.6, label="q(x)")
        ax.plot(xw, reconstruct(f, xw, 100.0, h, "trap"), "r", lw=0.9, label="q̃(x), трапеции")
        ax.plot(xw, reconstruct(f, xw, 100.0, h, "spline"), "b--", lw=0.9, label="q̃(x), «шапочки»")
        ax.set_title(f"Функция 9, A=100, h={h}:  период повторения 2π/h = {2 * PI / h:.2f}")
        ax.legend(loc="upper right", fontsize=7)
    axs[-1].set_xlabel("x")
    save(fig, LAB, "fig_h_influence.png")

    # --- сравнение квадратурных формул: ошибка от h --------------------------------
    hs = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6])
    methods = {"rect": "прямоугольники", "trap": "трапеции", "simpson": "Симпсон", "spline": "«шапочки»"}
    xin = np.linspace(-2.5, 2.5, 501)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    meth_err = {}
    for ax, k in zip(axs, (2, 8)):
        f = F[k]
        meth_err[str(f["id"])] = {}
        for m, lab in methods.items():
            e = [errors(f, xin, reconstruct(f, xin, 100.0, h, m))[1] for h in hs]
            meth_err[str(f["id"])][m] = e
            ax.loglog(hs, e, "o-", ms=3, label=lab)
        ax.set_title(f"Функция {f['id']}, A = 100")
        ax.set_xlabel("шаг h по α")
        ax.set_ylabel("СКО |q - q̃|")
        ax.legend(fontsize=7)
    save(fig, LAB, "fig_methods.png")
    res["meth_err"] = meth_err
    res["hs"] = hs.tolist()

    # --- Гиббс у разрыва (функция 1) ------------------------------------------------
    f = F[0]
    xg = np.linspace(0.6 * a, 1.4 * a, 801)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(xg, f["q"](xg), "k", lw=1.8, label="q(x)")
    over = {}
    for A in (25, 100, 400):
        qt = reconstruct(f, xg, A, 0.01)
        ax.plot(xg, qt, lw=0.9, label=f"A={A}")
        over[str(A)] = float((qt.max() - 1.0))
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("Функция 1 в окрестности разрыва x = a: явление Гиббса")
    ax.set_xlabel("x")
    ax.legend()
    save(fig, LAB, "fig_gibbs.png")
    res["gibbs"] = over

    # --- таблицы для отчёта ----------------------------------------------------------
    rows = ["| № | ошибка формулы Q(α) | A = 20: max / СКО | A = 100: max / СКО | A = 500: max / СКО |",
            "|:-:|:-:|:-:|:-:|:-:|"]
    for k in map(str, range(1, 15)):
        cells = [f"{max(check[k].values()):.0e}"]
        for A in ("20", "100", "500"):
            mx, rms = table[k][A]
            cells.append(f"{mx:.1e} / {rms:.1e}")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    res["md_table_err"] = "\n".join(rows)
    save_results(LAB, res)


if __name__ == "__main__":
    main()
