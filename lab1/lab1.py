"""Лабораторная работа № 1: прямое и обратное (непрерывное) преобразование Фурье.

Алгоритмы (common/ctft.py, по Theory_Lab1):
  прямое ПФ  – формула прямоугольников (2) и сплайн-«шапочки» (5);
  обратное ПФ – прямоугольники и «шапочки» по частоте (6)-(8).
Исследуется влияние: шага по времени Δt, верхней границы времени T, шага по частоте Δf,
верхней границы частоты F и алгоритма.

Запуск:  python lab1/lab1.py [файл1 файл2 ...]   (по умолчанию – сигналы из data/)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ctft import forward_ft, inverse_ft  # noqa: E402
from common.plotting import plt, save, save_results  # noqa: E402
from common.signals import Signal, load_signals  # noqa: E402

LAB = os.path.dirname(os.path.abspath(__file__))
METH = {"rect": "прямоугольники", "spline": "«шапочки»"}


def freq_grid(sig, df=None, fmax=None):
    """Сетка частот [0, F] с шагом Δf (по умолчанию F = fs/2, Δf = 1/(2T))."""
    fmax = sig.fs / 2 if fmax is None else fmax
    df = 1.0 / (2 * (sig.duration + sig.dt)) if df is None else df
    return np.arange(0, fmax + df / 2, df)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def spectrum_and_back(sig, method="rect", df=None, fmax=None):
    f = freq_grid(sig, df, fmax)
    P = forward_ft(sig.t, sig.x, f, method)
    rec = inverse_ft(f, P, sig.t, method)
    return f, P, rec


# --------------------------------------------------------------------------------
def exp_overview(sigs, res):
    """Сигналы, модуль спектра и восстановленные по спектру сигналы."""
    fig1, ax1 = plt.subplots(len(sigs), 2, figsize=(7.2, 2.0 * len(sigs)))
    fig2, ax2 = plt.subplots(len(sigs), 2, figsize=(7.2, 2.0 * len(sigs)))
    res["overview"] = {}
    for i, s in enumerate(sigs):
        t0 = time.perf_counter()
        f, P, rec = spectrum_and_back(s, "rect")
        _, Ps, recs = spectrum_and_back(s, "spline")
        el = time.perf_counter() - t0
        res["overview"][s.name] = dict(N=s.n, dt=s.dt, T=s.duration, M=int(f.size), df=float(f[1]),
                                       err_rect=rel_err(rec, s.x), err_spline=rel_err(recs, s.x),
                                       seconds=el)
        print(f"  {s.name}: N={s.n}, M={f.size}, err rect={rel_err(rec, s.x):.2e}, "
              f"spline={rel_err(recs, s.x):.2e}, {el:.1f} c")
        ax1[i, 0].plot(s.t, s.x, lw=0.6)
        ax1[i, 0].set_title(f"{s.name}: сигнал p(t)")
        ax1[i, 0].set_xlabel(f"t, {s.t_unit}")
        ax1[i, 1].plot(f, np.abs(P), lw=0.7, label=METH["rect"])
        ax1[i, 1].plot(f, np.abs(Ps), "--", lw=0.7, label=METH["spline"])
        ax1[i, 1].set_title(f"{s.name}: |P(f)|")
        ax1[i, 1].set_xlabel(f"f, {s.f_unit}")
        if s.f_unit == "кГц":
            ax1[i, 1].set_xlim(0, 500)
        ax1[i, 1].legend()
        # восстановление: весь сигнал и увеличенный фрагмент
        ax2[i, 0].plot(s.t, s.x, "k", lw=0.8, label="исходный")
        ax2[i, 0].plot(s.t, rec, "r--", lw=0.6, label="восстановленный")
        ax2[i, 0].set_title(f"{s.name}: исходный и восстановленный")
        ax2[i, 0].set_xlabel(f"t, {s.t_unit}")
        ax2[i, 0].legend(fontsize=6.5, loc="upper right")
        j = np.argmax(np.abs(s.x))
        w = 40
        sl = slice(max(0, j - w), min(s.n, j + w))
        ax2[i, 1].plot(s.t[sl], s.x[sl], "k.-", lw=0.8, ms=2.5, label="исходный")
        ax2[i, 1].plot(s.t[sl], rec[sl], "r--", lw=1, label=METH["rect"])
        ax2[i, 1].plot(s.t[sl], recs[sl], "b:", lw=1.2, label=METH["spline"])
        ax2[i, 1].set_title(f"фрагмент (ошибка: {rel_err(rec, s.x):.1e} / {rel_err(recs, s.x):.1e})")
        ax2[i, 1].set_xlabel(f"t, {s.t_unit}")
        ax2[i, 1].legend(fontsize=6.5, loc="upper right")
    save(fig1, LAB, "fig_signals_spectra.png")
    save(fig2, LAB, "fig_reconstruction.png")


def exp_dt(s, ks, fmax, name, res, f_lim=None):
    """Влияние шага по времени (прореживание в k раз) на |P(f)| для обоих алгоритмов."""
    fig, axs = plt.subplots(len(ks), 2, figsize=(7.2, 1.55 * len(ks) + 0.4), sharex=True)
    f = np.linspace(0, fmax, 3001)
    for r, k in enumerate(ks):
        d = s.decimate(k)
        for c, m in enumerate(("rect", "spline")):
            P = np.abs(forward_ft(d.t, d.x, f, m))
            ax = axs[r, c]
            ax.plot(f, P, lw=0.7)
            for q in range(1, int(fmax / d.fs * 2) + 1):       # границы Найквиста / кратные fs
                ax.axvline(q * d.fs / 2, color="gray", ls=":", lw=0.7)
            ax.set_xlim(0, fmax)
            ax.set_title(f"{METH[m]}: Δt = {d.dt * (1e3 if s.t_unit == 'мс' else 1e3):.3g} "
                         f"{'мкс' if s.t_unit == 'мс' else 'мс'} (fs = {d.fs:.4g} {s.f_unit})", fontsize=8)
            if f_lim:
                ax.set_xlim(*f_lim)
    for ax in axs[-1]:
        ax.set_xlabel(f"f, {s.f_unit}")
    save(fig, LAB, name)


def exp_T(s, Ts, fmax, name, f_lim=None, df=None):
    """Влияние верхней границы временного диапазона T на |P(f)|."""
    fig, axs = plt.subplots(len(Ts), 2, figsize=(7.2, 1.45 * len(Ts) + 0.4),
                            gridspec_kw=dict(width_ratios=[1, 1.4]))
    for r, T in enumerate(Ts):
        c = s.crop(T)
        f = np.linspace(0, fmax, 2501) if df is None else np.arange(0, fmax, df)
        P = np.abs(forward_ft(c.t, c.x, f, "spline"))
        axs[r, 0].plot(s.t, s.x, color="0.75", lw=0.5)
        axs[r, 0].plot(c.t, c.x, lw=0.6)
        axs[r, 0].set_title(f"T = {T:g} {s.t_unit} (N = {c.n})", fontsize=8)
        axs[r, 1].plot(f, P, lw=0.8)
        axs[r, 1].set_title(f"|P(f)|, разрешение по частоте ~1/T = {1 / T:.3g} {s.f_unit}", fontsize=8)
        if f_lim:
            axs[r, 1].set_xlim(*f_lim)
    axs[-1, 0].set_xlabel(f"t, {s.t_unit}")
    axs[-1, 1].set_xlabel(f"f, {s.f_unit}")
    save(fig, LAB, name)


def exp_inverse_df(s, dfs, res):
    """Влияние шага по частоте Δf на восстановленный сигнал (периодизация с периодом 1/Δf)."""
    tt = np.arange(-0.4, 1.6, s.dt)            # шире исходного диапазона [0, 0.6] мс
    fig, axs = plt.subplots(len(dfs), 1, figsize=(7.2, 1.5 * len(dfs) + 0.3), sharex=True)
    res["inverse_df"] = {}
    for ax, df in zip(axs, dfs):
        f = np.arange(0, s.fs / 2, df)
        P = forward_ft(s.t, s.x, f, "rect")
        ax.plot(s.t, s.x, "k", lw=1.0, label="исходный")
        for m, st in (("rect", "r-"), ("spline", "b--")):
            rec = inverse_ft(f, P, tt, m)
            ax.plot(tt, rec, st, lw=0.6, label=METH[m])
            m0 = (tt >= 0) & (tt <= s.t[-1] + 1e-9)
            res["inverse_df"].setdefault(m, []).append(rel_err(rec[m0][:s.n], s.x))
        ax.set_title(f"Δf = {df:g} {s.f_unit}: период повторения 1/Δf = {1 / df:.3g} {s.t_unit}", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right", ncol=3)
    axs[-1].set_xlabel(f"t, {s.t_unit}")
    res["inverse_df"]["df"] = list(dfs)
    save(fig, LAB, "fig_inv_df.png")


def exp_inverse_F(s, Fs, res):
    """Влияние верхней границы частотного диапазона F."""
    fig, axs = plt.subplots(len(Fs), 1, figsize=(7.2, 1.45 * len(Fs) + 0.3), sharex=True)
    res["inverse_F"] = {}
    for ax, F in zip(axs, Fs):
        f = freq_grid(s, fmax=F)
        P = forward_ft(s.t, s.x, f, "rect")
        rec = inverse_ft(f, P, s.t, "rect")
        e = rel_err(rec, s.x)
        res["inverse_F"][str(F)] = e
        ax.plot(s.t, s.x, "k", lw=0.9, label="исходный")
        ax.plot(s.t, rec, "r--", lw=0.8, label="восстановленный")
        ax.set_title(f"F = {F:g} {s.f_unit}: относительная ошибка {e:.2e}", fontsize=8)
        ax.set_xlim(0.18, 0.30)
        ax.legend(fontsize=6.5, loc="upper right")
    axs[-1].set_xlabel(f"t, {s.t_unit}")
    save(fig, LAB, "fig_inv_F.png")


def exp_analytic(res):
    """Тест с известным точным спектром: гауссов радиоимпульс."""
    t0, sig, f0 = 0.15, 0.02, 100.0          # мс, мс, кГц
    w0 = 2 * np.pi * f0
    p = lambda t: np.exp(-((t - t0) / sig) ** 2) * np.cos(w0 * (t - t0))  # noqa: E731

    def P_ex(f):
        w = 2 * np.pi * f
        return (sig * np.sqrt(np.pi) / 2 * np.exp(-1j * w * t0)
                * (np.exp(-sig ** 2 * (w - w0) ** 2 / 4) + np.exp(-sig ** 2 * (w + w0) ** 2 / 4)))

    f = np.linspace(0, 250, 1001)
    Pe = P_ex(f)
    dts = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]) * 1e-3   # мс
    err = {"rect": [], "spline": []}
    for dt in dts:
        t = np.arange(0, 0.3 + dt / 2, dt)
        for m in err:
            err[m].append(rel_err(forward_ft(t, p(t), f, m), Pe))
    # обратное ПФ по точному спектру: влияние шага Δf
    tt = np.arange(0, 0.3, 0.5e-3)
    dfs = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.3, 4.0, 5.0])
    erri = {"rect": [], "spline": []}
    for df in dfs:
        ff = np.arange(0, 400 + df / 2, df)
        for m in erri:
            erri[m].append(rel_err(inverse_ft(ff, P_ex(ff), tt, m), p(tt)))

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    tp = np.linspace(0, 0.3, 1500)
    axs[0].plot(tp, p(tp), lw=0.8)
    axs[0].set_title("тестовый сигнал")
    axs[0].set_xlabel("t, мс")
    ax = axs[1]
    for m in err:
        ax.semilogy(dts * 1e3, err[m], "o-", ms=3, label=METH[m])
    ax.axvline(1e3 / (2 * 200), color="gray", ls=":", lw=0.8)
    ax.set_title("прямое ПФ: ошибка от Δt")
    ax.set_xlabel("Δt, мкс")
    ax.set_ylabel("отн. ошибка P(f)")
    ax.legend()
    ax = axs[2]
    for m in erri:
        ax.semilogy(dfs, erri[m], "o-", ms=3, label=METH[m])
    ax.axvline(1 / 0.3, color="gray", ls=":", lw=0.8)
    ax.set_title("обратное ПФ: ошибка от Δf")
    ax.set_xlabel("Δf, кГц")
    ax.set_ylabel("отн. ошибка p(t)")
    ax.legend()
    save(fig, LAB, "fig_analytic_test.png")
    res["analytic"] = dict(dt_us=(dts * 1e3).tolist(), fwd=err, df=dfs.tolist(), inv=erri)


def main():
    sigs = load_signals(sys.argv[1:] or None)
    res = {}
    print("Обзор сигналов, спектров и восстановления")
    exp_overview(sigs, res)
    s1 = next((s for s in sigs if s.t_unit == "мс"), sigs[0])
    sa = next((s for s in sigs if s.t_unit == "с"), sigs[-1])
    res["s1"], res["sa"] = s1.name, sa.name
    print("Влияние шага по времени")
    exp_dt(s1, (1, 2, 4, 8), 1000.0, "fig_dt_signal.png", res)
    seg = sa.crop(0.4)                       # первая гласная / нота: 0.4 с
    exp_dt(seg, (1, 2, 4), 4000.0, "fig_dt_audio.png", res)
    print("Влияние верхней границы времени")
    exp_T(s1, (0.12, 0.3, 0.6), 400.0, "fig_T_signal.png", f_lim=(0, 300))
    exp_T(sa, (0.01, 0.03, 0.1, 0.4), 1500.0, "fig_T_audio.png", df=0.5)
    print("Обратное ПФ: шаг и верхняя граница по частоте")
    exp_inverse_df(s1, (0.5, 1.25, 2.5, 4.0), res)
    exp_inverse_F(s1, (80.0, 120.0, 160.0, 1000.0), res)
    print("Тест с известным спектром")
    exp_analytic(res)
    save_results(LAB, res)


if __name__ == "__main__":
    main()
