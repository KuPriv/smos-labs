"""Лабораторная работа № 3: простейшие цифровые фильтры (нерекурсивные, КИХ).

1) «Скользящее среднее» шириной M отсчётов.
2) Приближение идеального ФНЧ фильтрами Баттерворта, Чебышева I рода и эллиптическим:
   АЧХ аналогового фильтра-прототипа |K(f)| (частота среза f_c, порядок n) реализуется
   КИХ-фильтром методом частотной выборки с окном (линейная ФЧХ, без фазовых искажений).
   Для сравнения – классические рекурсивные (БИХ) реализации scipy.signal (билинейное преобразование).
3) Полосовой фильтр с косинусоидальным сглаживанием АЧХ (Сергиенко, формулы (6.15), (6.16)):
   h_bp = h_lp(f2) - h_lp(f1), начальная и конечная частоты полосы f1, f2.

Запуск:  python lab3/lab3.py [файл1 файл2 ...]
"""
import os
import sys

import numpy as np
from scipy import signal as ss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.plotting import plt, save, save_results  # noqa: E402
from common.signals import Signal, load_signals  # noqa: E402

LAB = os.path.dirname(os.path.abspath(__file__))
NAMES = {"butter": "Баттерворт", "cheby1": "Чебышев I", "ellip": "эллиптический"}
RP, RS = 1.0, 40.0         # пульсации в полосе пропускания (дБ) и затухание в полосе задерживания (дБ)


# -------------------------------------------------------------------------------
# 1. Скользящее среднее
# -------------------------------------------------------------------------------
def moving_average(x, M):
    """y_k = (1/M) Σ_{m=-(M-1)/2}^{(M-1)/2} x_{k+m} – центрированное (нулевая фаза) скользящее среднее.
    Реализация через кумулятивную сумму: O(N) независимо от M. Края дополняются крайними значениями."""
    x = np.asarray(x, float)
    left, right = (M - 1) // 2, M // 2
    xp = np.concatenate([np.full(left, x[0]), x, np.full(right, x[-1])])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    return (c[M:] - c[:-M]) / M


def ma_response(f, M, dt):
    """|H(f)| = |sin(π f M Δt) / (M sin(π f Δt))|."""
    z = np.pi * f * dt
    with np.errstate(all="ignore"):
        H = np.where(np.abs(np.sin(z)) < 1e-12, 1.0, np.sin(M * z) / (M * np.sin(z)))
    return np.abs(H)


# -------------------------------------------------------------------------------
# 2. ФНЧ: АЧХ прототипов и КИХ-реализация
# -------------------------------------------------------------------------------
def chebyshev_poly(n, x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    m = np.abs(x) <= 1
    out[m] = np.cos(n * np.arccos(x[m]))
    out[~m] = np.cosh(n * np.arccosh(np.abs(x[~m]))) * np.sign(x[~m]) ** n
    return out


def prototype_response(kind, f, fc, n):
    """АЧХ аналогового ФНЧ-прототипа |K(f)|:
       Баттерворт:  1/sqrt(1 + (f/fc)^{2n});
       Чебышев I:   1/sqrt(1 + ε² T_n²(f/fc)),  ε² = 10^{Rp/10} - 1;
       эллиптический: 1/sqrt(1 + ε² R_n²(f/fc)) – рассчитывается по нулям/полюсам scipy (analog=True)."""
    f = np.abs(np.asarray(f, float))
    if kind == "butter":
        return 1 / np.sqrt(1 + (f / fc) ** (2 * n))
    if kind == "cheby1":
        eps2 = 10 ** (RP / 10) - 1
        return 1 / np.sqrt(1 + eps2 * chebyshev_poly(n, f / fc) ** 2)
    if kind == "ellip":
        z, p, k = ss.ellip(n, RP, RS, 2 * np.pi * fc, analog=True, output="zpk")
        _, h = ss.freqs_zpk(z, p, k, worN=2 * np.pi * np.maximum(f, 1e-12))
        return np.abs(h)
    raise ValueError(kind)


def fir_from_response(resp, dt, L=255, nfft=1 << 15, window="hamming"):
    """КИХ-фильтр длины L (нечётной) по заданной АЧХ resp(f) (метод частотной выборки + окно):
    1) АЧХ задаётся на сетке f_k = k/(nfft Δt) (нулевая фаза => h симметрична);
    2) h = ОБПФ, циклический сдвиг к центру, усечение до L отсчётов;
    3) умножение на окно (сглаживает эффект Гиббса от усечения)."""
    f = np.fft.fftfreq(nfft, dt)
    h = np.fft.ifft(resp(f)).real
    h = np.roll(h, L // 2)[:L]
    return h * ss.get_window(window, L)


def apply_fir(x, h):
    """Свёртка с симметричной импульсной характеристикой, центрированная (без задержки)."""
    return np.convolve(x, h, mode="same")


def fir_response(h, f, dt):
    """АЧХ КИХ-фильтра |H(f)| = |Σ h_k e^{-i2πf kΔt}|."""
    k = np.arange(h.size)
    return np.abs(np.exp(-2j * np.pi * np.outer(f, k) * dt) @ h)


def lowpass(x, dt, kind, fc, n, L=255):
    h = fir_from_response(lambda f: prototype_response(kind, f, fc, n), dt, L)
    return apply_fir(x, h), h


# -------------------------------------------------------------------------------
# 3. Полосовой фильтр с косинусоидальным сглаживанием АЧХ
# -------------------------------------------------------------------------------
def raised_cosine_lp_h(t, f0, alpha):
    """Импульсная характеристика ФНЧ с косинусоидальным сглаживанием (6.16):
    h(t) = (ω0/π) · sin(ω0 t)/(ω0 t) · cos(α ω0 t) / (1 - (2 α ω0 t / π)^2),
    в точках α ω0 t = ±π/2 – предел h = (ω0 α / 2π) sin(π / 2α)."""
    w0 = 2 * np.pi * f0
    t = np.asarray(t, float)
    den = 1 - (2 * alpha * w0 * t / np.pi) ** 2
    sing = np.abs(den) < 1e-10
    h = np.empty_like(t)
    tt = t[~sing]
    h[~sing] = w0 / np.pi * np.sinc(w0 * tt / np.pi) * np.cos(alpha * w0 * tt) / den[~sing]
    h[sing] = w0 * alpha / (2 * np.pi) * np.sin(np.pi / (2 * alpha))
    return h


def raised_cosine_lp_K(f, f0, alpha):
    """АЧХ (6.15): 1 при |f| ≤ f0(1-α); 1/2 - 1/2 sin(π(|f|-f0)/(2αf0)) в переходной зоне; 0 далее."""
    f = np.abs(np.asarray(f, float))
    K = np.zeros_like(f)
    K[f <= f0 * (1 - alpha)] = 1
    m = (f > f0 * (1 - alpha)) & (f <= f0 * (1 + alpha))
    K[m] = 0.5 - 0.5 * np.sin(np.pi * (f[m] - f0) / (2 * alpha * f0))
    return K


def bandpass_raised_cosine(dt, f1, f2, df_tr=None, L=401, window=None):
    """Полосовой КИХ-фильтр [f1, f2]: разность двух ФНЧ с косинусоидальным сглаживанием.
    Ширина переходной зоны 2·df_tr одинакова на обоих краях (α_i = df_tr / f_i).
    h_k = Δt · [h_lp(kΔt; f2) - h_lp(kΔt; f1)],  k = -(L-1)/2 .. (L-1)/2."""
    df_tr = 0.15 * (f2 - f1) if df_tr is None else df_tr
    a1, a2 = df_tr / f1, df_tr / f2
    k = np.arange(L) - L // 2
    t = k * dt
    h = dt * (raised_cosine_lp_h(t, f2, a2) - raised_cosine_lp_h(t, f1, a1))
    if window:
        h *= ss.get_window(window, L)
    ideal = lambda f: raised_cosine_lp_K(f, f2, a2) - raised_cosine_lp_K(f, f1, a1)  # noqa: E731
    return h, ideal


# -------------------------------------------------------------------------------
def spec(x, dt):
    f = np.fft.rfftfreq(x.size, dt)
    return f, np.abs(np.fft.rfft(x)) * dt


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-8))


def plot_compare(sig, outs, fname, title, f_lim, t_lim=None, resp=None):
    """Сигналы (исходный + отфильтрованные), их спектры и (опционально) АЧХ фильтров."""
    ncol = 3 if resp else 2
    fig, axs = plt.subplots(len(outs), ncol, figsize=(7.4, 1.9 * len(outs) + 0.3), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.25, 1, 1][:ncol]))
    f0, S0 = spec(sig.x, sig.dt)
    for r, (lab, y) in enumerate(outs.items()):
        ax = axs[r, 0]
        m = slice(None) if t_lim is None else (sig.t >= t_lim[0]) & (sig.t <= t_lim[1])
        ax.plot(sig.t[m], sig.x[m], color="0.6", lw=0.7, label="исходный")
        ax.plot(sig.t[m], y[m], "r", lw=0.8, label="отфильтр.")
        ax.set_title(lab, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax = axs[r, 1]
        f, S = spec(y, sig.dt)
        ax.plot(f0, S0, color="0.6", lw=0.7, label="исходный")
        ax.plot(f, S, "r", lw=0.8, label="отфильтр.")
        ax.set_xlim(*f_lim)
        ax.set_title("модуль спектра", fontsize=8)
        if resp:
            ax = axs[r, 2]
            ff = np.linspace(*f_lim, 1500)
            ax.plot(ff, resp[lab](ff), "b", lw=0.9)
            ax.set_title("АЧХ фильтра", fontsize=8)
            ax.set_ylim(-0.05, 1.15)
    axs[-1, 0].set_xlabel(f"t, {sig.t_unit}")
    for c in range(1, ncol):
        axs[-1, c].set_xlabel(f"f, {sig.f_unit}")
    fig.suptitle(title, fontsize=9)
    save(fig, LAB, fname)


def main():
    sigs = load_signals(sys.argv[1:] or None)
    s1 = next((s for s in sigs if s.name == "signal1"), sigs[0])
    s2 = next((s for s in sigs if s.name == "signal2"), sigs[1])
    sv = next((s for s in sigs if s.t_unit == "с"), sigs[-1])
    res = {}

    # ---------------- 1. скользящее среднее ----------------
    rng = np.random.default_rng(7)
    s1n = Signal("signal1 + шум", s1.t, s1.x + 0.15 * rng.standard_normal(s1.n), s1.t_unit, s1.f_unit)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    f = np.linspace(0, s1.fs / 2, 2000)
    for M in (3, 5, 11, 21):
        ax.plot(f, ma_response(f, M, s1.dt), lw=0.9, label=f"M = {M}: первый ноль {s1.fs / M:.0f} кГц")
    ax.axvspan(60, 140, color="orange", alpha=0.15, label="полоса signal1")
    ax.set_xlabel("f, кГц")
    ax.set_ylabel("|H(f)|")
    ax.set_title("АЧХ фильтра «скользящее среднее» (Δt = 0,5 мкс)")
    ax.legend(fontsize=7)
    save(fig, LAB, "fig_ma_response.png")
    outs = {f"M = {M}": moving_average(s1n.x, M) for M in (5, 11, 21)}
    plot_compare(s1n, outs, "fig_ma_signal1.png", "Скользящее среднее: signal1 с добавленным шумом",
                 (0, 600), t_lim=(0.18, 0.30),
                 resp={k: (lambda ff, M=int(k.split()[-1]): ma_response(ff, M, s1.dt)) for k in outs})
    res["ma"] = {}
    for M in (3, 5, 7, 11, 15, 21, 31):
        y = moving_average(s1n.x, M)
        res["ma"][str(M)] = float(np.linalg.norm(y - s1.x) / np.linalg.norm(s1.x))
    res["ma_noisy"] = float(np.linalg.norm(s1n.x - s1.x) / np.linalg.norm(s1.x))
    seg = sv.crop(0.4)
    outs = {f"M = {M}": moving_average(seg.x, M) for M in (3, 9)}
    plot_compare(seg, outs, "fig_ma_voice.png", "Скользящее среднее: голос (фрагмент «а»)", (0, 4000),
                 t_lim=(0.20, 0.23),
                 resp={k: (lambda ff, M=int(k.split()[-1]): ma_response(ff, M, seg.dt)) for k in outs})

    # ---------------- 2. ФНЧ ----------------
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    fc = 100.0
    ff = np.linspace(0, 400, 2000)
    for c, kind in enumerate(NAMES):
        for n in (2, 4, 8):
            K = prototype_response(kind, ff, fc, n)
            h = fir_from_response(lambda x: prototype_response(kind, x, fc, n), s2.dt)
            H = fir_response(h, ff, s2.dt)
            axs[0, c].plot(ff, K, lw=0.9, label=f"n = {n}")
            axs[1, c].plot(ff, db(K), lw=0.9, label=f"n = {n}, прототип")
            axs[1, c].plot(ff, db(H), "k:", lw=0.7)
        axs[0, c].plot(ff, ff <= fc, color="0.5", ls="--", lw=0.7, label="идеальный")
        axs[0, c].set_title(f"{NAMES[kind]}, f_c = {fc:g} кГц", fontsize=8)
        axs[1, c].set_ylim(-90, 5)
        axs[1, c].set_xlabel("f, кГц")
        axs[0, c].legend(fontsize=6)
    axs[0, 0].set_ylabel("|K(f)|")
    axs[1, 0].set_ylabel("|K(f)|, дБ")
    axs[1, 0].text(5, -85, "пунктир – КИХ-реализация (L = 255)", fontsize=6)
    save(fig, LAB, "fig_lpf_responses.png")

    fc, n = 80.0, 6
    outs, resp = {}, {}
    res["lpf"] = {}
    for kind in NAMES:
        y, h = lowpass(s2.x, s2.dt, kind, fc, n)
        lab = f"{NAMES[kind]}, n = {n}, f_c = {fc:g} кГц"
        outs[lab] = y
        resp[lab] = lambda x, h=h: fir_response(h, x, s2.dt)
        fq, S = spec(y, s2.dt)
        _, S0 = spec(s2.x, s2.dt)
        b = (fq > 110) & (fq < 130)
        res["lpf"][kind] = float(S[b].max() / S0[b].max())       # ослабление пика 120 кГц
    plot_compare(s2, outs, "fig_lpf_signal2.png", "ФНЧ: signal2 (компоненты 45, 120, 250, 310 кГц)",
                 (0, 450), resp=resp)

    outs, resp = {}, {}
    for fcv in (60.0, 180.0, 280.0):
        y, h = lowpass(s2.x, s2.dt, "butter", fcv, 6)
        lab = f"Баттерворт, n = 6, f_c = {fcv:g} кГц"
        outs[lab], resp[lab] = y, (lambda x, h=h: fir_response(h, x, s2.dt))
    plot_compare(s2, outs, "fig_lpf_fc.png", "Влияние частоты среза", (0, 450), resp=resp)

    outs, resp = {}, {}
    for n in (1, 3, 10):
        y, h = lowpass(s2.x, s2.dt, "cheby1", 80.0, n)
        lab = f"Чебышев I, n = {n}, f_c = 80 кГц"
        outs[lab], resp[lab] = y, (lambda x, h=h: fir_response(h, x, s2.dt))
    plot_compare(s2, outs, "fig_lpf_order.png", "Влияние порядка фильтра", (0, 450), resp=resp)

    outs, resp = {}, {}
    for kind in NAMES:
        y, h = lowpass(seg.x, seg.dt, kind, 600.0, 6, L=401)
        lab = f"{NAMES[kind]}, n = 6, f_c = 600 Гц"
        outs[lab], resp[lab] = y, (lambda x, h=h: fir_response(h, x, seg.dt))
    plot_compare(seg, outs, "fig_lpf_voice.png", "ФНЧ: голос (фрагмент «а»)", (0, 2500),
                 t_lim=(0.20, 0.23), resp=resp)

    # КИХ (линейная ФЧХ) против БИХ (scipy, билинейное преобразование)
    fc, n = 150.0, 6
    y_fir, _ = lowpass(s1n.x, s1.dt, "butter", fc, n)
    sos = ss.butter(n, fc, fs=s1.fs, output="sos")
    y_iir = ss.sosfilt(sos, s1n.x)
    y_iir2 = ss.sosfiltfilt(sos, s1n.x)
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6))
    m = (s1.t > 0.19) & (s1.t < 0.27)
    axs[0].plot(s1.t[m], s1.x[m], "k", lw=1.4, label="сигнал без шума")
    axs[0].plot(s1.t[m], y_fir[m], "r", lw=0.9, label="КИХ (наша реализация, линейная ФЧХ)")
    axs[0].plot(s1.t[m], y_iir[m], "b", lw=0.9, label="БИХ scipy.sosfilt (каузальный)")
    axs[0].plot(s1.t[m], y_iir2[m], "g--", lw=0.9, label="БИХ scipy.sosfiltfilt (прямой+обратный проход)")
    axs[0].set_title(f"Баттерворт n = {n}, f_c = {fc:g} кГц: signal1 + шум", fontsize=8)
    axs[0].set_xlabel("t, мс")
    axs[0].legend(fontsize=6.5)
    w, hh = ss.sosfreqz(sos, worN=4000, fs=s1.fs)
    gd = -np.gradient(np.unwrap(np.angle(hh)), 2 * np.pi * w) * 1e3   # мкс
    axs[1].plot(w, gd, "b", lw=0.9, label="групповая задержка БИХ")
    axs[1].axhline(0, color="r", lw=0.9, label="КИХ (центрированный) – 0")
    axs[1].set_xlim(0, 300)
    axs[1].set_ylim(-1, 25)
    axs[1].set_xlabel("f, кГц")
    axs[1].set_ylabel("задержка, мкс")
    axs[1].legend(fontsize=7)
    save(fig, LAB, "fig_fir_vs_iir.png")
    i0 = np.argmax(np.correlate(y_iir, y_fir, "full")) - (s1.n - 1)
    res["iir_delay_us"] = float(i0 * s1.dt * 1e3)

    # ---------------- 3. полосовой фильтр с косинусоидальным сглаживанием ----------------
    f1, f2 = 100.0, 140.0
    ff = np.linspace(0, 300, 3000)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for L in (51, 201, 801):
        h, ideal = bandpass_raised_cosine(s2.dt, f1, f2, L=L)
        axs[0].plot(ff, fir_response(h, ff, s2.dt), lw=0.8, label=f"L = {L}")
    axs[0].plot(ff, ideal(ff), "k--", lw=0.8, label="(6.15)")
    axs[0].set_title(f"АЧХ, полоса {f1:g}–{f2:g} кГц", fontsize=8)
    axs[0].set_xlabel("f, кГц")
    axs[0].legend(fontsize=6)
    for dtr in (2.0, 6.0, 15.0):
        h, ideal = bandpass_raised_cosine(s2.dt, f1, f2, df_tr=dtr, L=401)
        axs[1].plot(ff, db(fir_response(h, ff, s2.dt)), lw=0.8, label=f"Δf_пер = {dtr:g} кГц")
    axs[1].set_ylim(-80, 5)
    axs[1].set_title("АЧХ, дБ (L = 401)", fontsize=8)
    axs[1].set_xlabel("f, кГц")
    axs[1].legend(fontsize=6)
    h, _ = bandpass_raised_cosine(s2.dt, f1, f2, L=401)
    k = (np.arange(401) - 200) * s2.dt * 1e3
    axs[2].plot(k, h, lw=0.8)
    axs[2].set_title("импульсная характеристика", fontsize=8)
    axs[2].set_xlabel("t, мкс")
    save(fig, LAB, "fig_bpf_response.png")

    outs, resp = {}, {}
    res["bpf"] = {}
    for (a, b) in ((30.0, 60.0), (100.0, 140.0), (280.0, 340.0)):
        h, ideal = bandpass_raised_cosine(s2.dt, a, b, L=401)
        lab = f"полоса {a:g}–{b:g} кГц"
        outs[lab] = apply_fir(s2.x, h)
        resp[lab] = lambda x, h=h: fir_response(h, x, s2.dt)
        res["bpf"][lab] = float(np.max(np.abs(fir_response(h, np.linspace(a + 0.15 * (b - a), b - 0.15 * (b - a), 200), s2.dt) - 1)))
    plot_compare(s2, outs, "fig_bpf_signal2.png",
                 "Полосовая фильтрация signal2 (косинусоидальное сглаживание АЧХ)", (0, 450), resp=resp)
    outs, resp = {}, {}
    for (a, b) in ((100.0, 400.0), (500.0, 900.0), (1000.0, 1600.0)):
        h, ideal = bandpass_raised_cosine(seg.dt, a, b, L=801)
        lab = f"полоса {a:g}–{b:g} Гц"
        outs[lab] = apply_fir(seg.x, h)
        resp[lab] = lambda x, h=h: fir_response(h, x, seg.dt)
    plot_compare(seg, outs, "fig_bpf_voice.png", "Полосовая фильтрация голоса (фрагмент «а»)", (0, 2500),
                 t_lim=(0.20, 0.23), resp=resp)
    save_results(LAB, res)


if __name__ == "__main__":
    main()
