"""Лабораторная работа № 2: дискретное преобразование Фурье (ДПФ) и БПФ.

Реализованы:
  * прямое и обратное ДПФ по определению (Theory_Lab2, формулы (1), (2)), O(N^2);
  * собственные БПФ по основанию 2: с прореживанием по времени (DIT) и по частоте (DIF), O(N log N);
    обратное БПФ – через сопряжение спектра и результата (лекции 2-3);
  * сравнение с библиотечным БПФ (numpy.fft) и с непрерывным ПФ из ЛР1 (common/ctft.py).
Анализ: совпадение спектров, восстановление сигнала, быстродействие в зависимости от N,
влияние числа отсчётов и дополнения нулями.

Запуск:  python lab2/lab2.py [файл1 файл2 ...]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ctft import forward_ft, inverse_ft  # noqa: E402
from common.plotting import plt, save, save_results  # noqa: E402
from common.signals import load_signals  # noqa: E402

LAB = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------------
# ДПФ по определению
# ---------------------------------------------------------------------------------
def dft(p):
    """P(k) = Σ_{n=0}^{N-1} p(n) e^{-i 2π k n / N},  k = 0..N-1  (матрица по блокам строк)."""
    p = np.asarray(p, complex)
    N = p.size
    n = np.arange(N)
    P = np.empty(N, complex)
    step = max(1, 4_000_000 // N)
    for s in range(0, N, step):
        k = n[s:s + step, None]
        P[s:s + step] = np.exp(-2j * np.pi * ((k * n) % N) / N) @ p   # (k n mod N) – точнее при больших N
    return P


def idft(P):
    """p(n) = (1/N) Σ_k P(k) e^{+i 2π k n / N}.
    (В формуле (2) Theory_Lab2 множитель 1/N пропущен – без него p увеличивается в N раз.)"""
    P = np.asarray(P, complex)
    N = P.size
    n = np.arange(N)
    p = np.empty(N, complex)
    step = max(1, 4_000_000 // N)
    for s in range(0, N, step):
        k = n[s:s + step, None]
        p[s:s + step] = np.exp(2j * np.pi * ((k * n) % N) / N) @ P
    return p / N


# ---------------------------------------------------------------------------------
# Собственные реализации БПФ (N = 2^l)
# ---------------------------------------------------------------------------------
def bit_reverse_permutation(N):
    l = N.bit_length() - 1
    idx = np.arange(N)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(l):
        rev |= ((idx >> b) & 1) << (l - 1 - b)
    return rev


def fft_dit(p):
    """БПФ с прореживанием по времени (Кули–Тьюки), итеративный вариант.
    1) отсчёты переставляются в бит-реверсном порядке;
    2) log2 N этапов «бабочек»: X = A + W·B, Y = A - W·B, W = e^{-i2πk/m}.
    Бабочки одного этапа выполняются векторно (NumPy)."""
    x = np.asarray(p, complex)
    N = x.size
    if N & (N - 1):
        raise ValueError("N должно быть степенью двойки")
    x = x[bit_reverse_permutation(N)].copy()
    m = 2
    while m <= N:
        half = m // 2
        W = np.exp(-2j * np.pi * np.arange(half) / m)       # поворачивающие множители
        x = x.reshape(-1, m)                                 # N/m блоков длины m
        a = x[:, :half].copy()
        b = x[:, half:] * W
        x[:, :half] = a + b
        x[:, half:] = a - b
        x = x.ravel()
        m *= 2
    return x


def fft_dif(p):
    """БПФ с прореживанием по частоте (Сэнде–Тьюки): на каждом этапе
    a' = a + b, b' = (a - b)·W; в конце – бит-реверсная перестановка результата."""
    x = np.asarray(p, complex).copy()
    N = x.size
    if N & (N - 1):
        raise ValueError("N должно быть степенью двойки")
    m = N
    while m >= 2:
        half = m // 2
        W = np.exp(-2j * np.pi * np.arange(half) / m)
        x = x.reshape(-1, m)
        a = x[:, :half].copy()
        b = x[:, half:].copy()
        x[:, :half] = a + b
        x[:, half:] = (a - b) * W
        x = x.ravel()
        m //= 2
    return x[bit_reverse_permutation(N)]


def ifft_via_fft(P, fft=fft_dit):
    """Обратное БПФ через прямое: p = conj(FFT(conj(P))) / N."""
    P = np.asarray(P, complex)
    return np.conj(fft(np.conj(P))) / P.size


def fft_recursive(p):
    """Рекурсивный вариант DIT (для пояснения идеи): X_k = E_k + W^k O_k, X_{k+N/2} = E_k - W^k O_k."""
    p = np.asarray(p, complex)
    N = p.size
    if N == 1:
        return p
    E, O = fft_recursive(p[0::2]), fft_recursive(p[1::2])
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([E + W * O, E - W * O])


# ---------------------------------------------------------------------------------
def best_time(fn, *a, rep=3, budget=2.0):
    t_best = np.inf
    t0 = time.perf_counter()
    for _ in range(rep):
        s = time.perf_counter()
        fn(*a)
        t_best = min(t_best, time.perf_counter() - s)
        if time.perf_counter() - t0 > budget:
            break
    return t_best


def exp_compare(sig, res, fname, f_lim=None, zoom=None):
    """Спектр: непрерывное ПФ (ЛР1), ДПФ, БПФ; восстановление: обратное ПФ, ОДПФ, ОБПФ."""
    N = sig.n
    Np2 = 1 << (N - 1).bit_length()                 # для собственного БПФ – дополнение нулями до 2^l
    fk = np.arange(N) / (N * sig.dt)                # частоты ДПФ f_k = k / (N Δt)
    half = N // 2 + 1
    t0 = sig.t - sig.t[0]
    P_cont = forward_ft(t0, sig.x, fk[:half], "rect")
    P_spl = forward_ft(t0, sig.x, fk[:half], "spline")
    P_dft = dft(sig.x) * sig.dt
    P_np = np.fft.fft(sig.x) * sig.dt
    P_own = fft_dit(np.r_[sig.x, np.zeros(Np2 - N)]) * sig.dt
    fk2 = np.arange(Np2) / (Np2 * sig.dt)
    h2 = Np2 // 2 + 1
    P_cont2 = forward_ft(t0, sig.x, fk2[:h2], "rect")   # непрерывное ПФ на сетке БПФ с нулями
    d = dict(dft_vs_cont=float(np.abs(P_dft[:half] - P_cont).max() / np.abs(P_cont).max()),
             np_vs_dft=float(np.abs(P_np - P_dft).max() / np.abs(P_dft).max()),
             own_vs_np=float(np.abs(P_own - np.fft.fft(np.r_[sig.x, np.zeros(Np2 - N)]) * sig.dt).max()
                             / np.abs(P_own).max()),
             own_vs_cont=float(np.abs(P_own[:h2] - P_cont2).max() / np.abs(P_cont2).max()),
             N=N, Np2=Np2)
    # восстановление
    rec_cont = inverse_ft(fk[:half], P_cont, t0, "rect")
    rec_idft = idft(P_dft / sig.dt).real
    rec_ifft = ifft_via_fft(P_own / sig.dt).real[:N]
    rel = lambda a: float(np.linalg.norm(a - sig.x) / np.linalg.norm(sig.x))  # noqa: E731
    d.update(err_cont=rel(rec_cont), err_idft=rel(rec_idft), err_ifft=rel(rec_ifft))
    res.setdefault("compare", {})[sig.name] = d

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.4))
    axs[0].plot(sig.t, sig.x, lw=0.6)
    axs[0].set_title(f"{sig.name}: сигнал (N = {N})")
    axs[0].set_xlabel(f"t, {sig.t_unit}")
    ax = axs[1]
    ax.plot(fk2[:h2], np.abs(P_cont2), "k-", lw=1.6, label="непрерывное ПФ (прямоугольники)")
    ax.plot(fk[:half], np.abs(P_spl), "g-", lw=0.8, label="непрерывное ПФ («шапочки»)")
    ax.plot(fk[:half], np.abs(P_dft[:half]), "r--", lw=0.9, label="ДПФ·Δt")
    ax.plot(fk[:half], np.abs(P_np[:half]), "c:", lw=1.2, label="БПФ numpy·Δt")
    ax.plot(fk2[:h2], np.abs(P_own[:h2]), "m.", ms=1.5,
            label=f"собств. БПФ·Δt (N→{Np2})")
    ax.set_title("модуль спектра")
    ax.set_xlabel(f"f, {sig.f_unit}")
    if f_lim:
        ax.set_xlim(*f_lim)
    ax.legend(fontsize=6.5)
    ax = axs[2]
    sl = slice(None) if zoom is None else (sig.t >= zoom[0]) & (sig.t <= zoom[1])
    ax.plot(sig.t[sl], sig.x[sl], "k.-", lw=1.0, ms=3, label="исходный")
    ax.plot(sig.t[sl], rec_cont[sl], "g-", lw=0.8, label=f"обратное ПФ ({d['err_cont']:.1e})")
    ax.plot(sig.t[sl], rec_idft[sl], "r--", lw=0.9, label=f"ОДПФ ({d['err_idft']:.1e})")
    ax.plot(sig.t[sl], rec_ifft[sl], "c:", lw=1.3, label=f"ОБПФ ({d['err_ifft']:.1e})")
    ax.set_title("восстановленный сигнал (фрагмент), в скобках – отн. ошибка")
    ax.set_xlabel(f"t, {sig.t_unit}")
    ax.legend(fontsize=6.5, loc="upper right")
    save(fig, LAB, fname)


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def exp_timing_all(res):
    rng = np.random.default_rng(1)
    algs = [
        ("непрерывное ПФ (ЛР1), M = N/2+1",
         lambda x: forward_ft(np.arange(x.size), x, np.arange(x.size // 2 + 1) / x.size), 2 ** 13),
        ("ДПФ по определению", dft, 2 ** 13),
        ("БПФ рекурсивное (собств.)", fft_recursive, 2 ** 15),
        ("БПФ DIT (собств.)", fft_dit, 2 ** 20),
        ("БПФ DIF (собств.)", fft_dif, 2 ** 20),
        ("БПФ numpy", np.fft.fft, 2 ** 22),
    ]
    Ns = 2 ** np.arange(5, 23)
    out = {}
    for name, fn, nmax in algs:
        out[name] = {}
        for N in Ns[Ns <= nmax]:
            out[name][int(N)] = best_time(fn, rng.standard_normal(int(N)))
        print(f"  {name}: t(N={nmax}) = {out[name][int(nmax)]:.3g} c")
    # numpy БПФ: N = 2^l, N = 2^l + 1 (составное/простое) и ближайшее простое
    np_kinds = {"2^l": {}, "простое N": {}}
    for N in 2 ** np.arange(8, 21):
        np_kinds["2^l"][int(N)] = best_time(np.fft.fft, rng.standard_normal(int(N)))
        q = int(N) + 1
        while not is_prime(q):
            q += 1
        np_kinds["простое N"][q] = best_time(np.fft.fft, rng.standard_normal(q))

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax = axs[0]
    for name, d in out.items():
        n = np.array(list(d))
        ax.loglog(n, list(d.values()), "o-", ms=3, label=name)
    n = np.array([2 ** 6, 2 ** 13])
    ax.loglog(n, 3e-9 * n ** 2, "k:", lw=0.8, label="~N²")
    n = np.array([2 ** 6, 2 ** 22])
    ax.loglog(n, 2e-8 * n * np.log2(n), "k--", lw=0.8, label="~N log N")
    ax.set_xlabel("N (длина сигнала)")
    ax.set_ylabel("время, с")
    ax.set_title("быстродействие алгоритмов")
    ax.legend(fontsize=6)
    ax = axs[1]
    for name, d in np_kinds.items():
        ax.loglog(list(d), list(d.values()), "o-", ms=3, label=f"numpy, {name}")
    d = out["БПФ DIT (собств.)"]
    ax.loglog(list(d), list(d.values()), "s-", ms=3, label="собств. DIT, 2^l")
    ax.set_xlabel("N")
    ax.set_ylabel("время, с")
    ax.set_title("БПФ: влияние разложимости N")
    ax.legend(fontsize=6.5)
    save(fig, LAB, "fig_timing.png")
    res["timing"] = {k: {str(n): v for n, v in d.items()} for k, d in out.items()}
    res["t_dit_max"] = out["БПФ DIT (собств.)"][2 ** 20]
    res["speedup_8192"] = out["ДПФ по определению"][2 ** 13] / out["БПФ DIT (собств.)"][2 ** 13]
    res["timing_np"] = {k: {str(n): v for n, v in d.items()} for k, d in np_kinds.items()}
    # таблица для отчёта
    cols = [2 ** 8, 2 ** 10, 2 ** 12, 2 ** 13, 2 ** 16, 2 ** 20]
    rows = ["| алгоритм | " + " | ".join(f"N = 2^{int(np.log2(c))}^" for c in cols) + " |",
            "|:--|" + ":-:|" * len(cols)]
    for name, d in out.items():
        cells = [f"{d[c] * 1e3:.3g}" if c in d else "—" for c in cols]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    res["md_timing"] = "\n".join(rows)


def exp_zero_padding(sig, res):
    """Влияние числа отсчётов и дополнения нулями на результат ДПФ."""
    # (а) фрагмент сигнала с пакетом A0, N отсчётов; дополнение нулями до L·N
    seg = sig.crop(0.3)
    m = seg.t >= 0.15
    t, x = seg.t[m] - seg.t[m][0], seg.x[m]
    N = x.size
    f_ref = np.linspace(0, 250, 2001)
    P_ref = np.abs(forward_ft(t, x, f_ref, "rect"))
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for ax, L in zip(axs[0], (1, 8)):
        Lp = L * N
        P = np.abs(np.fft.fft(x, Lp)) * sig.dt
        f = np.arange(Lp) / (Lp * sig.dt)
        ax.plot(f_ref, P_ref, "k-", lw=1.4, label="непрерывное ПФ")
        ax.plot(f[:Lp // 2], P[:Lp // 2], "ro", ms=2.5, mfc="none", label=f"ДПФ, {Lp} отсч.")
        ax.set_xlim(0, 250)
        ax.set_title(f"N = {N}, дополнение нулями до {Lp}: Δf = {1 / (Lp * sig.dt):.2f} {sig.f_unit}", fontsize=8)
        ax.set_xlabel(f"f, {sig.f_unit}")
        ax.legend(fontsize=6.5)
    # (б) два близких тона: разрешение определяется длительностью сигнала, а не нулями
    dt = sig.dt
    f1, f2 = 100.0, 106.0
    ax = axs[1, 0]
    for Nn, st in ((100, "b-"), (400, "g-"), (1600, "r-")):
        tt = np.arange(Nn) * dt
        xx = np.sin(2 * np.pi * f1 * tt) + np.sin(2 * np.pi * f2 * tt)
        Lp = 16384
        P = np.abs(np.fft.fft(xx, Lp)) * dt
        f = np.arange(Lp) / (Lp * dt)
        ax.plot(f[:Lp // 2], P[:Lp // 2] / P.max(), st, lw=0.9,
                label=f"N = {Nn} (T = {Nn * dt * 1e3:.0f} мкс, 1/T = {1 / (Nn * dt):.1f} кГц)")
    ax.set_xlim(80, 126)
    ax.axvline(f1, color="gray", ls=":", lw=0.8)
    ax.axvline(f2, color="gray", ls=":", lw=0.8)
    ax.set_title("тоны 100 и 106 кГц, все дополнены нулями до 16384", fontsize=8)
    ax.set_xlabel("f, кГц")
    ax.legend(fontsize=6)
    # (в) число отсчётов: усечение signal1
    ax = axs[1, 1]
    for Nn, st in ((200, "b.-"), (450, "g.-"), (1200, "r.-")):
        xx = sig.x[:Nn] if Nn <= sig.n else sig.x
        tt0 = sig.t[:Nn]
        P = np.abs(np.fft.fft(xx)) * dt
        f = np.arange(xx.size) / (xx.size * dt)
        ax.plot(f[:xx.size // 2], P[:xx.size // 2], st, lw=0.7, ms=2.5,
                label=f"N = {xx.size}, T = {tt0[-1] + dt:.3g} мс, Δf = {1 / (xx.size * dt):.2f} кГц")
    ax.set_xlim(0, 250)
    ax.set_title("signal1: влияние числа отсчётов N", fontsize=8)
    ax.set_xlabel("f, кГц")
    ax.legend(fontsize=6)
    save(fig, LAB, "fig_zero_padding.png")

    # дополнение нулями не меняет восстановленный сигнал
    Lp = 4 * N
    rec = np.fft.ifft(np.fft.fft(x, Lp)).real
    res["zp"] = dict(N=N, rec_err=float(np.abs(rec[:N] - x).max()), tail=float(np.abs(rec[N:]).max()))


def main():
    sigs = load_signals(sys.argv[1:] or None)
    res = {}
    s1 = next((s for s in sigs if s.t_unit == "мс"), sigs[0])
    sa = next((s for s in sigs if s.t_unit == "с"), sigs[-1])
    res["s1"], res["sa"] = s1.name, sa.name
    print("Проверка собственных БПФ")
    x = np.random.default_rng(0).standard_normal(1024)
    res["check"] = {
        "dit": float(np.abs(fft_dit(x) - np.fft.fft(x)).max()),
        "dif": float(np.abs(fft_dif(x) - np.fft.fft(x)).max()),
        "rec": float(np.abs(fft_recursive(x) - np.fft.fft(x)).max()),
        "dft": float(np.abs(dft(x) - np.fft.fft(x)).max()),
        "ifft": float(np.abs(ifft_via_fft(np.fft.fft(x)) - x).max()),
    }
    print(" ", res["check"])
    print("Сравнение спектров и восстановления")
    exp_compare(s1, res, "fig_compare_signal1.png", f_lim=(0, 300), zoom=(0.2, 0.25))
    for s in sigs:
        if s is not s1 and s.t_unit == "мс":
            exp_compare(s, res, f"fig_compare_{s.name}.png", f_lim=(0, 500), zoom=(0.13, 0.18))
    exp_compare(sa, res, f"fig_compare_{sa.name}.png", f_lim=(0, 4000), zoom=(1.5, 1.52))
    print("Быстродействие")
    exp_timing_all(res)
    print("Дополнение нулями")
    exp_zero_padding(s1, res)
    save_results(LAB, res)


if __name__ == "__main__":
    main()
