"""Генерация тестовых сигналов для ЛР 1-4.

Архива «Signals.zip» в материалах курса нет, поэтому используются синтетические
сигналы, близкие по характеру к рассматриваемым в лекциях:

  signal1.txt – модельная запись датчика при распространении упругих волн Лэмба
                в пластине (зондирующий импульс 5 периодов 100 кГц с окном Ханна;
                пакеты мод S0 и A0 (A0 с дисперсией) и отражение A0 от края);
  signal2.txt – отклик на ударное воздействие: сумма затухающих колебаний
                45, 120 и 310 кГц, включающихся в разные моменты времени,
                + короткий тональный импульс 250 кГц;
  voice.wav   – синтезированная речь: гласные «а-о-у-и-э» (формантный синтез,
                основной тон 110-150 Гц), 8 кГц;
  melody.wav  – синтезированная мелодия из 9 нот с обертонами, 8 кГц.

В текстовых файлах время – в миллисекундах (частота – в кГц), как в Theory_Lab1.
Любой из файлов можно заменить реальной записью (например, своим голосом в .wav).
"""
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2025)


def hann_burst(t, f0, ncycles, t0=0.0):
    """Тональный импульс: ncycles периодов синусоиды f0 с окном Ханна, начало в t0."""
    tau = t - t0
    T = ncycles / f0
    w = np.where((tau >= 0) & (tau <= T), 0.5 * (1 - np.cos(2 * np.pi * tau / T)), 0.0)
    return w * np.sin(2 * np.pi * f0 * tau)


def dispersive(x, dt, tg0, f0, k):
    """Пакет с групповой задержкой tg(f) = tg0 + k*(f0 - f)/f0 (дисперсия)."""
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    # фаза = 2*pi * интеграл tg(f) df
    phase = 2 * np.pi * (tg0 * f + k * (f - f ** 2 / (2 * f0)))
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-1j * phase), n)


def make_signal1():
    dt = 0.5e-3                       # мс  (частота дискретизации 2 МГц)
    t = np.arange(0, 0.6, dt)         # 0 ... 0.6 мс, N = 1200
    f0 = 100.0                        # кГц
    burst = hann_burst(t, f0, 5)
    s0 = 0.35 * dispersive(burst, dt, 0.07, f0, 0.0)
    a0 = 1.00 * dispersive(burst, dt, 0.20, f0, 0.03)
    refl = -0.45 * dispersive(burst, dt, 0.40, f0, 0.06)
    x = s0 + a0 + refl + 0.015 * rng.standard_normal(len(t))
    return t, x


def make_signal2():
    dt = 0.5e-3
    t = np.arange(0, 1.0, dt)         # 0 ... 1 мс, N = 2000
    x = np.zeros_like(t)
    for f, t0, tau, a in [(45.0, 0.05, 0.25, 0.8), (120.0, 0.15, 0.10, 0.6), (310.0, 0.40, 0.04, 0.5)]:
        m = t >= t0
        x[m] += a * np.exp(-(t[m] - t0) / tau) * np.sin(2 * np.pi * f * (t[m] - t0))
    x += 0.7 * hann_burst(t, 250.0, 3, 0.70)
    x += 0.01 * rng.standard_normal(len(t))
    return t, x


def resonator(x, fc, bw, fs):
    """Цифровой резонатор 2-го порядка (формантный фильтр)."""
    r = np.exp(-np.pi * bw / fs)
    th = 2 * np.pi * fc / fs
    a = [1, -2 * r * np.cos(th), r * r]
    return lfilter([1 - r], a, x)


def make_voice(fs=8000):
    vowels = {  # приближённые форманты гласных русского языка, Гц
        "а": (700, 1200, 2500), "о": (500, 900, 2400), "у": (320, 800, 2300),
        "и": (280, 2250, 2900), "э": (500, 1800, 2500)}
    seg, gap = 0.40, 0.08
    out = []
    for i, v in enumerate("аоуиэ"):
        n = int(seg * fs)
        tt = np.arange(n) / fs
        f0 = 150 - 8 * i - 15 * tt / seg + 3 * np.sin(2 * np.pi * 5.5 * tt)   # интонация + вибрато
        phase = np.cumsum(f0) / fs
        pulses = np.diff(np.floor(phase), prepend=0.0)                         # импульсы основного тона
        src = lfilter([1], [1, -0.97], pulses) + 0.02 * rng.standard_normal(n)  # «голосовой источник»
        y = sum(resonator(src, F, 80 + 0.06 * F, fs) * g
                for F, g in zip(vowels[v], (1.0, 0.6, 0.3)))
        y = lfilter([1, -1], [1], y)                                            # излучение губ (убирает пост. составляющую)
        env = np.minimum(1, np.minimum(tt / 0.04, (seg - tt) / 0.06))
        out += [y * env, np.zeros(int(gap * fs))]
    x = np.concatenate(out)
    return x / np.abs(x).max() * 0.9


def make_melody(fs=8000):
    notes = [("C5", .3), ("E5", .3), ("G5", .3), ("A5", .45), ("G5", .3),
             ("E5", .3), ("F5", .3), ("D5", .3), ("C5", .6)]
    semis = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    out = []
    for name, dur in notes:
        f = 440 * 2 ** ((semis[name[0]] + 12 * (int(name[1]) - 4)) / 12)
        n = int(dur * fs)
        tt = np.arange(n) / fs
        y = sum(np.sin(2 * np.pi * k * f * tt) / k ** 1.3 for k in range(1, 6))
        env = np.minimum(tt / 0.01, 1) * np.exp(-3 * tt)
        env *= np.minimum(1, (dur - tt) / 0.02)
        out.append(y * env)
    x = np.concatenate(out)
    return x / np.abs(x).max() * 0.9


def main():
    for name, (t, x) in {"signal1": make_signal1(), "signal2": make_signal2()}.items():
        np.savetxt(os.path.join(HERE, f"{name}.txt"), np.column_stack([t, x]),
                   fmt="%.6e", header="t, мс    p(t)")
    for name, x in {"voice": make_voice(), "melody": make_melody()}.items():
        wavfile.write(os.path.join(HERE, f"{name}.wav"), 8000, (x * 32767).astype(np.int16))
    print("Тестовые сигналы записаны в", HERE)


if __name__ == "__main__":
    main()
