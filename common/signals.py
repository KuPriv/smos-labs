"""Загрузка сигналов для лабораторных работ 1-4.

Поддерживаются два формата входных данных (см. задание ЛР1):
  * текстовый файл: два столбца «время  амплитуда» (время в мс, как в Theory_Lab1)
    либо один столбец амплитуд (тогда шаг по времени задаётся параметром dt);
  * аудиофайл .wav (время в секундах).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Сигналы по умолчанию (можно заменить своими файлами через аргументы командной строки).
DEFAULT_SIGNALS = [
    os.path.join(DATA_DIR, "signal1.txt"),
    os.path.join(DATA_DIR, "signal2.txt"),
    os.path.join(DATA_DIR, "voice.wav"),
    os.path.join(DATA_DIR, "melody.wav"),
]


@dataclass
class Signal:
    name: str
    t: np.ndarray            # моменты времени t_j
    x: np.ndarray            # отсчёты p_j
    t_unit: str = "мс"       # единицы времени
    f_unit: str = "кГц"      # соответствующие единицы частоты (f = 1/t)
    meta: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def label(self) -> str:
        return f"{self.name} (N={self.n}, Δt={self.dt:.3g} {self.t_unit})"

    def decimate(self, k: int) -> "Signal":
        """Прореживание без фильтрации: каждый k-й отсчёт (шаг k·Δt)."""
        return Signal(self.name, self.t[::k].copy(), self.x[::k].copy(),
                      self.t_unit, self.f_unit, dict(self.meta))

    def crop(self, t_max: float) -> "Signal":
        """Ограничение временного диапазона сверху: t <= t_max."""
        m = self.t <= self.t[0] + t_max + 1e-12
        return Signal(self.name, self.t[m].copy(), self.x[m].copy(),
                      self.t_unit, self.f_unit, dict(self.meta))


def load_signal(path: str, dt: float | None = None) -> Signal:
    name = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        fs, data = wavfile.read(path)
        data = np.asarray(data)
        if data.ndim > 1:                         # стерео -> моно
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data / float(np.iinfo(data.dtype).max)
        x = data.astype(float)
        t = np.arange(len(x)) / float(fs)
        return Signal(name, t, x, "с", "Гц", {"fs": fs, "path": path})

    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1 or arr.shape[1] == 1:
        x = arr.ravel().astype(float)
        if dt is None:
            raise ValueError("Для одностолбцового файла нужно задать шаг dt")
        t = np.arange(len(x)) * dt
    else:
        t, x = arr[:, 0].astype(float), arr[:, 1].astype(float)
    return Signal(name, t, x, "мс", "кГц", {"path": path})


def load_signals(paths=None) -> list[Signal]:
    paths = paths or DEFAULT_SIGNALS
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Нет файлов {missing}. Сгенерируйте тестовые сигналы: python data/make_data.py")
    return [load_signal(p) for p in paths]
