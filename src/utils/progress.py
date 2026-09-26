"""
Shared progress-bar helper, used by every training loop and every
long-running script in this project (baselines, acquire_all_data.py,
prepare_data.py) so progress is shown consistently rather than each
module inventing its own.

Goal: for anything that takes more than a few seconds, show a
percentage/ETA so a person watching the terminal (or logs/*.log, in a
headless run) can tell it is progressing and roughly how much is left --
never leave a multi-hour step silent (see classical_lstm.py's earlier
bug: a 50-minute epoch with zero output in between).

Falls back to tqdm if installed (pip install tqdm; already listed in
requirements.txt), else to a dependency-free bar that writes to stderr.
Both write one line per call to file_progress(), which is also appended
to run_logger at a low frequency, so headless / redirected-to-file runs
still get periodic progress lines in the actual .log file.
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class _FallbackBar:
    """Dependency-free replacement for tqdm's public surface used here
    (iteration, .update(), .set_postfix(), .close(), context manager)."""

    def __init__(self, iterable=None, total=None, desc="", unit="it", min_interval=0.2):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.desc = desc
        self.unit = unit
        self.min_interval = min_interval
        self.n = 0
        self._t0 = time.time()
        self._t_last = 0.0
        self._postfix = ""

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def _render(self, force=False):
        now = time.time()
        if not force and now - self._t_last < self.min_interval:
            return
        self._t_last = now
        elapsed = now - self._t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        if self.total:
            pct = 100.0 * self.n / self.total
            eta = (self.total - self.n) / rate if rate > 0 else float("inf")
            eta_str = f"{eta:5.0f}s" if eta != float("inf") else "  ?s"
            bar_len = 24
            filled = int(bar_len * self.n / self.total) if self.total else 0
            bar = "#" * filled + "-" * (bar_len - filled)
            line = (f"\r{self.desc} [{bar}] {pct:5.1f}% ({self.n}/{self.total} {self.unit}) "
                    f"{rate:.1f}{self.unit}/s ETA {eta_str} {self._postfix}")
        else:
            line = f"\r{self.desc} {self.n} {self.unit} ({rate:.1f}{self.unit}/s) {self._postfix}"
        sys.stderr.write(line[:160])
        sys.stderr.flush()

    def update(self, n=1):
        self.n += n
        self._render()

    def set_postfix(self, **kw):
        self._postfix = " ".join(f"{k}={v}" for k, v in kw.items())
        self._render()

    def close(self):
        self._render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def progress_bar(iterable: Optional[Iterable[T]] = None, total: Optional[int] = None,
                 desc: str = "", unit: str = "it"):
    """Returns a tqdm progress bar if tqdm is installed, else a minimal
    dependency-free fallback with the same iteration / .update() /
    .set_postfix() / .close() surface. Use exactly like tqdm:

        for item in progress_bar(items, desc="Training"):
            ...

        with progress_bar(total=n_epochs, desc="Epochs") as bar:
            for epoch in range(n_epochs):
                ...
                bar.update(1)
                bar.set_postfix(loss=f"{loss:.4f}")
    """
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, unit=unit, leave=True)
    return _FallbackBar(iterable, total=total, desc=desc, unit=unit)


def log_progress_milestone(run_logger, phase: str, current: int, total: int,
                           every_percent: int = 10, **extra) -> None:
    """For a headless run whose live terminal bar nobody is watching,
    writes ONE line to run_logger (-> the .log file) every `every_percent`
    percent, instead of either flooding the log every step or (the bug
    this replaces) saying nothing for the whole run. Call once per
    iteration; it self-throttles.
    """
    if run_logger is None or total <= 0:
        return
    pct = 100.0 * current / total
    step_pct = 100.0 * (current - 1) / total if current > 0 else -1
    if int(pct // every_percent) > int(step_pct // every_percent) or current == total:
        extras = " ".join(f"{k}={v}" for k, v in extra.items())
        run_logger.info(f"[{phase}] {pct:5.1f}% ({current}/{total}) {extras}")
