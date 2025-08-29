# -*- coding: utf-8 -*-
"""
ResourceMonitor – enkel ressourceovervågning til engine/trænings-jobs.

Funktioner:
- RAM/CPU + (valgfrit) GPU load/temperatur via GPUtil (failsafe hvis ikke tilgængelig)
- Periodisk status-print (kan slås fra)
- CSV-log med header (dato, ram_pct, cpu_pct, gpu_pct, gpu_temp_c)
- Actions på overload: 'warn' | 'pause' | 'kill'
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

import psutil

# GPUtil er valgfri – gør det failsafe
try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover
    GPUtil = None

from utils.project_path import PROJECT_ROOT


class ResourceMonitor:
    def __init__(
        self,
        ram_max: float = 90,
        cpu_max: float = 90,
        gpu_max: float = 90,
        gpu_temp_max: float = 85,
        check_interval: int = 10,
        action: str = "pause",          # 'pause' | 'kill' | 'warn'
        log_file: Optional[str | Path] = None,
        verbose: bool = True,
        cpu_sample_interval_sec: float = 1.0,  # hvor længe cpu_percent må blokere i hvert tjek
    ) -> None:
        self.ram_max = float(ram_max)
        self.cpu_max = float(cpu_max)
        self.gpu_max = float(gpu_max)
        self.gpu_temp_max = float(gpu_temp_max)
        self.check_interval = int(check_interval)
        self.action = str(action).lower().strip()
        self.verbose = bool(verbose)
        self.cpu_sample_interval_sec = float(cpu_sample_interval_sec)

        # Normalisér logsti til Path (eller None)
        self.log_path: Optional[Path] = Path(log_file) if log_file else None
        self._header_written = False

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    # ------------- Public API -------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()
        if self.verbose:
            print("[Monitor] Overvågning startet.")

    def stop(self, join_timeout: Optional[float] = 3.0) -> None:
        self._stop_event.set()
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=join_timeout)
        except Exception:
            pass
        if self.verbose:
            print("[Monitor] Overvågning stoppet.")

    # ------------- Internals -------------
    def _ensure_log_dir_and_header(self) -> None:
        if not self.log_path:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                # skriv header første gang
                with self.log_path.open("w", encoding="utf-8", newline="") as f:
                    f.write("timestamp,ram_pct,cpu_pct,gpu_pct,gpu_temp_c\n")
                self._header_written = True
        except Exception as e:
            # Ingen hård fejl – vi kører videre uden log.
            if self.verbose:
                print(f"[Monitor] Kunne ikke forberede logfil: {e}")

    @staticmethod
    def _get_gpu_stats() -> Tuple[float, float]:
        """Returnér (gpu_load_pct, gpu_temp_c). Fallback til (0.0, 0.0) hvis ingen GPU/GPUtil."""
        try:
            if GPUtil is None:
                return 0.0, 0.0
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0, 0.0
            g = gpus[0]  # brug første GPU
            # GPUtil load er 0..1
            load_pct = float(getattr(g, "load", 0.0) or 0.0) * 100.0
            temp_c = float(getattr(g, "temperature", 0.0) or 0.0)
            # Clamp og NaN-sikring
            if not (0.0 <= load_pct <= 100.0):
                load_pct = max(0.0, min(100.0, load_pct))
            if temp_c < -50 or temp_c > 150:  # urimelige værdier → nulstil
                temp_c = 0.0
            return load_pct, temp_c
        except Exception:
            return 0.0, 0.0

    def _log_row(self, ts_str: str, ram: float, cpu: float, gpu: float, gpu_temp: float) -> None:
        if not self.log_path:
            return
        try:
            if not self._header_written and not self.log_path.exists():
                self._ensure_log_dir_and_header()
            with self.log_path.open("a", encoding="utf-8", newline="") as f:
                f.write(f"{ts_str},{ram:.2f},{cpu:.2f},{gpu:.2f},{gpu_temp:.2f}\n")
        except Exception as e:
            if self.verbose:
                print(f"[Monitor] Logskrivning fejlede: {e}")

    def _do_action_on_overload(self, reason: str) -> None:
        if self.action == "warn":
            return
        elif self.action == "pause":
            # Bemærk: Dette pauser kun monitor-tråden – ikke hovedprocessen.
            # Vil du *faktisk* pause arbejdstråden, så udvid med en callback/signal.
            print(f"[Monitor] Pauser pga. overload ({reason}) i 60 sek...")
            time.sleep(60)
        elif self.action == "kill":
            print(f"[Monitor] Lukker proces pga. overload ({reason})...")
            os._exit(1)  # hårdt exit
        else:
            if self.verbose:
                print("[Monitor] Ukendt action – fortsætter.")

    def _monitor(self) -> None:
        # Sørg for logfil/mapper
        self._ensure_log_dir_and_header()

        while not self._stop_event.is_set():
            try:
                # cpu_percent kan blokere for prøveperioden
                ram_pct = float(psutil.virtual_memory().percent)
                cpu_pct = float(psutil.cpu_percent(interval=self.cpu_sample_interval_sec))
                gpu_pct, gpu_temp = self._get_gpu_stats()

                if self.verbose:
                    print(
                        f"[Monitor] RAM: {ram_pct:.1f}% | CPU: {cpu_pct:.1f}% | "
                        f"GPU: {gpu_pct:.1f}% | GPU-temp: {gpu_temp:.1f}C"
                    )

                # skriv log
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                self._log_row(ts_str, ram_pct, cpu_pct, gpu_pct, gpu_temp)

                # overload?
                reasons = []
                if ram_pct > self.ram_max:
                    reasons.append("RAM")
                if cpu_pct > self.cpu_max:
                    reasons.append("CPU")
                if gpu_pct > self.gpu_max:
                    reasons.append("GPU")
                if gpu_temp > self.gpu_temp_max:
                    reasons.append("GPU-temp")

                if reasons:
                    print(f"🚨 RESSOURCE-ALARM ({', '.join(reasons)}) 🚨")
                    self._do_action_on_overload(", ".join(reasons))

            except Exception as e:
                # Beskyt tråden mod at dø; log og fortsæt
                print(f"[Monitor] Fejl i overvågning: {e}")

            # sov til næste loop (minus cpu_sample_interval_sec hvis du vil samme totale periode)
            remaining = max(0.0, self.check_interval - self.cpu_sample_interval_sec)
            time.sleep(remaining)


# Eksempel på brug
if __name__ == "__main__":
    monitor = ResourceMonitor(
        ram_max=80,
        cpu_max=90,
        gpu_max=95,
        gpu_temp_max=80,
        check_interval=10,
        action="pause",  # eller "kill" / "warn"
        log_file=PROJECT_ROOT / "outputs" / "debug" / "resource_log.csv",
        verbose=True,
    )
    try:
        monitor.start()
        # Dummy main loop
        for i in range(20):
            print(f"Dummy main loop {i}")
            time.sleep(3)
    finally:
        monitor.stop()
