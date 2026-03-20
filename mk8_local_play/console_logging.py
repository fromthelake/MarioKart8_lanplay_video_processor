import os
import sys
import time
from dataclasses import dataclass

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


@dataclass
class ResourceSnapshot:
    cpu_percent: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    gpu_percent: float | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None


class ResourceMonitor:
    def __init__(self):
        self.peak = ResourceSnapshot()
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def sample(self) -> ResourceSnapshot:
        snapshot = ResourceSnapshot()
        if psutil is not None:
            try:
                snapshot.cpu_percent = float(psutil.cpu_percent(interval=None))
                memory = psutil.virtual_memory()
                snapshot.ram_used_gb = memory.used / (1024 ** 3)
                snapshot.ram_total_gb = memory.total / (1024 ** 3)
            except Exception:
                pass

        self._update_peak(snapshot)
        return snapshot

    def _update_peak(self, snapshot: ResourceSnapshot) -> None:
        for field in ("cpu_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"):
            current = getattr(snapshot, field)
            peak_value = getattr(self.peak, field)
            if current is not None and (peak_value is None or current > peak_value):
                setattr(self.peak, field, current)
        for field in ("ram_total_gb", "vram_total_gb"):
            current = getattr(snapshot, field)
            if current is not None:
                setattr(self.peak, field, current)


class ConsoleLogger:
    COLORS = {
        "cyan": "\033[1;96m",
        "magenta": "\033[1;95m",
        "green": "\033[1;92m",
        "yellow": "\033[1;93m",
        "red": "\033[1;91m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }

    def __init__(self):
        self.start_time = time.perf_counter()
        self.resources = ResourceMonitor()
        self.use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def reset(self) -> None:
        self.start_time = time.perf_counter()
        self.resources = ResourceMonitor()

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.start_time)

    def elapsed_label(self) -> str:
        return self.format_duration(self.elapsed_seconds())

    @staticmethod
    def format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{secs:02}"
        return f"{minutes:02}:{secs:02}"

    def color(self, text: str, color_name: str | None) -> str:
        if not self.use_color or not color_name:
            return text
        prefix = self.COLORS.get(color_name, "")
        reset = self.COLORS["reset"] if prefix else ""
        return f"{prefix}{text}{reset}"

    def log(self, scope: str, message: str, color_name: str | None = None) -> None:
        if scope:
            print(f"[{self.elapsed_label()}] {self.color(scope, color_name)} {message}")
        else:
            print(f"[{self.elapsed_label()}] {message}")

    def summary_block(self, scope: str, lines: list[str], color_name: str | None = None) -> None:
        self.log(scope, "", color_name=color_name)
        print("")
        for line in lines:
            print(f"  {line}")
        print("")

    def blank_lines(self, count: int = 1) -> None:
        for _ in range(max(0, count)):
            print("")

    def resource_text(self, snapshot: ResourceSnapshot | None = None) -> str:
        snapshot = snapshot or self.resources.sample()
        parts = []
        if snapshot.cpu_percent is not None:
            parts.append(f"CPU {snapshot.cpu_percent:.0f}%")
        if snapshot.ram_used_gb is not None and snapshot.ram_total_gb is not None:
            parts.append(f"RAM {snapshot.ram_used_gb:.1f}/{snapshot.ram_total_gb:.1f} GB")
        return " | ".join(parts)

    def peak_lines(self, snapshot: ResourceSnapshot | None = None) -> list[str]:
        snapshot = snapshot or self.resources.peak
        lines: list[str] = []
        if snapshot.cpu_percent is not None:
            lines.append(f"Peak CPU: {snapshot.cpu_percent:.0f}%")
        if snapshot.ram_used_gb is not None and snapshot.ram_total_gb is not None:
            lines.append(f"Peak RAM: {snapshot.ram_used_gb:.1f} / {snapshot.ram_total_gb:.1f} GB")
        return lines


LOGGER = ConsoleLogger()
