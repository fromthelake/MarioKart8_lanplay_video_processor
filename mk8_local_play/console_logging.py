import os
import re
import sys
import time
import colorsys
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
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    NEON_VIDEO_PALETTE = [
        (0, 255, 255),   # cyan
        (255, 64, 255),  # hot magenta
        (64, 255, 64),   # green
        (255, 96, 96),   # red
        (255, 192, 0),   # amber
        (64, 160, 255),  # blue
        (192, 96, 255),  # violet
        (0, 255, 160),   # mint
        (255, 128, 0),   # orange
        (255, 0, 128),   # pink-red
        (160, 255, 0),   # lime
        (0, 224, 255),   # ice blue
    ]
    COLORS = {
        "cyan": "\033[1;96m",
        "magenta": "\033[1;95m",
        "green": "\033[1;92m",
        "yellow": "\033[1;93m",
        "red": "\033[1;91m",
        "neon_blue": "\033[1;38;5;51m",
        "neon_pink": "\033[1;38;5;213m",
        "neon_lime": "\033[1;38;5;118m",
        "neon_orange": "\033[1;38;5;208m",
        "neon_purple": "\033[1;38;5;141m",
        "neon_teal": "\033[1;38;5;87m",
        "neon_yellow": "\033[1;38;5;226m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }

    def __init__(self):
        self.start_time = time.perf_counter()
        self.resources = ResourceMonitor()
        self.use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self._video_color_cache: dict[str, tuple[int, int, int]] = {}
        self._video_color_order: list[str] = []

    def reset(self) -> None:
        self.start_time = time.perf_counter()
        self.resources = ResourceMonitor()
        self._video_color_cache.clear()
        self._video_color_order.clear()

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
        return f"{hours:02}:{minutes:02}:{secs:02}"

    @classmethod
    def strip_ansi(cls, text: str) -> str:
        return cls.ANSI_RE.sub("", str(text))

    @classmethod
    def visible_width(cls, text: str) -> int:
        return len(cls.strip_ansi(text))

    @classmethod
    def fit_cell(cls, value: object, width: int, alignment: str = "left") -> str:
        text = str(value)
        plain = cls.strip_ansi(text)
        if len(plain) > width:
            if width <= 0:
                return ""
            if width <= 3:
                text = plain[:width]
            else:
                text = plain[: max(0, width - 3)] + "..."
            plain = cls.strip_ansi(text)
        padding = max(0, width - len(plain))
        if alignment == "right":
            return (" " * padding) + text
        return text + (" " * padding)

    def render_table(
        self,
        headers: list[str],
        rows: list[list[object]],
        *,
        alignments: list[str] | None = None,
        indent: str = "  ",
    ) -> list[str]:
        if not rows:
            return []
        alignments = list(alignments or ["left"] * len(headers))
        widths = [self.visible_width(header) for header in headers]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], self.visible_width(str(value)))
        lines = [
            indent + "  ".join(self.fit_cell(headers[index], widths[index], alignments[index]) for index in range(len(headers))),
            indent + "  ".join("-" * widths[index] for index in range(len(headers))),
        ]
        for row in rows:
            lines.append(
                indent + "  ".join(self.fit_cell(row[index], widths[index], alignments[index]) for index in range(len(headers)))
            )
        return lines

    def render_kv_table(self, rows: list[tuple[object, object]], *, indent: str = "  ") -> list[str]:
        if not rows:
            return []
        return self.render_table(
            ["Metric", "Value"],
            [[metric, value] for metric, value in rows],
            alignments=["left", "left"],
            indent=indent,
        )

    def color(self, text: str, color_name: str | None) -> str:
        if not self.use_color or not color_name:
            return text
        prefix = self.COLORS.get(color_name, "")
        reset = self.COLORS["reset"] if prefix else ""
        return f"{prefix}{text}{reset}"

    def bold(self, text: str) -> str:
        if not self.use_color:
            return text
        return f"\033[1m{text}{self.COLORS['reset']}"

    def color_rgb(self, text: str, red: int, green: int, blue: int) -> str:
        if not self.use_color:
            return text
        red = max(0, min(255, int(red)))
        green = max(0, min(255, int(green)))
        blue = max(0, min(255, int(blue)))
        return f"\033[1;38;2;{red};{green};{blue}m{text}{self.COLORS['reset']}"

    def _video_rgb_from_token(self, token: object) -> tuple[int, int, int]:
        token_text = str(token or "")
        cached = self._video_color_cache.get(token_text)
        if cached is not None:
            return cached

        assignment_index = len(self._video_color_order)
        self._video_color_order.append(token_text)
        if assignment_index < len(self.NEON_VIDEO_PALETTE):
            rgb = self.NEON_VIDEO_PALETTE[assignment_index]
        else:
            extra_index = assignment_index - len(self.NEON_VIDEO_PALETTE)
            hue = ((extra_index * 0.61803398875) + 0.07) % 1.0
            saturation = 0.72 if (extra_index % 2 == 0) else 0.84
            value = 1.0 if (extra_index % 3 != 0) else 0.92
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb = (int(red * 255), int(green * 255), int(blue * 255))
        self._video_color_cache[token_text] = rgb
        return rgb

    def color_video_identity(self, text: str, video_identity: object) -> str:
        red, green, blue = self._video_rgb_from_token(video_identity)
        return self.color_rgb(text, red, green, blue)

    def color_video_text(self, text: str, video_index: int) -> str:
        return self.color_video_identity(text, video_index)

    def video_value(self, value: object, video_identity: object) -> str:
        return self.color_video_identity(str(value), video_identity)

    def video_field(self, label: str, value: object, video_identity: object) -> str:
        return f"{label}{self.video_value(value, video_identity)}"

    def log(self, scope: str, message: str, color_name: str | None = None) -> None:
        if scope:
            print(f"[{self.elapsed_label()}] {self.color(scope, color_name)} {message}")
        else:
            print(f"[{self.elapsed_label()}] {message}")

    def summary_block(self, scope: str, lines: list[str], color_name: str | None = None) -> None:
        self.log(scope, "", color_name=color_name)
        print("")
        for line in lines:
            if not line:
                print("")
                continue
            self.log("", f"  {line}")
        print("")

    def blank_lines(self, count: int = 1) -> None:
        for _ in range(max(0, count)):
            print("")

    def resource_text(self, snapshot: ResourceSnapshot | None = None, *, value_color_token: object | None = None) -> str:
        snapshot = snapshot or self.resources.sample()
        parts = []
        if snapshot.cpu_percent is not None:
            cpu_value = f"{snapshot.cpu_percent:.0f}%"
            parts.append(self.video_field("CPU ", cpu_value, value_color_token) if value_color_token is not None else f"CPU {cpu_value}")
        if snapshot.ram_used_gb is not None and snapshot.ram_total_gb is not None:
            ram_value = f"{snapshot.ram_used_gb:.1f}/{snapshot.ram_total_gb:.1f} GB"
            parts.append(self.video_field("RAM ", ram_value, value_color_token) if value_color_token is not None else f"RAM {ram_value}")
        if snapshot.gpu_percent is not None:
            gpu_value = f"{snapshot.gpu_percent:.0f}%"
            parts.append(self.video_field("GPU ", gpu_value, value_color_token) if value_color_token is not None else f"GPU {gpu_value}")
        if snapshot.vram_used_gb is not None and snapshot.vram_total_gb is not None:
            vram_value = f"{snapshot.vram_used_gb:.1f}/{snapshot.vram_total_gb:.1f} GB"
            parts.append(self.video_field("VRAM ", vram_value, value_color_token) if value_color_token is not None else f"VRAM {vram_value}")
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
