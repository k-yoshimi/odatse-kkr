"""
Shared MetricExtractor utilities for parsing AkaiKKR outputs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict

__all__ = ["MetricExtractor"]


DEFAULT_METRIC_PATTERNS = {
    "total_energy": r"total energy=?\s+([-\d.+Ee]+)",
    "band_energy": r"band energy=?\s+([-\d.+Ee]+)",
}


class MetricExtractor:
    """Utility to parse scalar metrics from AkaiKKR output."""

    _TRANSFORMS = {
        "identity": lambda x: x,
        "abs": lambda x: abs(x),
        "log": lambda x: math.log(x),
        "log1p": lambda x: math.log1p(x),
        "sqrt": lambda x: math.sqrt(x),
        "square": lambda x: x * x,
    }

    def __init__(self, metric_cfg: Dict[str, Any], *, default_name: str = "total_energy"):
        name = metric_cfg.get("name", default_name)
        pattern = metric_cfg.get("pattern") or DEFAULT_METRIC_PATTERNS.get(name)
        if not pattern:
            raise ValueError(
                f"Unsupported metric '{name}'. Provide a custom regex via metric.pattern."
            )

        ignore_case = metric_cfg.get("ignore_case", True)
        flags = re.IGNORECASE if ignore_case else 0
        self.pattern = re.compile(pattern, flags)
        self.group = int(metric_cfg.get("group", 1))
        self.scale = float(metric_cfg.get("scale", 1.0))
        self.name = name

        transform_name = metric_cfg.get("transform", "identity")
        transform = self._TRANSFORMS.get(transform_name)
        if transform is None:
            supported = ", ".join(sorted(self._TRANSFORMS))
            raise ValueError(
                f"Unsupported metric.transform '{transform_name}'. "
                f"Choose from: {supported}"
            )
        self.transform = transform

    def extract(self, output_path: Path) -> float:
        if not output_path.exists():
            raise FileNotFoundError(f"{output_path} was not created by AkaiKKR.")

        with output_path.open("r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                match = self.pattern.search(line)
                if match:
                    value = float(match.group(self.group))
                    scaled = value * self.scale
                    return self.transform(scaled)

        raise RuntimeError(
            f"Metric '{self.name}' not found in {output_path}. "
            "Adjust metric settings if the output format differs."
        )
