"""Errors raised by pipeline stages. Each carries a hint telling the user what to try next."""

from __future__ import annotations


class DigitizeError(Exception):
    """Base class for every expected failure in the pipeline."""

    stage: str = "pipeline"

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        text = f"[{self.stage}] {self.message}"
        if self.hint:
            text += f"\nHint: {self.hint}"
        return text


class ConfigError(DigitizeError):
    stage = "config"


class LoadError(DigitizeError):
    stage = "load"


class CornerError(DigitizeError):
    stage = "plot-area"


class CalibrationError(DigitizeError):
    stage = "calibrate"


class SelectionError(DigitizeError):
    stage = "select"


class TraceError(DigitizeError):
    stage = "trace"
