from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import ROOT

PALETTE = (
    ROOT.TColor.GetColor("#E63946"),
    ROOT.TColor.GetColor("#00A86B"),
    ROOT.TColor.GetColor("#7B2CBF"),
    ROOT.TColor.GetColor("#FF8C00"),
    ROOT.TColor.GetColor("#0077B6"),
)


class FitError(RuntimeError):
    pass


@dataclass
class ParameterResult:
    index: int
    name: str
    value: float
    error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "value": _finite_or_none(self.value),
            "error": _finite_or_none(self.error),
        }


@dataclass
class FitResult:
    config: dict[str, Any]
    fit: ROOT.TF1
    valid: bool
    status: int
    chi2: float
    ndf: int
    p_value: float
    parameters: tuple[ParameterResult, ...]

    @property
    def chi2_ndf(self) -> float | None:
        if self.ndf <= 0:
            return None
        return self.chi2 / self.ndf

    def mean_parameter(self) -> ParameterResult | None:
        for candidate in ("mean", "mu"):
            for parameter in self.parameters:
                if parameter.name.lower() == candidate:
                    return parameter
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.config["name"],
            "function": self.config["function"],
            "range": list(self.config["range"]),
            "valid": self.valid,
            "status": self.status,
            "chi2": _finite_or_none(self.chi2),
            "ndf": self.ndf,
            "chi2_ndf": _finite_or_none(self.chi2_ndf),
            "p_value": _finite_or_none(self.p_value),
            "parameters": (
                [parameter.to_dict() for parameter in self.parameters]
                if self.valid
                else []
            ),
        }

    def summary(self, channel: int) -> str:
        name = self.config["name"]
        if not self.valid:
            return f"  channel {channel}: {name} fit failed (status {self.status})"
        mean = self.mean_parameter()
        if mean is None:
            return (
                f"  channel {channel}: {name} "
                f"chi2/ndf = {self.chi2:.2f}/{self.ndf}"
            )
        return (
            f"  channel {channel}: {name} "
            f"mean = {mean.value:.4f} +/- {mean.error:.4f}"
        )


def fit_histogram(
    histogram: ROOT.TH1D,
    config: dict[str, Any],
    object_name: str,
    color: int | None,
) -> FitResult:
    low, high = config["range"]
    function = ROOT.TF1(object_name, config["function"], low, high)

    try:
        for parameter in config["parameters"]:
            index = parameter["index"]
            if parameter.get("init") is not None:
                function.SetParameter(index, parameter["init"])
            if parameter.get("limits") is not None:
                function.SetParLimits(index, *parameter["limits"])
            function.SetParName(index, parameter["name"])
    except Exception as exc:
        raise FitError(
            f"Cannot configure fit {config['name']!r}: {exc}"
        ) from exc

    function.SetLineColor(color if color is not None else PALETTE[0])
    function.SetLineWidth(4)

    result = histogram.Fit(function, "RSQ", "", low, high)

    valid = False
    status = -1
    fit_result = result.Get() if result is not None else None
    if fit_result is not None:
        try:
            status = int(fit_result.Status())
            valid = True
        except ReferenceError:
            valid = False

    parameters = tuple(
        ParameterResult(
            index=index,
            name=_parameter_name(config, function, index),
            value=float(function.GetParameter(index)),
            error=float(function.GetParError(index)),
        )
        for index in range(function.GetNpar())
    )

    return FitResult(
        config=config,
        fit=function,
        valid=valid,
        status=status,
        chi2=float(function.GetChisquare()),
        ndf=int(function.GetNDF()),
        p_value=float(function.GetProb()),
        parameters=parameters,
    )


def _parameter_name(
    config: dict[str, Any],
    function: ROOT.TF1,
    index: int,
) -> str:
    for parameter in config["parameters"]:
        if parameter["index"] == index:
            return parameter["name"]
    name = function.GetParName(index)
    return name if name else f"p{index}"


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
