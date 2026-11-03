from __future__ import annotations

from pathlib import Path
from typing import Any

import ROOT

from fitting import FitResult


def plot_histogram(
    histogram: ROOT.TH1D,
    fits: list[FitResult],
    path: Path,
    config: dict[str, Any],
) -> None:
    apply_style()

    canvas = ROOT.TCanvas(
        f"c_{histogram.GetName()}",
        histogram.GetTitle(),
        config["width"],
        config["height"],
    )
    canvas.SetMargin(0.11, 0.05, 0.12, 0.10)

    line_color = ROOT.TColor.GetColor("#2B6CB0")
    fill_color = ROOT.TColor.GetColor("#AED3F2")
    histogram.SetLineColor(line_color)
    histogram.SetLineWidth(2)
    histogram.SetFillColor(fill_color)
    histogram.SetFillColorAlpha(fill_color, 0.55)
    histogram.Draw("hist")

    for fit in fits:
        if fit.valid:
            fit.fit.Draw("same")

    legend = _legend(fits)
    legend.Draw()

    if config["log_y"]:
        canvas.SetLogy()

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def apply_style() -> None:
    style = ROOT.gStyle
    style.SetOptStat(0)
    style.SetOptFit(0)
    style.SetTitleFont(42, "t")
    style.SetTitleSize(0.052, "t")
    style.SetTitleFont(42, "x")
    style.SetTitleFont(42, "y")
    style.SetTitleSize(0.043, "x")
    style.SetTitleSize(0.043, "y")
    style.SetTitleOffset(1.05, "x")
    style.SetTitleOffset(1.30, "y")
    style.SetLabelFont(42, "x")
    style.SetLabelFont(42, "y")
    style.SetLabelSize(0.038, "x")
    style.SetLabelSize(0.038, "y")
    style.SetTickLength(0.018, "x")
    style.SetTickLength(0.018, "y")
    style.SetGridColor(ROOT.kGray + 1)
    style.SetGridStyle(2)
    style.SetGridWidth(1)
    style.SetFrameLineWidth(1)
    style.SetFrameLineColor(ROOT.kGray + 2)
    style.SetLegendFont(42)
    style.SetLegendTextSize(0.036)


def _sigma_parameter(fit: FitResult):
    for parameter in fit.parameters:
        if parameter.name.lower() in ("sigma", "σ"):
            return parameter
    return None


def _legend(fits: list[FitResult]) -> ROOT.TLegend:
    lines = sum(
        2 if _has_sigma_line(fit) else 1
        for fit in fits
    )
    height = 0.048 * lines + 0.02
    legend = ROOT.TLegend(0.50, 0.88 - height, 0.91, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextAlign(12)

    for fit in fits:
        mean = fit.mean_parameter()
        sigma = _sigma_parameter(fit)
        if not fit.valid:
            text = f"{fit.config['name']}: fit failed"
        elif mean is not None:
            text = (
                f"{fit.config['name']}: #mu = {mean.value:.2f} "
                f"#pm {mean.error:.2f}"
            )
        else:
            text = f"{fit.config['name']}"
        legend.AddEntry(fit.fit, text, "l" if fit.valid else "")
        if _has_sigma_line(fit):
            legend.AddEntry(
                fit.fit,
                f"#sigma = {sigma.value:.2f} #pm {sigma.error:.2f}",
                "",
            )

    return legend


def _has_sigma_line(fit: FitResult) -> bool:
    return fit.valid and fit.mean_parameter() is not None and _sigma_parameter(fit) is not None
