from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ROOT

import read


FWHM_PER_SIGMA = 2.3548200450309493

TAG_DEFAULTS = {
    "tac_ns": {
        "x_title": "Delay (ns)",
        "y_title": "ADC channel",
        "slope_units": "channel/ns",
        "measure": "mean",
        "linear": True,
    },
    "tac_cm": {
        "x_title": "Cable length (cm)",
        "y_title": "ADC channel",
        "slope_units": "channel/cm",
        "measure": "mean",
        "linear": True,
    },
    "walk": {
        "x_title": "Delay (ns)",
        "y_title": "FWHM (ADC channel)",
        "slope_units": "channel/ns",
        "measure": "fwhm",
        "linear": False,
    },
}


class CalibrationError(RuntimeError):
    pass


def load_config(
    path: str | Path, tag: str | None = None
) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise CalibrationError(f"Configuration file not found: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise CalibrationError("The configuration root must be a JSON object.")

    base = config_path.parent
    output_config = config.get("output", "output")
    if isinstance(output_config, dict):
        output_path = output_config.get("path", "output")
        save_root = output_config.get("save_root", True)
    else:
        output_path = output_config
        save_root = config.get("save_root", True)

    output_path = Path(str(output_path)).expanduser()
    if output_path.is_absolute():
        raise CalibrationError(
            '"output.path" must be relative to the output folder.'
        )
    config["output"] = (read.OUTPUT_ROOT / output_path).resolve()
    config["save_root"] = save_root

    plot_config = config.setdefault("plot", {})
    plot_config.setdefault("enabled", True)
    plot_config.setdefault("width", 1600)
    plot_config.setdefault("height", 1000)

    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise CalibrationError('Missing "calibration" object.')

    if tag is not None:
        calibration["tag"] = tag

    tag = calibration.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        raise CalibrationError('"calibration.tag" must be a non-empty string.')
    tag = tag.strip()
    calibration["tag"] = tag

    block = calibration.get(tag)
    if isinstance(block, dict):
        for key in (
            "name",
            "x_title",
            "y_title",
            "slope_units",
            "points",
            "measure",
            "error_source",
            "linear",
        ):
            if key in block:
                calibration[key] = block[key]

    defaults = TAG_DEFAULTS.get(tag)
    if defaults is None:
        raise CalibrationError(
            f'Unknown calibration tag {tag!r}; '
            f"expected one of {sorted(TAG_DEFAULTS)}."
        )
    for key, value in defaults.items():
        calibration.setdefault(key, value)

    calibration.setdefault("name", "Calibration")
    calibration.setdefault("error_source", "mean_error")

    if calibration["measure"] not in ("mean", "sigma", "fwhm"):
        raise CalibrationError(
            '"calibration.measure" must be "mean", "sigma" or "fwhm".'
        )
    if calibration["error_source"] not in (
        "mean_error",
        "sigma",
        "sigma_error",
        "none",
    ):
        raise CalibrationError(
            '"calibration.error_source" must be "mean_error", "sigma", '
            '"sigma_error" or "none".'
        )
    if not isinstance(calibration["linear"], bool):
        raise CalibrationError('"calibration.linear" must be true or false.')

    points = calibration.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise CalibrationError(
            f'"calibration.{tag}.points" needs at least two points.'
        )

    for index, point in enumerate(points):
        where = f"calibration.{tag}.points[{index}]"
        if not isinstance(point, dict):
            raise CalibrationError(f"{where} must be an object.")
        if "value" not in point:
            raise CalibrationError(f'{where} needs "value".')
        if "results" not in point and "config" not in point:
            raise CalibrationError(f'{where} needs "results" or "config".')
        if "results" in point:
            point["results"] = str(_resolve(point["results"], base))
        if "config" in point:
            point["config"] = str(_resolve(point["config"], base))
        point["value"] = _number(point["value"], f"{where}.value")
        if "channel" in point:
            point["channel"] = int(_number(point["channel"], f"{where}.channel"))

    return config, config_path


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{where} must be a number.")
    return float(value)


def fitting_configs(config: dict[str, Any]) -> list[Path]:
    configs: list[Path] = []
    seen: set[str] = set()
    for point in config["calibration"]["points"]:
        path = point.get("config")
        if path is not None and path not in seen:
            seen.add(path)
            configs.append(Path(path))
    return configs


def results_path(point: dict[str, Any]) -> Path:
    if "results" in point:
        return Path(point["results"])
    config, _ = read.load_config(point["config"])
    return read.output_directory(config, None) / "results.json"


def read_point(
    point: dict[str, Any], measure: str, error_source: str
) -> dict[str, Any]:
    path = results_path(point)
    if not path.is_file():
        raise CalibrationError(
            f"Fitting result not found: {path}. Run the fitting first."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Invalid JSON in {path}: {exc}") from exc

    file_filter = point.get("file")
    channel_filter = point.get("channel")
    histogram_filter = point.get("histogram")
    fit_filter = point.get("fit")

    for file_entry in data["files"]:
        if file_filter is not None and Path(file_entry["input_file"]).stem != file_filter:
            continue
        for channel_entry in file_entry["channels"]:
            if channel_filter is not None and channel_entry["channel"] != channel_filter:
                continue
            for histogram in channel_entry["histograms"]:
                if histogram_filter is not None and histogram["name"] != histogram_filter:
                    continue
                for fit in histogram["fits"]:
                    if fit_filter is not None and fit["name"] != fit_filter:
                        continue
                    measured = _measure(fit, measure, error_source)
                    if measured is None:
                        continue
                    return {
                        "results": str(path),
                        "config": point.get("config"),
                        "file": file_entry["input_file"],
                        "channel": channel_entry["channel"],
                        "histogram": histogram["name"],
                        "fit": fit["name"],
                        "x": point["value"],
                        "y": measured[0],
                        "y_error": measured[1],
                        "chi2_ndf": fit["chi2_ndf"],
                    }

    raise CalibrationError(
        f"No matching fit in {path} "
        f"(channel={channel_filter}, histogram={histogram_filter}, "
        f"fit={fit_filter}, file={file_filter})."
    )


def _measure(
    fit: dict[str, Any], measure: str, error_source: str
) -> tuple[float, float] | None:
    mean = _parameter(fit, ("mean", "mu"))
    sigma = _parameter(fit, ("sigma", "σ"))

    if measure == "mean":
        if mean is None:
            return None
        value = mean["value"]
    elif measure == "sigma":
        if sigma is None:
            return None
        value = sigma["value"]
    else:
        if sigma is None:
            return None
        value = FWHM_PER_SIGMA * sigma["value"]

    error = _error(error_source, mean, sigma)
    if measure == "fwhm" and error_source in ("sigma", "sigma_error"):
        error = FWHM_PER_SIGMA * error
    return value, error


def _error(
    source: str,
    mean: dict[str, Any] | None,
    sigma: dict[str, Any] | None,
) -> float:
    if source == "none":
        return 0.0
    if source == "mean_error":
        return float(mean["error"]) if mean is not None else 0.0
    if source == "sigma":
        return float(sigma["value"]) if sigma is not None else 0.0
    return float(sigma["error"]) if sigma is not None else 0.0


def _parameter(
    fit: dict[str, Any], names: tuple[str, ...]
) -> dict[str, Any] | None:
    for parameter in fit.get("parameters", []):
        if parameter["name"].lower() in names:
            return parameter
    return None


def make_graph(points: list[dict[str, Any]]) -> ROOT.TGraphErrors:
    graph = ROOT.TGraphErrors(len(points))
    for index, point in enumerate(points):
        graph.SetPoint(index, point["x"], point["y"])
        graph.SetPointError(index, 0.0, point["y_error"])
    graph.SetTitle("")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.2)
    graph.SetMarkerColor(ROOT.TColor.GetColor("#2B6CB0"))
    graph.SetLineColor(ROOT.TColor.GetColor("#2B6CB0"))
    return graph


def linear_fit(
    graph: ROOT.TGraphErrors,
    points: list[dict[str, Any]],
) -> ROOT.TF1:
    low = min(point["x"] for point in points)
    high = max(point["x"] for point in points)
    span = high - low if high > low else 1.0
    function = ROOT.TF1(
        "f_calibration",
        "pol1",
        low - 0.1 * span,
        high + 0.1 * span,
    )
    values = [point["y"] for point in points]
    value_span = max(values) - min(values)
    function.SetParameter(0, min(values))
    function.SetParameter(1, value_span / span if span else 1.0)
    graph.Fit(function, "SQ")
    return function


def plot_calibration(
    graph: ROOT.TGraphErrors,
    function: ROOT.TF1 | None,
    path: Path,
    config: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
    from plot import apply_style

    apply_style()
    canvas = ROOT.TCanvas(
        "c_calibration",
        calibration["name"],
        config["width"],
        config["height"],
    )
    canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
    canvas.SetTicks(1, 1)
    canvas.SetFrameLineWidth(2)
    canvas.SetFrameLineColor(ROOT.kGray + 2)

    ROOT.gStyle.SetOptFit(0)
    graph.Draw("AP")
    graph.SetTitle(calibration["name"])
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.4)
    graph.SetMarkerColor(ROOT.TColor.GetColor("#1F4E79"))
    graph.SetLineColor(ROOT.TColor.GetColor("#1F4E79"))
    graph.SetLineWidth(2)

    x_axis = graph.GetXaxis()
    x_axis.SetTitle(calibration["x_title"])
    x_axis.SetTitleSize(0.045)
    x_axis.SetTitleOffset(1.15)
    x_axis.SetLabelSize(0.040)
    x_axis.CenterTitle(True)

    y_axis = graph.GetYaxis()
    y_axis.SetTitle(calibration["y_title"])
    y_axis.SetTitleSize(0.045)
    y_axis.SetTitleOffset(1.30)
    y_axis.SetLabelSize(0.040)
    y_axis.CenterTitle(True)

    if function is not None:
        function.SetLineColor(ROOT.TColor.GetColor("#E63946"))
        function.SetLineWidth(3)
        function.Draw("same")

    if function is not None:
        slope = function.GetParameter(1)
        slope_error = function.GetParError(1)
        intercept = function.GetParameter(0)
        intercept_error = function.GetParError(0)
        units = calibration.get("slope_units", "")
        unit_text = f" {units}" if units else ""

        box = ROOT.TPaveText(0.15, 0.72, 0.62, 0.89, "NDC")
        box.SetBorderSize(0)
        box.SetFillStyle(1001)
        box.SetFillColorAlpha(ROOT.kWhite, 0.85)
        box.SetTextAlign(12)
        box.SetTextFont(42)
        box.SetTextSize(0.034)
        box.AddText("Linear calibration")
        box.AddText(
            f"slope = ({slope:.5f} #pm {slope_error:.5f}){unit_text}"
        )
        box.AddText(
            f"intercept = ({intercept:.4f} #pm {intercept_error:.4f})"
        )
        box.Draw()

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def write_results(result: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    tags = calibration["tag"]
    points = [
        read_point(point, calibration["measure"], calibration["error_source"])
        for point in calibration["points"]
    ]

    graph = make_graph(points)
    function = linear_fit(graph, points) if calibration["linear"] else None

    output = config["output"] / tags
    output.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "tag": tags,
        "name": calibration["name"],
        "measure": calibration["measure"],
        "error_source": calibration["error_source"],
        "linear": calibration["linear"],
        "x_title": calibration["x_title"],
        "y_title": calibration["y_title"],
        "points": points,
    }

    if function is not None:
        slope = float(function.GetParameter(1))
        slope_error = float(function.GetParError(1))
        intercept = float(function.GetParameter(0))
        intercept_error = float(function.GetParError(0))
        result.update(
            slope=slope,
            slope_error=slope_error,
            slope_units=calibration.get("slope_units", ""),
            intercept=intercept,
            intercept_error=intercept_error,
            chi2=float(function.GetChisquare()),
            ndf=int(function.GetNDF()),
            chi2_ndf=(
                float(function.GetChisquare()) / int(function.GetNDF())
                if function.GetNDF() > 0
                else None
            ),
        )

    if config["plot"]["enabled"]:
        for old in output.glob("calibration*.png"):
            old.unlink()
        if function is not None:
            chi2_ndf = result.get("chi2_ndf")
            shown = (
                f"{chi2_ndf:.2f}" if chi2_ndf is not None else "undefined"
            )
            plot_name = f"calibration_chi2ndf_{shown}.png"
        else:
            plot_name = "calibration.png"
        plot_calibration(
            graph,
            function,
            output / plot_name,
            config["plot"],
            calibration,
        )

    if config["save_root"]:
        root_file = ROOT.TFile.Open(str(output / "calibration.root"), "RECREATE")
        if not root_file or root_file.IsZombie():
            raise CalibrationError(f"Cannot create {output / 'calibration.root'}")
        graph.Write("graph")
        if function is not None:
            function.Write("fit")
        root_file.Close()

    write_results(result, output / "calibration.json")

    print(f"\n{calibration['name']} ({tags})")
    for point in points:
        print(
            f"  {Path(point['file']).stem}: x = {point['x']:.2f}, "
            f"y = {point['y']:.3f} +/- {point['y_error']:.3f}"
        )
    if function is not None:
        units = calibration.get("slope_units", "")
        unit_text = f" [{units}]" if units else ""
        print(
            f"  slope = {result['slope']:.5f} +/- {result['slope_error']:.5f}"
            f"{unit_text}, intercept = {result['intercept']:.4f} +/- "
            f"{result['intercept_error']:.4f}, "
            f"chi2/ndf = {result['chi2']:.2f}/{result['ndf']}"
        )
    print(f"  Calibration written to {output / 'calibration.json'}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a TAC or walk plot from fitting result JSON files."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "config_calibration_tac.json",
    )
    args = parser.parse_args(argv)

    ROOT.gROOT.SetBatch(True)
    try:
        config, _ = load_config(args.config)
        run(config)
    except (
        CalibrationError,
        read.ConfigError,
        read.ReaderError,
        read.FitError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
