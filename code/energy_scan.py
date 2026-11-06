from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import ROOT

import read


FWHM_PER_SIGMA = 2.3548200450309493


class EnergyScanError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise EnergyScanError(f"Configuration file not found: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnergyScanError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise EnergyScanError("The configuration root must be a JSON object.")

    base = config_path.parent
    input_config = config.get("input")
    if not isinstance(input_config, dict) or "path" not in input_config:
        raise EnergyScanError('Missing "input.path".')
    input_config["path"] = _resolve(input_config["path"], base)
    input_config.setdefault("pattern", "*.root")
    input_config.setdefault("tree", "Data_R")

    output_config = config.get("output", "energy_scan")
    if isinstance(output_config, dict):
        output_path = output_config.get("path", "energy_scan")
        save_root = output_config.get("save_root", True)
    else:
        output_path = output_config
        save_root = config.get("save_root", True)
    output_path = Path(str(output_path)).expanduser()
    if output_path.is_absolute():
        raise EnergyScanError(
            '"output.path" must be relative to the output folder.'
        )
    config["output"] = (read.OUTPUT_ROOT / output_path).resolve()
    config["save_root"] = save_root

    plot_config = config.setdefault("plot", {})
    plot_config.setdefault("enabled", True)
    plot_config.setdefault("width", 1600)
    plot_config.setdefault("height", 1000)

    scan = config.get("energy_scan")
    if not isinstance(scan, dict):
        raise EnergyScanError('Missing "energy_scan" object.')

    scan.setdefault("tac_range", [850, 1150])
    scan.setdefault("bins", 300)
    scan.setdefault("fit_margin", 150.0)
    scan.setdefault("min_events", 200)

    calibration = scan.get("calibration")
    if not isinstance(calibration, dict):
        raise EnergyScanError('Missing "energy_scan.calibration".')
    detectors = calibration.get("detectors")
    if not isinstance(detectors, list) or len(detectors) != 2:
        raise EnergyScanError(
            '"energy_scan.calibration.detectors" must contain two detectors.'
        )
    for detector in detectors:
        detector["results"] = str(_resolve(detector["results"], base))
        points = detector.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise EnergyScanError(
                '"detectors[].points" needs at least two points.'
            )
        detector["channel"] = int(detector["channel"])

    if "tac" not in calibration:
        raise EnergyScanError('Missing "energy_scan.calibration.tac".')
    calibration["tac"] = str(_resolve(calibration["tac"], base))

    comparison = scan.get("co60")
    if isinstance(comparison, dict) and "results" in comparison:
        comparison["results"] = str(_resolve(comparison["results"], base))
        for detector in comparison.get("detectors", []):
            detector["channel"] = int(detector["channel"])

    thresholds = scan.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise EnergyScanError('"energy_scan.thresholds" must be non-empty.')
    scan["thresholds"] = [float(value) for value in thresholds]

    windows = scan.get("windows", [])
    scan["windows"] = [
        [float(window[0]), float(window[1])] for window in windows
    ]

    scan.setdefault(
        "adc_thresholds",
        [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340],
    )
    scan["adc_thresholds"] = [float(value) for value in scan["adc_thresholds"]]

    scan.setdefault(
        "adc_windows",
        [[20, 80], [80, 140], [140, 200], [200, 260], [260, 320]],
    )
    scan["adc_windows"] = [
        [float(window[0]), float(window[1])] for window in scan["adc_windows"]
    ]

    return config, config_path


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def na_points(detector: dict[str, Any]) -> tuple[list[float], list[float]]:
    path = Path(detector["results"])
    if not path.is_file():
        raise EnergyScanError(f"Energy fit result not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    means: dict[str, float] = {}
    for file_entry in data["files"]:
        for channel in file_entry["channels"]:
            if channel["channel"] != detector["channel"]:
                continue
            for histogram in channel["histograms"]:
                for fit in histogram["fits"]:
                    for parameter in fit["parameters"]:
                        if parameter["name"].lower() in ("mean", "mu"):
                            means[fit["name"]] = parameter["value"]

    centroids = []
    energies = []
    for point in detector["points"]:
        name = point["fit"]
        if name not in means:
            raise EnergyScanError(
                f"Fit {name!r} not found in {path}."
            )
        centroids.append(means[name])
        energies.append(float(point["energy"]))
    return centroids, energies


def energy_calibration(detector: dict[str, Any]) -> tuple[float, float]:
    centroids, energies = na_points(detector)
    slope, intercept = np.polyfit(centroids, energies, 1)
    return float(slope), float(intercept)


def co60_centroid(spec: dict[str, Any]) -> float:
    path = Path(spec["results"])
    if not path.is_file():
        raise EnergyScanError(f"Co-60 fit result not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for file_entry in data["files"]:
        for channel in file_entry["channels"]:
            if channel["channel"] != spec["channel"]:
                continue
            for histogram in channel["histograms"]:
                for fit in histogram["fits"]:
                    if fit["name"] != spec["fit"]:
                        continue
                    for parameter in fit["parameters"]:
                        if parameter["name"].lower() in ("mean", "mu"):
                            return float(parameter["value"])
    raise EnergyScanError(
        f"Co-60 fit {spec['fit']!r} not found in {path}."
    )


def plot_calibration(
    title: str,
    na_centroids: list[float],
    na_energies: list[float],
    slope: float,
    intercept: float,
    co60_centroid: float,
    co60_energy: float,
    edge_energies: list[float],
    path: Path,
    config: dict[str, Any],
) -> None:
    from plot import apply_style

    apply_style()
    canvas = ROOT.TCanvas("c_calibration", title, config["width"], config["height"])
    canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
    canvas.SetTicks(1, 1)
    canvas.SetFrameLineWidth(2)
    canvas.SetFrameLineColor(ROOT.kGray + 2)

    na_graph = ROOT.TGraph(len(na_centroids))
    for index, (centroid, energy) in enumerate(zip(na_centroids, na_energies)):
        na_graph.SetPoint(index, centroid, energy)
    na_graph.SetTitle(title)
    na_graph.SetMarkerStyle(20)
    na_graph.SetMarkerSize(1.4)
    na_graph.SetMarkerColor(ROOT.TColor.GetColor("#1F4E79"))
    na_graph.Draw("AP")

    high = max(max(na_centroids), co60_centroid) * 1.15
    function = ROOT.TF1("f_cal_line", "pol1", 0.0, high)
    function.SetParameter(0, intercept)
    function.SetParameter(1, slope)
    function.SetLineColor(ROOT.TColor.GetColor("#E63946"))
    function.SetLineWidth(3)
    function.Draw("same")

    co60_graph = ROOT.TGraph(1)
    co60_graph.SetPoint(0, co60_centroid, co60_energy)
    co60_graph.SetMarkerStyle(29)
    co60_graph.SetMarkerSize(2.2)
    co60_graph.SetMarkerColor(ROOT.TColor.GetColor("#E63946"))
    co60_graph.Draw("P same")

    for energy in edge_energies:
        line = ROOT.TLine(
            0.0, energy, high, energy
        )
        line.SetLineColor(ROOT.kGray + 1)
        line.SetLineStyle(3)
        line.Draw("same")

    na_graph.GetXaxis().SetTitle("ADC channel")
    na_graph.GetYaxis().SetTitle("Energy (keV)")
    na_graph.GetXaxis().SetTitleSize(0.045)
    na_graph.GetYaxis().SetTitleSize(0.045)
    na_graph.GetXaxis().SetLabelSize(0.040)
    na_graph.GetYaxis().SetLabelSize(0.040)

    box = ROOT.TPaveText(0.16, 0.72, 0.62, 0.89, "NDC")
    box.SetBorderSize(0)
    box.SetFillStyle(1001)
    box.SetFillColorAlpha(ROOT.kWhite, 0.85)
    box.SetTextAlign(12)
    box.SetTextFont(42)
    box.SetTextSize(0.034)
    box.AddText("Energy calibration (Na-22)")
    box.AddText(f"E = ({slope:.4f}) #times ch + ({intercept:.2f}) keV")
    box.AddText(
        f"Co-60 CE: ch = {co60_centroid:.1f}, "
        f"expected E #approx {co60_energy:.0f} keV"
    )
    box.Draw()

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def tac_calibration(path: Path) -> float:
    if not path.is_file():
        raise EnergyScanError(f"TAC calibration not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    slope = float(data["slope"])
    if slope <= 0:
        raise EnergyScanError(f"Invalid TAC slope in {path}.")
    return slope


def load_events(
    input_config: dict[str, Any],
    detector_channels: list[int],
    tac_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        files = read.find_input_files(input_config)
        source = [str(path) for path in files]
        if len(source) == 1:
            source = source[0]
        frame = ROOT.RDataFrame(
            input_config["tree"], source
        ).AsNumpy(columns=["Channel", "Timestamp", "Energy"])
    except Exception as exc:
        raise EnergyScanError(
            f"Cannot read {input_config['path']}: {exc}"
        ) from exc

    channels = np.asarray(frame["Channel"]).astype(np.int16)
    timestamps = np.asarray(frame["Timestamp"]).astype(np.int64)
    energy = np.asarray(frame["Energy"]).astype(np.float64)

    per_channel = {}
    for channel in detector_channels + [tac_channel]:
        mask = channels == channel
        stamps = timestamps[mask]
        order = np.argsort(stamps)
        per_channel[channel] = (stamps[order], energy[mask][order])

    common = np.intersect1d(
        per_channel[detector_channels[0]][0],
        per_channel[detector_channels[1]][0],
    )
    common = np.intersect1d(common, per_channel[tac_channel][0])

    values = []
    for channel in detector_channels + [tac_channel]:
        stamps, values_channel = per_channel[channel]
        indices = np.searchsorted(stamps, common)
        values.append(values_channel[indices])
    return values[0], values[1], values[2]


def fit_tac(
    values: np.ndarray,
    tac_range: list[float],
    bins: int,
    margin: float,
    slope: float,
    min_events: int,
    name: str,
    plot_path: Path | None = None,
    title: str = "",
) -> tuple[float | None, float | None]:
    low, high = tac_range
    selected = values[(values >= low) & (values <= high)]
    if selected.size < min_events:
        return None, None

    histogram = ROOT.TH1D(name, "", bins, low, high)
    histogram.SetDirectory(0)
    histogram.FillN(
        selected.size, selected, np.ones(selected.size, dtype=np.float64)
    )
    peak = histogram.GetXaxis().GetBinCenter(histogram.GetMaximumBin())
    fit_low = max(low, peak - margin)
    fit_high = min(high, peak + margin)
    if fit_high <= fit_low:
        return None, None

    function = ROOT.TF1(f"{name}_fit", "gaus", fit_low, fit_high)
    function.SetParameter(0, histogram.GetMaximum())
    function.SetParameter(1, peak)
    function.SetParameter(2, 40.0)
    result = histogram.Fit(function, "RSQ", "", fit_low, fit_high)
    fit_result = result.Get()
    if fit_result is None or fit_result.Status() != 0:
        return None, None

    sigma = float(function.GetParameter(2))
    sigma_error = float(function.GetParError(2))
    if not np.isfinite(sigma) or sigma <= 0:
        return None, None
    fwhm = FWHM_PER_SIGMA * sigma / slope
    fwhm_error = FWHM_PER_SIGMA * sigma_error / slope

    if plot_path is not None:
        from plot import apply_style

        apply_style()
        canvas = ROOT.TCanvas(f"c_{name}", title, 1200, 800)
        canvas.SetMargin(0.12, 0.05, 0.12, 0.10)
        histogram.SetLineColor(ROOT.TColor.GetColor("#1F4E79"))
        histogram.SetLineWidth(2)
        histogram.SetFillColor(ROOT.TColor.GetColor("#AED3F2"))
        histogram.SetFillColorAlpha(ROOT.TColor.GetColor("#AED3F2"), 0.35)
        histogram.Draw("hist")
        function.SetLineColor(ROOT.TColor.GetColor("#E63946"))
        function.SetLineWidth(4)
        function.Draw("same")
        histogram.GetXaxis().SetTitle("TAC channel")
        histogram.GetYaxis().SetTitle("Entries")
        histogram.SetTitle(title)

        box = ROOT.TPaveText(0.15, 0.72, 0.62, 0.89, "NDC")
        box.SetBorderSize(0)
        box.SetFillStyle(1001)
        box.SetFillColorAlpha(ROOT.kWhite, 0.85)
        box.SetTextAlign(12)
        box.SetTextFont(42)
        box.SetTextSize(0.034)
        box.AddText(f"Entries: {int(selected.size):,}")
        box.AddText(
            f"#sigma = {sigma:.2f} #pm {sigma_error:.2f} channel"
        )
        box.AddText(f"FWHM = {fwhm:.3f} ns")
        box.Draw()

        canvas.SetGridx()
        canvas.SetGridy()
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs(str(plot_path))
        canvas.Close()

    return fwhm, fwhm_error


def plot_scan(
    x: list[float],
    y: list[float | None],
    y_error: list[float | None],
    x_error: list[float],
    path: Path,
    title: str,
    x_title: str,
    config: dict[str, Any],
    x_limits: list[float] | None = None,
) -> None:
    from plot import apply_style

    apply_style()
    canvas = ROOT.TCanvas("c_energy_scan", title, config["width"], config["height"])
    canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
    canvas.SetTicks(1, 1)
    canvas.SetFrameLineWidth(2)
    canvas.SetFrameLineColor(ROOT.kGray + 2)

    points = [
        (xv, yv, yev, xev)
        for xv, yv, yev, xev in zip(x, y, y_error, x_error)
        if yv is not None
    ]
    graph = ROOT.TGraphErrors(len(points))
    for index, (xv, yv, yev, xev) in enumerate(points):
        graph.SetPoint(index, xv, yv)
        graph.SetPointError(index, xev, yev or 0.0)
    graph.SetTitle(title)
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.4)
    graph.SetMarkerColor(ROOT.TColor.GetColor("#1F4E79"))
    graph.SetLineColor(ROOT.TColor.GetColor("#1F4E79"))
    graph.SetLineWidth(2)
    if x_limits is not None:
        graph.GetXaxis().SetLimits(*x_limits)
    graph.Draw("AP")

    x_axis = graph.GetXaxis()
    x_axis.SetTitle(x_title)
    x_axis.SetTitleSize(0.045)
    x_axis.SetLabelSize(0.040)
    x_axis.CenterTitle(True)
    y_axis = graph.GetYaxis()
    y_axis.SetTitle("Time resolution FWHM (ns)")
    y_axis.SetTitleSize(0.045)
    y_axis.SetLabelSize(0.040)
    y_axis.CenterTitle(True)

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def _scan_set(
    first: np.ndarray,
    second: np.ndarray,
    tac: np.ndarray,
    thresholds: list[float],
    windows: list[list[float]],
    scan: dict[str, Any],
    tac_slope: float,
    tag: str,
    fit_dir: Path | None = None,
) -> dict[str, Any]:
    threshold_fwhm = []
    threshold_error = []
    for threshold in thresholds:
        mask = (first >= threshold) & (second >= threshold)
        plot_path = (
            fit_dir / f"{tag}_threshold_{int(threshold)}.png"
            if fit_dir is not None
            else None
        )
        fwhm, error = fit_tac(
            tac[mask],
            scan["tac_range"],
            scan["bins"],
            scan["fit_margin"],
            tac_slope,
            scan["min_events"],
            f"h_{tag}_thr_{int(threshold)}",
            plot_path,
            f"TAC fit: E > {threshold:.0f} keV",
        )
        threshold_fwhm.append(fwhm)
        threshold_error.append(error)

    window_centers = []
    window_half = []
    window_fwhm = []
    window_error = []
    for low, high in windows:
        mask = (
            (first >= low)
            & (first <= high)
            & (second >= low)
            & (second <= high)
        )
        plot_path = (
            fit_dir / f"{tag}_window_{int(low)}_{int(high)}.png"
            if fit_dir is not None
            else None
        )
        fwhm, error = fit_tac(
            tac[mask],
            scan["tac_range"],
            scan["bins"],
            scan["fit_margin"],
            tac_slope,
            scan["min_events"],
            f"h_{tag}_win_{int(low)}_{int(high)}",
            plot_path,
            f"TAC fit: {low:.0f}-{high:.0f} keV",
        )
        window_centers.append((low + high) / 2.0)
        window_half.append((high - low) / 2.0)
        window_fwhm.append(fwhm)
        window_error.append(error)

    return {
        "thresholds": thresholds,
        "threshold_fwhm": threshold_fwhm,
        "threshold_error": threshold_error,
        "window_centers": window_centers,
        "window_half_width": window_half,
        "window_fwhm": window_fwhm,
        "window_error": window_error,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    scan = config["energy_scan"]
    calibration = scan["calibration"]
    detectors = calibration["detectors"]
    channels = [detector["channel"] for detector in detectors]

    scales = []
    for detector in detectors:
        slope, intercept = energy_calibration(detector)
        scales.append((slope, intercept))

    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)

    co60 = scan.get("co60")
    if config["plot"]["enabled"] and isinstance(co60, dict):
        edge_energies = co60.get("edges", [963.0, 1118.0])
        for index, (detector, spec) in enumerate(
            zip(detectors, co60["detectors"]), start=1
        ):
            centroids, energies = na_points(detector)
            slope, intercept = scales[index - 1]
            centroid = co60_centroid(
                dict(spec, results=co60["results"])
            )
            plot_calibration(
                f"Energy calibration - Detector {index}",
                centroids,
                energies,
                slope,
                intercept,
                centroid,
                float(spec.get("energy", 1040.0)),
                edge_energies,
                output / f"calibration_det{index}.png",
                config["plot"],
            )

    for index, (slope, intercept) in enumerate(scales, start=1):
        print(
            f"Energy calibration detector {index}: "
            f"E = ({slope:.4f}) * ch + ({intercept:.2f}) keV"
        )

    tac_slope = tac_calibration(Path(calibration["tac"]))

    first, second, tac = load_events(
        config["input"], channels, int(scan["tac_channel"])
    )
    energy_first = scales[0][0] * first + scales[0][1]
    energy_second = scales[1][0] * second + scales[1][1]

    adc = _scan_set(
        first,
        second,
        tac,
        scan["adc_thresholds"],
        scan["adc_windows"],
        scan,
        tac_slope,
        "adc",
    )

    fit_dir = output / "fits"
    if config["plot"]["enabled"]:
        fit_dir.mkdir(parents=True, exist_ok=True)
        for old in fit_dir.glob("energy_*.png"):
            old.unlink()
    energy = _scan_set(
        energy_first,
        energy_second,
        tac,
        scan["thresholds"],
        scan["windows"],
        scan,
        tac_slope,
        "energy",
        fit_dir if config["plot"]["enabled"] else None,
    )

    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)

    config_plot = config["plot"]
    if config_plot["enabled"]:
        for old in output.glob("resolution_*.png"):
            old.unlink()
        plot_scan(
            adc["thresholds"],
            adc["threshold_fwhm"],
            adc["threshold_error"],
            [0.0] * len(adc["thresholds"]),
            output / "resolution_vs_threshold_adc.png",
            "Time resolution vs lower threshold (ADC)",
            "Lower threshold (ADC channel)",
            config_plot,
            [0.0, 4095.0],
        )
        plot_scan(
            energy["thresholds"],
            energy["threshold_fwhm"],
            energy["threshold_error"],
            [0.0] * len(energy["thresholds"]),
            output / "resolution_vs_threshold_energy.png",
            "Time resolution vs lower energy threshold",
            "Lower energy threshold (keV)",
            config_plot,
        )
        plot_scan(
            adc["window_centers"],
            adc["window_fwhm"],
            adc["window_error"],
            adc["window_half_width"],
            output / "resolution_vs_window_adc.png",
            "Time resolution vs window (ADC)",
            "Window center (ADC channel)",
            config_plot,
            [0.0, 4095.0],
        )
        plot_scan(
            energy["window_centers"],
            energy["window_fwhm"],
            energy["window_error"],
            energy["window_half_width"],
            output / "resolution_vs_window_energy.png",
            "Time resolution vs energy window",
            "Energy window center (keV)",
            config_plot,
        )

    result = {
        "adc": adc,
        "energy": energy,
        "energy_calibration": [
            {"channel": detector["channel"], "slope": slope, "intercept": intercept}
            for detector, (slope, intercept) in zip(detectors, scales)
        ],
        "tac_slope_channel_per_ns": tac_slope,
    }
    (output / "energy_scan.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for mode, data, unit in (
        ("ADC channel", adc, "ADC"),
        ("energy (keV)", energy, "keV"),
    ):
        print(f"\nTime resolution vs lower threshold [{mode}]")
        for threshold, fwhm, error in zip(
            data["thresholds"],
            data["threshold_fwhm"],
            data["threshold_error"],
        ):
            shown = (
                f"{fwhm:.3f} +/- {error:.3f}" if fwhm is not None else "no fit"
            )
            print(f"  > {threshold:7.1f} {unit} : {shown} ns")
        print(f"\nTime resolution vs window [{mode}]")
        for center, half, fwhm, error in zip(
            data["window_centers"],
            data["window_half_width"],
            data["window_fwhm"],
            data["window_error"],
        ):
            shown = (
                f"{fwhm:.3f} +/- {error:.3f}" if fwhm is not None else "no fit"
            )
            print(
                f"  [{center - half:.1f}, {center + half:.1f}] {unit} : "
                f"{shown} ns"
            )
    print(f"\nResults written to {output / 'energy_scan.json'}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time resolution as a function of energy range (Co-60)."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "config_energy_scan.json",
    )
    args = parser.parse_args(argv)

    ROOT.gROOT.SetBatch(True)
    try:
        config, _ = load_config(args.config)
        run(config)
    except (EnergyScanError, read.ConfigError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
