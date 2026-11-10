from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import ROOT

import read


FWHM_PER_SIGMA = 2.3548200450309493


class CfdError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise CfdError(f"Configuration file not found: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CfdError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise CfdError("The configuration root must be a JSON object.")

    input_config = config.get("input")
    if not isinstance(input_config, dict) or "path" not in input_config:
        raise CfdError('Missing "input.path".')
    base = config_path.parent
    input_config["path"] = _resolve(input_config["path"], base)
    input_config.setdefault("tree", "Data_R")

    output_config = config.get("output", "waves")
    if isinstance(output_config, dict):
        output_path = output_config.get("path", "waves")
        save_root = output_config.get("save_root", True)
    else:
        output_path = output_config
        save_root = config.get("save_root", True)
    output_path = Path(str(output_path)).expanduser()
    if output_path.is_absolute():
        raise CfdError('"output.path" must be relative to the output folder.')
    config["output"] = (read.OUTPUT_ROOT / output_path).resolve()
    config["save_root"] = save_root

    plot_config = config.setdefault("plot", {})
    plot_config.setdefault("enabled", True)
    plot_config.setdefault("width", 1600)
    plot_config.setdefault("height", 1000)

    cfd = config.get("cfd")
    if not isinstance(cfd, dict):
        raise CfdError('Missing "cfd" object.')
    cfd.setdefault("channels", [0, 1])
    cfd.setdefault("baseline_samples", 100)
    cfd.setdefault("scan_events", 5000)
    cfd.setdefault("fit_half_width", 3.0)
    cfd.setdefault("hist_half_width", 8.0)
    cfd.setdefault("bins", 200)
    cfd.setdefault("min_amplitude", 50.0)
    cfd.setdefault("amplitude_bins", 60)
    cfd.setdefault("feature_bins", 60)
    cfd.setdefault(
        "feature_fits",
        {
            "rise_time": [2.0, 5.0],
            "fall_time": [12.0, 20.0],
            "integral": [300.0, 500.0],
        },
    )
    for key, bounds in cfd["feature_fits"].items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise CfdError(
                f'"cfd.feature_fits.{key}" must contain two numbers.'
            )
        cfd["feature_fits"][key] = [
            _number(bounds[0], f"cfd.feature_fits.{key}"),
            _number(bounds[1], f"cfd.feature_fits.{key}"),
        ]
    cfd.setdefault("density_time_bins", 600)
    cfd.setdefault("density_amplitude_bins", 400)
    cfd.setdefault("density_amplitude_max", 1000.0)
    cfd.setdefault("refine_events", 30000)
    cfd.setdefault("refine_half_range", 1.0)
    cfd.setdefault("refine_step", 0.1)

    channels = cfd["channels"]
    if not isinstance(channels, list) or len(channels) != 2:
        raise CfdError('"cfd.channels" must contain two channel numbers.')

    delays = cfd.get("delays")
    if not isinstance(delays, list) or not delays:
        raise CfdError('"cfd.delays" must be a non-empty array.')
    cfd["delays"] = [_number(value, "cfd.delays") for value in delays]
    if any(delay <= 0 for delay in cfd["delays"]):
        raise CfdError('"cfd.delays" must be positive (ns).')

    fractions = cfd.get("fractions")
    if not isinstance(fractions, list) or not fractions:
        raise CfdError('"cfd.fractions" must be a non-empty array.')
    cfd["fractions"] = [
        _number(value, "cfd.fractions") for value in fractions
    ]

    for key in (
        "baseline_samples",
        "scan_events",
        "fit_half_width",
        "hist_half_width",
        "bins",
        "min_amplitude",
        "amplitude_bins",
        "feature_bins",
        "density_time_bins",
        "density_amplitude_bins",
        "density_amplitude_max",
        "refine_events",
        "refine_half_range",
        "refine_step",
    ):
        if key in (
            "baseline_samples",
            "scan_events",
            "bins",
            "amplitude_bins",
            "feature_bins",
            "density_time_bins",
            "density_amplitude_bins",
            "refine_events",
        ):
            cfd[key] = int(cfd[key])
        else:
            cfd[key] = float(cfd[key])

    calibration_source = config.get("energy_calibration")
    if calibration_source:
        calibration_path = _resolve(calibration_source, base)
        if not calibration_path.is_file():
            raise CfdError(
                f"Energy calibration file not found: {calibration_path}"
            )
        try:
            calibration = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise CfdError(
                f"Invalid JSON in {calibration_path}: {exc.msg}"
            ) from exc
        entries = calibration.get("energy_calibration")
        if not isinstance(entries, list):
            raise CfdError(
                f'Missing "energy_calibration" array in {calibration_path}'
            )
        calibration_values = {}
        for entry in entries:
            calibration_values[int(entry["channel"])] = (
                float(entry["slope"]),
                float(entry["intercept"]),
            )
        config["energy_calibration"] = calibration_values

    return config, config_path


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CfdError(f"{where} must contain numbers.")
    return float(value)


class Waveforms:
    def __init__(
        self,
        path: Path,
        tree_name: str,
        baseline_samples: int,
        channels: list[int],
    ) -> None:
        self.path = Path(path)
        self.file = ROOT.TFile.Open(str(path))
        if not self.file or self.file.IsZombie():
            raise CfdError(f"Cannot open {path}")
        self.tree = self.file.Get(tree_name)
        if self.tree is None:
            raise CfdError(f"Tree {tree_name!r} not found in {path}")
        self.baseline_samples = baseline_samples
        self.channels = channels
        self.tree.GetEntry(0)
        self.size = int(self.tree.Samples.GetSize())
        self.pairs, self.seconds = self._index()

    def _index(self) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        frame = ROOT.RDataFrame(
            self.tree.GetName(), str(self.path)
        ).AsNumpy(columns=["Channel", "Timestamp"])
        channels = np.asarray(frame["Channel"]).astype(np.int16)
        timestamps = np.asarray(frame["Timestamp"]).astype(np.int64)

        first, second = _channel_maps(channels, timestamps, self.channels)
        common = sorted(set(first) & set(second))
        pairs = [(stamp, first[stamp], second[stamp]) for stamp in common]
        seconds = (timestamps.max() - timestamps.min()) / 1e12
        return pairs, seconds

    def signal(self, entry: int) -> np.ndarray:
        self.tree.GetEntry(entry)
        return self.current_signal()

    def current_signal(self) -> np.ndarray:
        samples = np.frombuffer(
            self.tree.Samples.GetArray(),
            dtype=np.int16,
            count=self.size,
        )
        baseline = np.median(samples[: self.baseline_samples])
        return (baseline - samples).astype(np.float32)


def _channel_maps(
    channels: np.ndarray,
    timestamps: np.ndarray,
    pair: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    maps = [{}, {}]
    for channel, target in ((pair[0], maps[0]), (pair[1], maps[1])):
        for index in np.flatnonzero(channels == channel):
            target[int(timestamps[index])] = int(index)
    return maps[0], maps[1]


def crossing(
    signal: np.ndarray,
    delay: float,
    fraction: float,
    min_amplitude: float,
) -> float | None:
    peak = int(np.argmax(signal))
    amplitude = float(signal[peak])
    if amplitude < min_amplitude:
        return None

    onset = peak
    while onset > 0 and signal[onset - 1] > 0.05 * amplitude:
        onset -= 1

    start = max(int(np.ceil(delay)), onset - 1)
    end = min(len(signal), peak + int(np.ceil(delay)) + 5)
    indices = np.arange(start, end)
    if indices.size < 2:
        return None

    delayed = np.interp(
        indices - delay, np.arange(len(signal)), signal
    )
    bipolar = delayed - fraction * signal[indices]
    signs = (bipolar[:-1] < 0.0) & (bipolar[1:] >= 0.0)
    crossings = np.flatnonzero(signs)
    if crossings.size == 0:
        return None

    index = int(crossings[0])
    left = float(bipolar[index])
    right = float(bipolar[index + 1])
    if right == left:
        return float(indices[index])
    return float(indices[index]) + (-left) / (right - left)


def time_differences(
    waveforms: Waveforms,
    delay: int,
    fraction: float,
    min_amplitude: float,
    entries: list[tuple[int, int, int]],
    cache: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    values = []
    for position, (_, first, second) in enumerate(entries):
        if cache is not None:
            signal_a, signal_b = cache[position]
        else:
            signal_a = waveforms.signal(first)
            signal_b = waveforms.signal(second)
        time_a = crossing(signal_a, delay, fraction, min_amplitude)
        time_b = crossing(signal_b, delay, fraction, min_amplitude)
        if time_a is not None and time_b is not None:
            values.append(time_a - time_b)
    return np.asarray(values, dtype=np.float64)


def waveform_features(
    signal: np.ndarray,
) -> tuple[float, float, float] | None:
    peak = int(np.argmax(signal))
    amplitude = float(signal[peak])
    if amplitude <= 0:
        return None

    def leading(level: float) -> float | None:
        index = peak
        while index > 0 and signal[index - 1] >= level:
            index -= 1
        if index == 0:
            return None
        left = float(signal[index - 1])
        right = float(signal[index])
        if right == left:
            return float(index)
        return (index - 1) + (level - left) / (right - left)

    def trailing(level: float) -> float | None:
        index = peak
        end = len(signal) - 1
        while index < end and signal[index + 1] >= level:
            index += 1
        if index >= end:
            return None
        left = float(signal[index])
        right = float(signal[index + 1])
        if right == left:
            return float(index + 1)
        return index + (level - left) / (right - left)

    time_10 = leading(0.1 * amplitude)
    time_90 = leading(0.9 * amplitude)
    fall_90 = trailing(0.9 * amplitude)
    fall_10 = trailing(0.1 * amplitude)
    if None in (time_10, time_90, fall_90, fall_10):
        return None

    rise = float(time_90 - time_10)
    fall = float(fall_10 - fall_90)
    start = max(0, peak - 200)
    end = min(len(signal), peak + 1500)
    integral = float(np.sum(signal[start:end]))
    return rise, fall, integral


def integral_energy_factors(
    waveforms: Waveforms,
    channels: list[int],
    min_amplitude: float,
    sample_events: int = 20000,
) -> dict[int, float]:
    ratios: dict[int, list[float]] = {channel: [] for channel in channels}
    tree = waveforms.tree
    collected = 0
    for entry in range(int(tree.GetEntries())):
        if collected >= sample_events:
            break
        tree.GetEntry(entry)
        channel = int(tree.Channel)
        if channel not in ratios:
            continue
        energy = float(tree.Energy)
        signal = waveforms.current_signal()
        peak = int(np.argmax(signal))
        if float(signal[peak]) < min_amplitude or energy <= 0:
            continue
        integral = float(signal[max(0, peak - 200) : peak + 1500].sum())
        if integral <= 0:
            continue
        ratios[channel].append(energy / integral)
        collected += 1
    return {
        channel: float(np.median(values)) if values else 1.0
        for channel, values in ratios.items()
    }


def sequential_time_differences(
    waveforms: Waveforms,
    delay: float,
    fraction: float,
    min_amplitude: float,
    with_amplitudes: bool = False,
    with_features: bool = False,
):
    tree = waveforms.tree
    first, second = waveforms.channels
    maps = {first: {}, second: {}}
    amplitudes = {first: [], second: []}
    features = {first: [], second: []}
    for index in range(int(tree.GetEntries())):
        tree.GetEntry(index)
        channel = int(tree.Channel)
        if channel not in maps:
            continue
        signal = waveforms.current_signal()
        amplitudes[channel].append(float(signal.max()))
        if with_features:
            features[channel].append(waveform_features(signal))
        value = crossing(signal, delay, fraction, min_amplitude)
        if value is not None:
            maps[channel][int(tree.Timestamp)] = value

    common = maps[first].keys() & maps[second].keys()
    values = np.asarray(
        [maps[first][stamp] - maps[second][stamp] for stamp in common],
        dtype=np.float64,
    )
    result = [values]
    if with_amplitudes:
        result.extend(
            [
                np.asarray(amplitudes[first], dtype=np.float64),
                np.asarray(amplitudes[second], dtype=np.float64),
            ]
        )
    if with_features:
        result.extend([features[first], features[second]])
    return result[0] if len(result) == 1 else tuple(result)


def plot_waveform_features(
    features: dict[int, list[Any]],
    output: Path,
    config: dict[str, Any],
    bins: int = 60,
    energy_scale: dict[int, tuple[float, float]] | None = None,
    fit_ranges: dict[str, list[float]] | None = None,
) -> dict[str, dict[str, dict[str, float | None]]]:
    from plot import apply_style

    results: dict[str, dict[str, dict[str, float | None]]] = {}
    quantities = (
        (0, "Rise time", "Rise time 10-90% (ns)", "rise_time", "ns"),
        (1, "Fall time", "Fall time 90-10% (ns)", "fall_time", "ns"),
        (2, "Integral", "Pulse integral (ADC #times ns)", "integral", "keV"),
    )
    for channel in (0, 1):
        rows = [
            feature
            for feature in features[channel]
            if feature is not None
        ]
        if not rows:
            continue
        values = np.asarray(rows, dtype=np.float64)
        color = (
            ROOT.TColor.GetColor("#1F4E79")
            if channel == 0
            else ROOT.TColor.GetColor("#E63946")
        )
        fill = (
            ROOT.TColor.GetColor("#AED3F2")
            if channel == 0
            else ROOT.TColor.GetColor("#F4A6A0")
        )
        for column, title, x_title, name, unit in quantities:
            data = np.ascontiguousarray(values[:, column], dtype=np.float64)
            calibrated = column == 2 and energy_scale and channel in energy_scale
            if calibrated:
                scale, offset = energy_scale[channel]
                data = data * scale + offset
                x_title = "Pulse integral (keV)"
            limit = float(np.percentile(data, 99.5) * 1.15)
            if limit <= 0:
                continue
            apply_style()
            histogram = ROOT.TH1D(
                f"h_{name}_{channel}",
                f"{title} - Detector {channel + 1}",
                bins,
                0.0,
                limit,
            )
            histogram.SetDirectory(0)
            histogram.FillN(
                data.size, data, np.ones(data.size, dtype=np.float64)
            )
            canvas = ROOT.TCanvas(
                f"c_{name}_{channel}",
                title,
                config["width"],
                config["height"],
            )
            canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
            canvas.SetTicks(1, 1)
            histogram.SetLineColor(color)
            histogram.SetLineWidth(3)
            histogram.SetFillColor(fill)
            histogram.SetFillColorAlpha(fill, 0.40)
            histogram.GetXaxis().SetTitle(x_title)
            histogram.GetYaxis().SetTitle("Entries")
            histogram.GetXaxis().SetTitleSize(0.045)
            histogram.GetYaxis().SetTitleSize(0.045)
            histogram.GetXaxis().SetLabelSize(0.040)
            histogram.GetYaxis().SetLabelSize(0.040)
            histogram.Draw("hist")

            fit_range = None
            if fit_ranges and name in fit_ranges:
                if name != "integral" or calibrated:
                    fit_range = fit_ranges[name]
            if fit_range:
                low_fit = max(
                    float(fit_range[0]), histogram.GetXaxis().GetXmin()
                )
                high_fit = min(
                    float(fit_range[1]), histogram.GetXaxis().GetXmax()
                )
                if high_fit > low_fit:
                    low_bin = histogram.GetXaxis().FindBin(low_fit)
                    high_bin = histogram.GetXaxis().FindBin(high_fit)
                    peak_bin = max(
                        range(low_bin, high_bin + 1),
                        key=histogram.GetBinContent,
                    )
                    amplitude = histogram.GetBinContent(peak_bin)
                    function = ROOT.TF1(
                        f"f_{name}_{channel}", "gaus", low_fit, high_fit
                    )
                    function.SetParameter(0, amplitude)
                    function.SetParameter(1, histogram.GetBinCenter(peak_bin))
                    function.SetParameter(2, (high_fit - low_fit) / 6.0)
                    function.SetParLimits(0, 0.0, max(amplitude, 1.0) * 10.0)
                    function.SetParLimits(
                        2, 1e-6, high_fit - low_fit
                    )
                    histogram.Fit(function, "RSQ", "", low_fit, high_fit)
                    mean = float(function.GetParameter(1))
                    mean_error = float(function.GetParError(1))
                    sigma = float(function.GetParameter(2))
                    sigma_error = float(function.GetParError(2))
                    ndf = int(function.GetNDF())
                    chi2_ndf = (
                        float(function.GetChisquare() / ndf) if ndf > 0 else None
                    )
                    if (
                        np.isfinite(mean)
                        and np.isfinite(sigma)
                        and sigma > 0.0
                        and low_fit < mean < high_fit
                    ):
                        function.SetLineColor(ROOT.kBlack)
                        function.SetLineWidth(4)
                        function.Draw("same")
                        box = ROOT.TPaveText(0.52, 0.70, 0.93, 0.89, "NDC")
                        box.SetBorderSize(0)
                        box.SetFillStyle(1001)
                        box.SetFillColorAlpha(ROOT.kWhite, 0.85)
                        box.SetTextAlign(12)
                        box.SetTextFont(42)
                        box.SetTextSize(0.034)
                        box.AddText("Gaussian fit")
                        box.AddText(
                            f"#mu = {mean:.3f} #pm {mean_error:.3f} {unit}"
                        )
                        box.AddText(
                            f"#sigma = {sigma:.3f} #pm {sigma_error:.3f} {unit}"
                        )
                        box.Draw()
                        results.setdefault(name, {})[
                            f"det{channel + 1}"
                        ] = {
                            "mean": mean,
                            "mean_error": mean_error,
                            "sigma": sigma,
                            "sigma_error": sigma_error,
                            "chi2_ndf": chi2_ndf,
                        }

            canvas.SetGridx()
            canvas.SetGridy()
            canvas.Modified()
            canvas.Update()
            canvas.SaveAs(str(output / f"{name}_det{channel + 1}.png"))
            canvas.Close()

    return results


def plot_waveform_density(
    waveforms: Waveforms,
    output: Path,
    config: dict[str, Any],
    min_amplitude: float = 50.0,
    window: tuple[float, float] = (-100.0, 300.0),
    time_bins: int = 600,
    amplitude_bins: int = 400,
    amplitude_max: float = 1000.0,
) -> None:
    from plot import apply_style

    low, high = int(window[0]), int(window[1])
    offsets = np.arange(low, high + 1, dtype=np.float64)
    weights = np.ones(offsets.size, dtype=np.float64)
    histograms = {}
    amplitudes = {0: [], 1: []}
    for channel in (0, 1):
        histogram = ROOT.TH2D(
            f"h_wave_density_{channel}",
            f"Waveform density - Detector {channel + 1}",
            time_bins,
            low,
            high,
            amplitude_bins,
            -20.0,
            amplitude_max,
        )
        histogram.SetDirectory(0)
        histograms[channel] = histogram

    tree = waveforms.tree
    for entry in range(int(tree.GetEntries())):
        tree.GetEntry(entry)
        channel = int(tree.Channel)
        if channel not in histograms:
            continue
        signal = waveforms.current_signal()
        peak = int(np.argmax(signal))
        amplitude = float(signal[peak])
        if amplitude < min_amplitude:
            continue
        indices = np.arange(low, high + 1) + peak
        if indices[0] < 0 or indices[-1] >= signal.size:
            continue
        values = np.ascontiguousarray(signal[indices], dtype=np.float64)
        histograms[channel].FillN(offsets.size, offsets, values, weights)
        amplitudes[channel].append(amplitude)

    for channel in (0, 1):
        histogram = histograms[channel]
        if histogram.GetEntries() == 0:
            continue
        display_max = float(
            np.percentile(amplitudes[channel], 99.5) * 1.08
        )
        histogram.Smooth(1)
        histogram.GetYaxis().SetRangeUser(-20.0, display_max)
        apply_style()
        for log_z, suffix in ((False, ""), (True, "_log")):
            canvas = ROOT.TCanvas(
                f"c_wave_density_{channel}{suffix}",
                f"Waveform density detector {channel + 1}",
                config["width"],
                config["height"],
            )
            canvas.SetMargin(0.14, 0.14, 0.13, 0.09)
            canvas.SetTicks(1, 1)
            histogram.GetXaxis().SetTitle("Time relative to peak (ns)")
            histogram.GetYaxis().SetTitle("Amplitude (ADC counts)")
            histogram.GetZaxis().SetTitle("Entries")
            histogram.GetXaxis().SetTitleSize(0.045)
            histogram.GetYaxis().SetTitleSize(0.045)
            histogram.GetZaxis().SetTitleSize(0.045)
            histogram.GetXaxis().SetLabelSize(0.040)
            histogram.GetYaxis().SetLabelSize(0.040)
            histogram.GetZaxis().SetLabelSize(0.040)
            if log_z:
                histogram.SetMinimum(1.0)
                canvas.SetLogz()
            histogram.Draw("COLZ")
            canvas.Modified()
            canvas.Update()
            canvas.SaveAs(
                str(output / f"waveform_density_det{channel + 1}{suffix}.png")
            )
            canvas.Close()


def plot_scan_heatmap(
    results: list[dict[str, Any]],
    best: tuple[float, float, float],
    path: Path,
    config: dict[str, Any],
    log_z: bool = False,
) -> None:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from plot import apply_style

    points = [
        (entry["delay"], entry["fraction"], entry["fwhm"])
        for entry in results
        if entry["fwhm"] is not None and np.isfinite(entry["fwhm"])
    ]
    if not points:
        return

    delays = sorted({entry["delay"] for entry in results})
    fractions = sorted({entry["fraction"] for entry in results})
    x = np.asarray([point[0] for point in points])
    y = np.asarray([point[1] for point in points])
    z = np.asarray([point[2] for point in points])

    grid_x = np.linspace(delays[0], delays[-1], 451)
    grid_y = np.linspace(fractions[0], fractions[-1], 101)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata(
        np.column_stack([x, y]), z, (mesh_x, mesh_y), method="cubic"
    )
    missing = ~np.isfinite(grid_z)
    if missing.any():
        nearest = griddata(
            np.column_stack([x, y]), z, (mesh_x, mesh_y), method="nearest"
        )
        grid_z[missing] = nearest[missing]
    grid_z = gaussian_filter(grid_z, sigma=(2.0, 4.0))

    apply_style()
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    canvas = ROOT.TCanvas(
        "c_scan_heatmap",
        "Digital CFD scan",
        config["width"],
        config["height"],
    )
    canvas.SetMargin(0.13, 0.13, 0.13, 0.09)

    suffix = " (smoothed, log scale)" if log_z else " (smoothed)"
    histogram = ROOT.TH2D(
        "h_cfd_heatmap",
        f"Digital CFD: time resolution map{suffix};"
        "Delay (ns);Fraction F;FWHM (ns)",
        len(grid_x) - 1,
        grid_x[0],
        grid_x[-1],
        len(grid_y) - 1,
        grid_y[0],
        grid_y[-1],
    )
    for row in range(len(grid_y)):
        for column in range(len(grid_x)):
            value = grid_z[row, column]
            histogram.SetBinContent(
                column + 1, row + 1, float(value)
            )
    histogram.SetStats(False)
    if log_z:
        histogram.GetZaxis().SetRangeUser(
            max(0.3, float(np.min(grid_z))),
            min(2.0, float(np.percentile(grid_z, 99))),
        )
    else:
        histogram.GetZaxis().SetRangeUser(
            max(0.0, float(np.min(grid_z)) - 0.05),
            min(1.2, float(np.percentile(grid_z, 99))),
        )
    histogram.Draw("COLZ")
    if log_z:
        canvas.SetLogz()

    marker = ROOT.TMarker(best[1], best[2], ROOT.kOpenCircle)
    marker.SetMarkerColor(ROOT.kBlack)
    marker.SetMarkerSize(2.0)
    marker.Draw("same")

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def plot_amplitudes(
    first: np.ndarray,
    second: np.ndarray,
    output: Path,
    config: dict[str, Any],
    bins: int = 60,
) -> None:
    from plot import apply_style

    for channel, values in ((0, first), (1, second)):
        if values.size == 0:
            continue
        apply_style()
        limit = float(values.max()) * 1.02
        histogram = ROOT.TH1D(
            f"h_amp_{channel}",
            f"Pulse amplitude distribution - Detector {channel + 1}",
            bins,
            0.0,
            limit,
        )
        histogram.SetDirectory(0)
        histogram.FillN(
            values.size, values, np.ones(values.size, dtype=np.float64)
        )

        canvas = ROOT.TCanvas(
            f"c_amplitude_{channel}",
            f"Amplitude detector {channel + 1}",
            config["width"],
            config["height"],
        )
        canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
        canvas.SetTicks(1, 1)

        color = (
            ROOT.TColor.GetColor("#1F4E79")
            if channel == 0
            else ROOT.TColor.GetColor("#E63946")
        )
        fill = (
            ROOT.TColor.GetColor("#AED3F2")
            if channel == 0
            else ROOT.TColor.GetColor("#F4A6A0")
        )
        histogram.SetLineColor(color)
        histogram.SetLineWidth(3)
        histogram.SetFillColor(fill)
        histogram.SetFillColorAlpha(fill, 0.40)
        histogram.GetXaxis().SetTitle("Pulse amplitude (ADC channel)")
        histogram.GetYaxis().SetTitle("Entries")
        histogram.GetXaxis().SetTitleSize(0.045)
        histogram.GetYaxis().SetTitleSize(0.045)
        histogram.GetXaxis().SetLabelSize(0.040)
        histogram.GetYaxis().SetLabelSize(0.040)
        histogram.Draw("hist")

        canvas.SetGridx()
        canvas.SetGridy()
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs(str(output / f"amplitude_det{channel + 1}.png"))
        canvas.Close()


def gaussian_width(
    values: np.ndarray,
    center: float,
    histogram_half_width: float,
    fit_half_width: float,
    bins: int,
    name: str,
) -> tuple[float, float, ROOT.TH1D | None, ROOT.TF1 | None]:
    histogram = ROOT.TH1D(
        name,
        "",
        bins,
        center - histogram_half_width,
        center + histogram_half_width,
    )
    histogram.SetDirectory(0)
    selected = values[
        np.abs(values - center) <= histogram_half_width
    ]
    inside_fit = values[np.abs(values - center) <= fit_half_width]
    if inside_fit.size < 100:
        return float("inf"), float("nan"), histogram, None

    if selected.size:
        histogram.FillN(
            selected.size,
            selected,
            np.ones(selected.size, dtype=np.float64),
        )

    function = ROOT.TF1(
        f"{name}_fit",
        "gaus",
        center - fit_half_width,
        center + fit_half_width,
    )
    function.SetParameter(0, histogram.GetMaximum())
    function.SetParameter(1, center)
    function.SetParameter(2, 0.5)
    result = histogram.Fit(
        function,
        "RSQ",
        "",
        center - fit_half_width,
        center + fit_half_width,
    )
    fit_result = result.Get()
    if fit_result is None or fit_result.Status() != 0:
        return float("inf"), float("nan"), histogram, function

    sigma = float(function.GetParameter(2))
    sigma_error = float(function.GetParError(2))
    chi2_ndf = function.GetChisquare() / max(int(function.GetNDF()), 1)
    if (
        not np.isfinite(sigma)
        or sigma <= 0.15
        or sigma >= fit_half_width
        or chi2_ndf > 100.0
    ):
        return float("inf"), float("nan"), histogram, function
    return FWHM_PER_SIGMA * sigma, FWHM_PER_SIGMA * sigma_error, histogram, function


def median_center(
    waveforms: Waveforms,
    entries: list[tuple[int, int, int]],
    cache: list[tuple[np.ndarray, np.ndarray]],
    delay: float,
    fraction: float,
    min_amplitude: float,
) -> float:
    values = time_differences(
        waveforms, delay, fraction, min_amplitude, entries, cache
    )
    if values.size == 0:
        raise CfdError("No CFD zero crossings found.")
    return float(np.median(values))


def scan(
    waveforms: Waveforms,
    entries: list[tuple[int, int, int]],
    cache: list[tuple[np.ndarray, np.ndarray]],
    center: float,
    cfd: dict[str, Any],
) -> tuple[float, float, float, list[dict[str, Any]]]:
    results = []
    best = (float("inf"), cfd["delays"][0], cfd["fractions"][0])
    for delay in cfd["delays"]:
        for fraction in cfd["fractions"]:
            values = time_differences(
                waveforms,
                delay,
                fraction,
                cfd["min_amplitude"],
                entries,
                cache,
            )
            if values.size < 100:
                fwhm, fwhm_error = float("inf"), float("nan")
                local_center = center
            else:
                local_center = float(np.median(values))
                fwhm, fwhm_error, _, _ = gaussian_width(
                    values,
                    local_center,
                    cfd["hist_half_width"],
                    cfd["fit_half_width"],
                    cfd["bins"],
                    f"h_scan_{str(delay).replace('.', '_')}_"
                    f"{str(fraction).replace('.', '_')}",
                )
                if fwhm < 0.15:
                    fwhm, fwhm_error = float("inf"), float("nan")
            results.append(
                {
                    "delay": delay,
                    "fraction": fraction,
                    "center": local_center,
                    "fwhm": fwhm,
                    "fwhm_error": fwhm_error,
                }
            )
            if fwhm < best[0]:
                best = (fwhm, delay, fraction)
    return best[0], best[1], best[2], results


def refine_scan(
    waveforms: Waveforms,
    fraction: float,
    delay: float,
    cfd: dict[str, Any],
) -> tuple[float, float, list[dict[str, Any]]]:
    entries = waveforms.pairs[: cfd["refine_events"]]
    cache = [
        (waveforms.signal(first), waveforms.signal(second))
        for _, first, second in entries
    ]
    lower = max(1.0, delay - cfd["refine_half_range"])
    upper = delay + cfd["refine_half_range"]
    step = cfd["refine_step"]
    delays = np.round(np.arange(lower, upper + 0.5 * step, step), 3)
    results = []
    for value in delays:
        values = time_differences(
            waveforms,
            float(value),
            fraction,
            cfd["min_amplitude"],
            entries,
            cache,
        )
        if values.size < 100:
            continue
        center = float(np.median(values))
        fwhm, fwhm_error, _, _ = gaussian_width(
            values,
            center,
            cfd["hist_half_width"],
            cfd["fit_half_width"],
            cfd["bins"],
            f"h_refine_{str(value).replace('.', '_')}",
        )
        if not np.isfinite(fwhm) or fwhm < 0.15:
            continue
        results.append(
            {
                "delay": float(value),
                "center": center,
                "fwhm": fwhm,
                "fwhm_error": fwhm_error,
            }
        )

    if not results:
        return delay, float("inf"), results

    results.sort(key=lambda entry: entry["delay"])
    best_index = int(np.argmin([entry["fwhm"] for entry in results]))
    best = results[best_index]
    neighbors = results[max(0, best_index - 2) : best_index + 3]
    if len(neighbors) >= 3:
        x = np.asarray([entry["delay"] for entry in neighbors])
        y = np.asarray([entry["fwhm"] for entry in neighbors])
        coefficients = np.polyfit(x, y, 2)
        if coefficients[0] > 0:
            vertex = -coefficients[1] / (2.0 * coefficients[0])
            if x[0] <= vertex <= x[-1]:
                return float(vertex), float(best["fwhm"]), results
    return float(best["delay"]), float(best["fwhm"]), results


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _amplitude_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"entries": 0}
    return {
        "entries": int(values.size),
        "mean": _finite(float(np.mean(values))),
        "median": _finite(float(np.median(values))),
        "max": _finite(float(np.max(values))),
    }


def _bin_edges(values: list[float]) -> np.ndarray:
    ordered = sorted(values)
    if len(ordered) == 1:
        return np.asarray([ordered[0] - 0.5, ordered[0] + 0.5])
    edges = [ordered[0] - (ordered[1] - ordered[0]) / 2.0]
    for left, right in zip(ordered, ordered[1:]):
        edges.append((left + right) / 2.0)
    edges.append(ordered[-1] + (ordered[-1] - ordered[-2]) / 2.0)
    return np.asarray(edges, dtype=np.float64)


def _scan_histogram(
    results: list[dict[str, Any]],
    name: str,
    title: str,
) -> ROOT.TH2D:
    delays = sorted({entry["delay"] for entry in results})
    fractions = sorted({entry["fraction"] for entry in results})
    histogram = ROOT.TH2D(
        name,
        title,
        len(delays),
        _bin_edges(delays),
        len(fractions),
        _bin_edges(fractions),
    )
    histogram.SetDirectory(0)
    for entry in results:
        if not np.isfinite(entry["fwhm"]):
            continue
        histogram.Fill(entry["delay"], entry["fraction"], entry["fwhm"])
    histogram.SetMinimum(0.0)
    return histogram


def plot_scan(
    results: list[dict[str, Any]],
    best: tuple[float, float, float],
    path: Path,
    config: dict[str, Any],
) -> None:
    from plot import apply_style

    apply_style()
    canvas = ROOT.TCanvas("c_scan", "CFD scan", config["width"], config["height"])
    canvas.SetMargin(0.13, 0.12, 0.13, 0.09)
    canvas.SetTicks(1, 1)

    histogram = _scan_histogram(
        results,
        "h_cfd_scan",
        "CFD parameter scan;Delay (ns);Fraction F;FWHM (ns)",
    )
    histogram.GetZaxis().SetRangeUser(0.0, 1.0)
    histogram.GetZaxis().SetTitle("FWHM (ns)")
    histogram.Draw("COLZ")

    marker = ROOT.TMarker(best[1], best[2], ROOT.kOpenCircle)
    marker.SetMarkerColor(ROOT.kRed)
    marker.SetMarkerSize(2.0)
    marker.Draw("same")

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def plot_scan_3d(
    results: list[dict[str, Any]],
    best: tuple[float, float, float],
    path: Path,
    config: dict[str, Any],
) -> None:
    from plot import apply_style

    apply_style()
    histogram = _scan_histogram(
        results,
        "h_cfd_scan_3d",
        "Digital CFD: time resolution vs delay and fraction;"
        "Delay (ns);Fraction F;FWHM (ns)",
    )
    histogram.SetStats(False)
    histogram.GetZaxis().SetRangeUser(0.3, 1.0)

    ROOT.gStyle.SetPalette(ROOT.kBird)
    canvas = ROOT.TCanvas(
        "c_scan_3d", "Digital CFD scan", config["width"], config["height"]
    )
    canvas.SetTheta(25)
    canvas.SetPhi(135)
    histogram.Draw("LEGO2")

    marker = ROOT.TMarker3DBox(
        best[1],
        best[2],
        best[0],
        0.2,
        0.01,
        0.02,
        30.0,
        30.0,
    )
    marker.SetFillColor(ROOT.kRed)
    marker.SetLineColor(ROOT.kRed)
    marker.Draw("same")

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def plot_resolution(
    histogram: ROOT.TH1D,
    function: ROOT.TF1,
    path: Path,
    config: dict[str, Any],
    sigma: float,
    sigma_error: float,
) -> None:
    from plot import apply_style

    apply_style()
    canvas = ROOT.TCanvas(
        "c_resolution", "Digital CFD time resolution", config["width"], config["height"]
    )
    canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
    canvas.SetTicks(1, 1)
    canvas.SetFrameLineWidth(2)
    canvas.SetFrameLineColor(ROOT.kGray + 2)

    histogram.SetLineColor(ROOT.TColor.GetColor("#1F4E79"))
    histogram.SetLineWidth(2)
    histogram.SetFillColor(ROOT.TColor.GetColor("#AED3F2"))
    histogram.SetFillColorAlpha(ROOT.TColor.GetColor("#AED3F2"), 0.55)
    histogram.Draw("hist")

    function.SetLineColor(ROOT.TColor.GetColor("#E63946"))
    function.SetLineWidth(3)
    function.Draw("same")

    histogram.GetXaxis().SetTitle("t_{0} - t_{1} (ns)")
    histogram.GetYaxis().SetTitle("Entries")
    histogram.GetXaxis().SetTitleSize(0.045)
    histogram.GetYaxis().SetTitleSize(0.045)
    histogram.GetXaxis().SetLabelSize(0.040)
    histogram.GetYaxis().SetLabelSize(0.040)

    fwhm = FWHM_PER_SIGMA * sigma
    box = ROOT.TPaveText(0.15, 0.72, 0.60, 0.89, "NDC")
    box.SetBorderSize(0)
    box.SetFillStyle(1001)
    box.SetFillColorAlpha(ROOT.kWhite, 0.85)
    box.SetTextAlign(12)
    box.SetTextFont(42)
    box.SetTextSize(0.034)
    box.AddText("Digital CFD")
    box.AddText(f"#sigma = {sigma:.3f} #pm {sigma_error:.3f} ns")
    box.AddText(f"FWHM = {fwhm:.3f} ns")
    box.Draw()

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
    canvas.Close()


def run(config: dict[str, Any]) -> dict[str, Any]:
    cfd = config["cfd"]
    waveforms = Waveforms(
        config["input"]["path"],
        config["input"]["tree"],
        cfd["baseline_samples"],
        cfd["channels"],
    )
    if not waveforms.pairs:
        raise CfdError("No coincident events found.")

    scan_entries = waveforms.pairs[: cfd["scan_events"]]
    cache = [
        (waveforms.signal(first), waveforms.signal(second))
        for _, first, second in scan_entries
    ]

    reference_delay = cfd["delays"][len(cfd["delays"]) // 2]
    reference_fraction = cfd["fractions"][len(cfd["fractions"]) // 2]
    center = median_center(
        waveforms,
        scan_entries,
        cache,
        reference_delay,
        reference_fraction,
        cfd["min_amplitude"],
    )

    fwhm, delay, fraction, scan_results = scan(
        waveforms, scan_entries, cache, center, cfd
    )
    if not np.isfinite(fwhm):
        raise CfdError("CFD scan did not converge.")

    refined_delay, refined_fwhm, refinement = refine_scan(
        waveforms, fraction, delay, cfd
    )
    if np.isfinite(refined_fwhm):
        delay = refined_delay
        fwhm = refined_fwhm

    (
        values,
        amplitudes_first,
        amplitudes_second,
        features_first,
        features_second,
    ) = sequential_time_differences(
        waveforms,
        delay,
        fraction,
        cfd["min_amplitude"],
        with_amplitudes=True,
        with_features=True,
    )
    threshold = float(cfd["min_amplitude"])
    mask_first = amplitudes_first >= threshold
    mask_second = amplitudes_second >= threshold
    features_first = [
        feature
        for feature, keep in zip(features_first, mask_first)
        if keep and feature is not None
    ]
    features_second = [
        feature
        for feature, keep in zip(features_second, mask_second)
        if keep and feature is not None
    ]
    amplitudes_first = amplitudes_first[mask_first]
    amplitudes_second = amplitudes_second[mask_second]

    if values.size:
        center = float(np.median(values))
    fwhm, fwhm_error, histogram, function = gaussian_width(
        values,
        center,
        cfd["hist_half_width"],
        cfd["fit_half_width"],
        cfd["bins"],
        "h_resolution",
    )
    if not np.isfinite(fwhm):
        raise CfdError(
            f"Final fit failed for delay={delay} ns, F={fraction}."
        )

    sigma = fwhm / FWHM_PER_SIGMA
    sigma_error = fwhm_error / FWHM_PER_SIGMA

    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)

    energy_scale = None
    calibration = config.get("energy_calibration")
    if calibration:
        factors = integral_energy_factors(
            waveforms, cfd["channels"], cfd["min_amplitude"]
        )
        energy_scale = {
            channel: (slope * factors[channel], intercept)
            for channel, (slope, intercept) in calibration.items()
        }

    feature_fits: dict[str, dict[str, dict[str, float | None]]] = {}
    if config["plot"]["enabled"]:
        for old in output.glob("*.png"):
            old.unlink()
        plot_scan(
            scan_results,
            (fwhm, delay, fraction),
            output / "cfd_scan.png",
            config["plot"],
        )
        plot_scan_3d(
            scan_results,
            (fwhm, delay, fraction),
            output / "cfd_scan_3d.png",
            config["plot"],
        )
        plot_scan_heatmap(
            scan_results,
            (fwhm, delay, fraction),
            output / "cfd_scan_heatmap.png",
            config["plot"],
        )
        plot_scan_heatmap(
            scan_results,
            (fwhm, delay, fraction),
            output / "cfd_scan_heatmap_log.png",
            config["plot"],
            log_z=True,
        )
        plot_amplitudes(
            amplitudes_first,
            amplitudes_second,
            output,
            config["plot"],
            cfd["amplitude_bins"],
        )
        plot_waveform_density(
            waveforms,
            output,
            config["plot"],
            cfd["min_amplitude"],
            time_bins=cfd["density_time_bins"],
            amplitude_bins=cfd["density_amplitude_bins"],
            amplitude_max=cfd["density_amplitude_max"],
        )
        feature_fits = plot_waveform_features(
            {
                0: features_first,
                1: features_second,
            },
            output,
            config["plot"],
            cfd["feature_bins"],
            energy_scale,
            cfd["feature_fits"],
        )
        if histogram is not None and function is not None:
            chi2_ndf = (
                function.GetChisquare() / function.GetNDF()
                if function.GetNDF() > 0
                else None
            )
            shown = (
                f"{chi2_ndf:.2f}" if chi2_ndf is not None else "undefined"
            )
            plot_resolution(
                histogram,
                function,
                output / f"time_resolution_chi2ndf_{shown}.png",
                config["plot"],
                sigma,
                sigma_error,
            )

    result = {
        "events": len(waveforms.pairs),
        "scan_events": len(scan_entries),
        "seconds": waveforms.seconds,
        "delay": delay,
        "fraction": fraction,
        "center": center,
        "sigma": sigma,
        "sigma_error": sigma_error,
        "fwhm": fwhm,
        "fwhm_error": fwhm_error,
        "refinement": refinement,
        "amplitude": {
            "channel0": _amplitude_stats(amplitudes_first),
            "channel1": _amplitude_stats(amplitudes_second),
        },
        "feature_fits": feature_fits,
        "scan": [
            {
                "delay": entry["delay"],
                "fraction": entry["fraction"],
                "center": _finite(entry["center"]),
                "fwhm": _finite(entry["fwhm"]),
                "fwhm_error": _finite(entry["fwhm_error"]),
            }
            for entry in scan_results
        ],
    }
    (output / "waves.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nDigital CFD ({len(waveforms.pairs)} coincidences)")
    print(f"  optimal D = {delay} samples, F = {fraction:.1f}")
    print(f"  sigma = {sigma:.3f} +/- {sigma_error:.3f} ns")
    print(f"  FWHM = {fwhm:.3f} +/- {fwhm_error:.3f} ns")
    print(f"  Results written to {output / 'waves.json'}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Digital CFD time resolution from digitizer waveforms."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "config_waves.json",
    )
    args = parser.parse_args(argv)

    ROOT.gROOT.SetBatch(True)
    try:
        config, _ = load_config(args.config)
        run(config)
    except (CfdError, read.ConfigError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
