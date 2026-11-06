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
    ):
        if key in ("baseline_samples", "scan_events", "bins"):
            cfd[key] = int(cfd[key])
        else:
            cfd[key] = float(cfd[key])

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


def sequential_time_differences(
    waveforms: Waveforms,
    delay: float,
    fraction: float,
    min_amplitude: float,
    with_amplitudes: bool = False,
):
    tree = waveforms.tree
    first, second = waveforms.channels
    maps = {first: {}, second: {}}
    amplitudes = {first: [], second: []}
    for index in range(int(tree.GetEntries())):
        tree.GetEntry(index)
        channel = int(tree.Channel)
        if channel not in maps:
            continue
        signal = waveforms.current_signal()
        amplitudes[channel].append(float(signal.max()))
        value = crossing(signal, delay, fraction, min_amplitude)
        if value is not None:
            maps[channel][int(tree.Timestamp)] = value

    common = maps[first].keys() & maps[second].keys()
    values = np.asarray(
        [maps[first][stamp] - maps[second][stamp] for stamp in common],
        dtype=np.float64,
    )
    if with_amplitudes:
        return (
            values,
            np.asarray(amplitudes[first], dtype=np.float64),
            np.asarray(amplitudes[second], dtype=np.float64),
        )
    return values


def plot_amplitudes(
    first: np.ndarray,
    second: np.ndarray,
    path: Path,
    config: dict[str, Any],
) -> None:
    from plot import apply_style

    apply_style()
    limit = float(
        np.percentile(np.concatenate([first, second]), 99.5) * 1.15
    )
    bins = 300
    histogram_first = ROOT.TH1D("h_amp_first", "", bins, 0.0, limit)
    histogram_second = ROOT.TH1D("h_amp_second", "", bins, 0.0, limit)
    for histogram, values in (
        (histogram_first, first),
        (histogram_second, second),
    ):
        histogram.SetDirectory(0)
        histogram.FillN(
            values.size, values, np.ones(values.size, dtype=np.float64)
        )

    canvas = ROOT.TCanvas(
        "c_amplitude",
        "Pulse amplitude distribution",
        config["width"],
        config["height"],
    )
    canvas.SetMargin(0.13, 0.05, 0.13, 0.09)
    canvas.SetTicks(1, 1)

    histogram_first.SetLineColor(ROOT.TColor.GetColor("#1F4E79"))
    histogram_first.SetLineWidth(3)
    histogram_second.SetLineColor(ROOT.TColor.GetColor("#E63946"))
    histogram_second.SetLineWidth(3)

    histogram_first.GetXaxis().SetTitle("Pulse amplitude (ADC channel)")
    histogram_first.GetYaxis().SetTitle("Entries")
    histogram_first.SetTitle("Pulse amplitude distribution")
    histogram_first.Draw("hist")
    histogram_second.Draw("hist same")

    legend = ROOT.TLegend(0.66, 0.80, 0.91, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(1001)
    legend.SetFillColorAlpha(ROOT.kWhite, 0.85)
    legend.AddEntry(histogram_first, "Channel 0", "l")
    legend.AddEntry(histogram_second, "Channel 1", "l")
    legend.Draw()

    canvas.SetGridx()
    canvas.SetGridy()
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(path))
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

    values, amplitudes_first, amplitudes_second = sequential_time_differences(
        waveforms,
        delay,
        fraction,
        cfd["min_amplitude"],
        with_amplitudes=True,
    )
    threshold = float(cfd["min_amplitude"])
    amplitudes_first = amplitudes_first[amplitudes_first >= threshold]
    amplitudes_second = amplitudes_second[amplitudes_second >= threshold]

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
        plot_amplitudes(
            amplitudes_first,
            amplitudes_second,
            output / "amplitude_distribution.png",
            config["plot"],
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
        "amplitude": {
            "channel0": _amplitude_stats(amplitudes_first),
            "channel1": _amplitude_stats(amplitudes_second),
        },
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
