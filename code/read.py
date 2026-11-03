from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path
from typing import Any

try:
    import ROOT
except ImportError as exc:
    raise SystemExit(
        "PyROOT is not available for this Python interpreter.\n"
        "Run the script with a ROOT-enabled Python, for example:\n"
        "  ~/miniforge3/envs/rootenv/bin/python read.py --config config.json"
    ) from exc

import numpy as np

from fitting import PALETTE, FitError, fit_histogram
from plot import plot_histogram


class ConfigError(ValueError):
    pass


class ReaderError(RuntimeError):
    pass


class OutputError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a JSON object.")

    input_config = config.get("input")
    if not isinstance(input_config, dict) or "path" not in input_config:
        raise ConfigError('Missing "input.path".')

    base = config_path.parent
    input_config["path"] = _resolve(input_config["path"], base)
    input_config.setdefault("pattern", "*.root")
    input_config.setdefault("tree", "Data_R")

    output_config = config.get("output", "output")
    if isinstance(output_config, dict):
        output_path = output_config.get("path", "output")
        save_root = output_config.get("save_root", True)
    else:
        output_path = output_config
        save_root = config.get("save_root", True)

    config["output"] = _resolve(output_path, base)
    config["save_root"] = save_root

    plot_config = config.setdefault("plot", {})
    plot_config.setdefault("enabled", True)
    plot_config.setdefault("width", 1600)
    plot_config.setdefault("height", 1000)
    plot_config.setdefault("log_y", False)

    histograms = config.get("histograms")
    if not isinstance(histograms, list) or not histograms:
        raise ConfigError('"histograms" must be a non-empty array.')

    for index, histogram in enumerate(histograms):
        where = f"histograms[{index}]"
        if not isinstance(histogram, dict) or "name" not in histogram:
            raise ConfigError(f"{where} must have a name.")

        histogram.setdefault("expression", histogram["name"])
        for key in ("x_title", "y_title"):
            value = histogram.setdefault(key, histogram["name"] if key == "x_title" else "Entries")
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{where}.{key} must be a non-empty string.")
            histogram[key] = value

        title = histogram.get("title")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ConfigError(f"{where}.title must be a non-empty string.")
        histogram["title"] = title

        histogram.setdefault("fits", [])
        if not isinstance(histogram["fits"], list):
            raise ConfigError(f"{where}.fits must be an array.")

        bins = histogram.get("bins")
        if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
            raise ConfigError(f"{where}.bins must be a positive integer.")

        low, high = _histogram_range(histogram.get("range"), f"{where}.range")
        histogram["range"] = [low, high] if low is not None else None

        if histogram.get("channels") is not None:
            histogram["channels"] = _channels(
                histogram["channels"], f"{where}.channels"
            )

        range_low = histogram.get("range_low")
        if range_low is not None:
            if isinstance(range_low, bool) or not isinstance(range_low, (int, float)):
                raise ConfigError(f"{where}.range_low must be a number.")
            histogram["range_low"] = float(range_low)

        cut = histogram.get("cut_upper_outliers")
        if cut is not None:
            if isinstance(cut, bool) or not isinstance(cut, (int, float)) or cut <= 0:
                raise ConfigError(
                    f"{where}.cut_upper_outliers must be a positive number."
                )
            histogram["cut_upper_outliers"] = float(cut)

        x_range = histogram.get("x_range")
        if _is_auto(x_range):
            histogram["x_range"] = None
        elif x_range is not None:
            histogram["x_range"] = list(_range(x_range, f"{where}.x_range"))

        for fit_index, fit in enumerate(histogram["fits"]):
            fit_where = f"{where}.fits[{fit_index}]"
            if not isinstance(fit, dict):
                raise ConfigError(f"{fit_where} must be an object.")
            for key in ("name", "function", "range"):
                if key not in fit:
                    raise ConfigError(f'{fit_where} is missing "{key}".')

            if _is_auto(fit["range"]):
                fit["range"] = "auto"
            else:
                fit_low, fit_high = _range(fit["range"], f"{fit_where}.range")
                if low is not None and (fit_low < low or fit_high > high):
                    raise ConfigError(
                        f"{fit_where}.range must lie inside the histogram range."
                    )
                fit["range"] = [fit_low, fit_high]

            if fit.get("channels") is not None:
                fit["channels"] = _channels(
                    fit["channels"], f"{fit_where}.channels"
                )

            fit.setdefault("parameters", [])
            if not isinstance(fit["parameters"], list):
                raise ConfigError(f"{fit_where}.parameters must be an array.")

            for parameter in fit["parameters"]:
                if not isinstance(parameter, dict) or "index" not in parameter:
                    raise ConfigError(
                        f"{fit_where}.parameters entries need an index."
                    )
                parameter.setdefault("name", f"p{parameter['index']}")
                if parameter.get("limits") is not None:
                    parameter["limits"] = list(
                        _range(parameter["limits"], f"{fit_where}.limits")
                    )

    return config, config_path


def _resolve(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Paths in the configuration must be strings.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _histogram_range(
    value: Any, where: str
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if _is_auto(value):
        return None, None
    return _range(value, where)


def _channels(value: Any, where: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where} must be a non-empty array of channels.")
    channels = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise ConfigError(f"{where} must contain integer channels.")
        channels.append(channel)
    return channels


def _range(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{where} must be [min, max].")
    low, high = value
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
        or low >= high
    ):
        raise ConfigError(f"{where} must satisfy min < max.")
    return float(low), float(high)


def find_input_files(config: dict[str, Any]) -> list[Path]:
    path = config["path"]
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ReaderError(f"Input path does not exist: {path}")

    files = sorted(p for p in path.glob(config["pattern"]) if p.is_file())
    if not files:
        raise ReaderError(
            f"No files matching {config['pattern']!r} found in {path}"
        )
    return files


def read_file(
    config: dict[str, Any],
    path: Path,
    expressions: dict[str, str],
) -> dict[int, dict[str, np.ndarray]]:
    try:
        arrays = ROOT.RDataFrame(config["tree"], str(path)).AsNumpy()
    except Exception as exc:
        raise ReaderError(f"Cannot read {path}: {exc}") from exc

    if "Channel" not in arrays:
        raise ReaderError(f"{path} has no 'Channel' branch.")

    channels = np.asarray(arrays["Channel"]).astype(np.int64)
    environment = {
        name: np.asarray(values).astype(np.float64)
        for name, values in arrays.items()
        if name != "Channel"
    }

    values: dict[str, np.ndarray] = {}
    for name, expression in expressions.items():
        try:
            result = eval(
                expression,
                {"__builtins__": {}, "np": np},
                environment,
            )
        except Exception as exc:
            raise ReaderError(
                f"Cannot evaluate {expression!r} for {name!r}: {exc}"
            ) from exc
        values[name] = np.asarray(result, dtype=np.float64)

    grouped: dict[int, dict[str, np.ndarray]] = {}
    for channel in np.unique(channels):
        mask = channels == channel
        grouped[int(channel)] = {
            name: array[mask] for name, array in values.items()
        }
    return grouped


def build_histogram(
    config: dict[str, Any],
    values: np.ndarray,
    name: str,
    title: str,
) -> ROOT.TH1D:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    cut = config.get("cut_upper_outliers")
    if cut is not None and values.size:
        q1, q3 = np.percentile(values, [25, 75])
        values = values[values <= q3 + cut * (q3 - q1)]

    low, high = _histogram_bounds(config, values)
    histogram = ROOT.TH1D(name, title, config["bins"], low, high)
    histogram.SetDirectory(0)

    if values.size:
        weights = np.ones(values.size, dtype=np.float64)
        histogram.FillN(values.size, values, weights)
    return histogram


def _histogram_bounds(
    config: dict[str, Any],
    values: np.ndarray,
) -> tuple[float, float]:
    configured = config.get("range")
    if configured is not None:
        return float(configured[0]), float(configured[1])

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    low = config.get("range_low")
    if low is None:
        low = float(finite.min()) if finite.size else None
    high = float(finite.max()) if finite.size else None

    for fit in config["fits"]:
        if fit["range"] == "auto":
            continue
        fit_low, fit_high = fit["range"]
        low = fit_low if low is None else min(low, fit_low)
        high = fit_high if high is None else max(high, fit_high)

    if low is None or high is None:
        raise ReaderError(
            f"Histogram {config['name']!r}: range is 'auto' but there are "
            "no finite values to determine it."
        )
    if low == high:
        low -= 0.5
        high += 0.5
    if high > 0:
        high *= 1.5
    else:
        high += 0.5 * (high - low)
    return low, high


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "object"


def prepare_output(directory: Path) -> None:
    try:
        (directory / "plots").mkdir(parents=True, exist_ok=True)
        (directory / "root").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"Cannot create {directory}: {exc}") from exc


def write_results(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "results.json"
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


def write_table(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "results.csv"
    fieldnames = [
        "input_file",
        "channel",
        "histogram",
        "expression",
        "fit_name",
        "function",
        "range_min",
        "range_max",
        "valid",
        "status",
        "chi2",
        "ndf",
        "chi2_ndf",
        "p_value",
        "parameter_index",
        "parameter_name",
        "parameter_value",
        "parameter_error",
    ]
    try:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_flatten(payload))
    except OSError as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


def _flatten(payload: dict[str, Any]):
    for file_entry in payload["files"]:
        for channel_entry in file_entry["channels"]:
            for histogram in channel_entry["histograms"]:
                for fit in histogram["fits"]:
                    row = {
                        "input_file": file_entry["input_file"],
                        "channel": channel_entry["channel"],
                        "histogram": histogram["name"],
                        "expression": histogram["expression"],
                        "fit_name": fit["name"],
                        "function": fit["function"],
                        "range_min": fit["range"][0],
                        "range_max": fit["range"][1],
                        "valid": fit["valid"],
                        "status": fit["status"],
                        "chi2": fit["chi2"],
                        "ndf": fit["ndf"],
                        "chi2_ndf": fit["chi2_ndf"],
                        "p_value": fit["p_value"],
                    }
                    parameters = fit["parameters"] or [None]
                    for parameter in parameters:
                        output = dict(row)
                        if parameter is None:
                            output.update(
                                parameter_index=None,
                                parameter_name=None,
                                parameter_value=None,
                                parameter_error=None,
                            )
                        else:
                            output.update(
                                parameter_index=parameter["index"],
                                parameter_name=parameter["name"],
                                parameter_value=parameter["value"],
                                parameter_error=parameter["error"],
                            )
                        yield output


def write_root(
    directory: Path,
    file_name: str,
    items: list[tuple[ROOT.TH1D, list]],
) -> None:
    path = directory / "root" / file_name
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise OutputError(f"Cannot create ROOT file: {path}")

    try:
        for histogram, fits in items:
            root_file.cd()
            histogram.Write()
            for fit in fits:
                root_file.cd()
                fit.fit.Write()
                for parameter in fit.parameters:
                    name = sanitize(
                        f"{histogram.GetName()}_{fit.config['name']}_"
                        f"{parameter.name}"
                    )
                    ROOT.TParameter(
                        "double"
                    )(name, parameter.value).Write()
    finally:
        root_file.Close()


def input_label(config: dict[str, Any]) -> str:
    path = config["input"]["path"]
    return path.stem if path.is_file() else path.name


def output_directory(
    config: dict[str, Any],
    override: Path | None,
) -> Path:
    root = override if override is not None else config["output"]
    return root / input_label(config)


def _figure_title(
    config: dict[str, Any],
    channel: int,
    path: Path,
    stem: str,
) -> str:
    template = config.get("title")
    if template is None:
        return f"Channel {channel} ({stem})"
    try:
        return template.format(
            channel=channel,
            file=stem,
            path=path,
            histogram=config["name"],
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigError(
            f"Histogram {config['name']!r}: invalid title template {template!r}: "
            f"{exc}"
        ) from exc


def analyse_file(
    config: dict[str, Any],
    path: Path,
    channels: dict[int, dict[str, np.ndarray]],
    output: Path,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"input_file": str(path), "channels": []}

    for channel in sorted(channels):
        values = channels[channel]
        histogram_entries: list[dict[str, Any]] = []
        root_items: list[tuple[ROOT.TH1D, list]] = []

        for histogram_config in config["histograms"]:
            selected = histogram_config.get("channels")
            if selected is not None and channel not in selected:
                continue

            title = _figure_title(histogram_config, channel, path, path.stem)
            histogram = build_histogram(
                histogram_config,
                values[histogram_config["name"]],
                sanitize(f"h_{channel}_{path.stem}_{histogram_config['name']}"),
                f"{title};{histogram_config['x_title']};{histogram_config['y_title']}",
            )

            fits = []
            for fit_index, fit_config in enumerate(histogram_config["fits"]):
                fit_channels = fit_config.get("channels")
                if fit_channels is not None and channel not in fit_channels:
                    continue
                if fit_config["range"] == "auto":
                    fit_config = dict(
                        fit_config,
                        range=[
                            histogram.GetXaxis().GetXmin(),
                            histogram.GetXaxis().GetXmax(),
                        ],
                    )
                color = fit_config.get("color")
                if color is None:
                    color = PALETTE[fit_index % len(PALETTE)]
                fit_result = fit_histogram(
                    histogram,
                    fit_config,
                    sanitize(
                        f"f_{channel}_{path.stem}_"
                        f"{histogram_config['name']}_{fit_config['name']}"
                    ),
                    color,
                )
                fits.append(fit_result)
                if not fit_result.valid:
                    print(
                        f"  WARNING: fit {fit_config['name']!r} failed "
                        f"on {histogram.GetName()!r}"
                    )
                print(fit_result.summary(channel))

            x_range = histogram_config.get("x_range")
            if x_range is not None:
                histogram.GetXaxis().SetRangeUser(*x_range)

            valid_fits = [fit for fit in fits if fit.valid]

            if config["plot"]["enabled"]:
                plot_dir = output / "plots"
                base = (
                    f"Channel_{channel}_{path.stem}_"
                    f"{histogram_config['name']}"
                )
                for old in plot_dir.glob(f"{base}_*.png"):
                    old.unlink()
                (plot_dir / f"{base}.png").unlink(missing_ok=True)

                if valid_fits:
                    chi2_values = "_".join(
                        f"{fit.chi2_ndf:.2f}"
                        if fit.chi2_ndf is not None
                        else "undefined"
                        for fit in valid_fits
                    )
                    plot_histogram(
                        histogram,
                        valid_fits,
                        plot_dir / f"{base}_chi2ndf_{chi2_values}.png",
                        config["plot"],
                    )
                else:
                    plot_histogram(
                        histogram,
                        [],
                        plot_dir / f"{base}.png",
                        config["plot"],
                    )

            histogram_entries.append(
                {
                    "name": histogram_config["name"],
                    "expression": histogram_config["expression"],
                    "bins": histogram_config["bins"],
                    "range": [
                        histogram.GetXaxis().GetXmin(),
                        histogram.GetXaxis().GetXmax(),
                    ],
                    "entries": int(histogram.GetEntries()),
                    "fits": [fit.to_dict() for fit in valid_fits],
                }
            )
            root_items.append((histogram, fits))

        if config["save_root"]:
            write_root(
                output,
                f"Channel_{channel}_{path.stem}.root",
                root_items,
            )

        entry["channels"].append(
            {"channel": channel, "histograms": histogram_entries}
        )

    return entry


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit configured peaks in CoMPASS ROOT files."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Override the output root directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ROOT.gROOT.SetBatch(True)

    try:
        config, config_path = load_config(args.config)
        files = find_input_files(config["input"])
        override = args.output.expanduser().resolve() if args.output else None
        output = output_directory(config, override)
        prepare_output(output)

        expressions = {
            histogram["name"]: histogram["expression"]
            for histogram in config["histograms"]
        }
        payload: dict[str, Any] = {
            "generated": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "config_file": str(config_path),
            "input": {
                "path": str(config["input"]["path"]),
                "pattern": config["input"]["pattern"],
                "tree": config["input"]["tree"],
            },
            "output": str(output),
            "files": [],
        }

        for index, path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {path}")
            channels = read_file(config["input"], path, expressions)
            payload["files"].append(analyse_file(config, path, channels, output))

        json_path = write_results(output, payload)
        csv_path = write_table(output, payload)

        print()
        print(f"Results written to {json_path}")
        print(f"Table written to {csv_path}")
        print(f"Plots in {output / 'plots'}")
        print(f"ROOT files in {output / 'root'}")
    except (ConfigError, ReaderError, FitError, OutputError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
