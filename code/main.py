from __future__ import annotations

import argparse
import json
from pathlib import Path

import ROOT

import calibration
import read


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the histogram fits and the TAC/walk calibrations."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Extra fitting configuration to run (repeatable).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).parent / "config" / "config_calibration_tac.json",
        help="Calibration configuration.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Override the calibration tag (e.g. tac_ns, tac_cm, walk).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every tag in the calibration config plus the walk plot.",
    )
    parser.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Run only the calibration plots.",
    )
    parser.add_argument(
        "--all-fits",
        action="store_true",
        help="Run every fitting config found in the config folder.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Run only the histogram fits.",
    )
    return parser.parse_args(argv)


def all_fitting_configs() -> list[str]:
    config_dir = Path(__file__).parent / "config"
    paths = []
    for path in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "calibration" not in data:
            paths.append(str(path.resolve()))
    return paths


def calibration_configs(args: argparse.Namespace) -> list[dict]:
    if not args.all:
        config, _ = calibration.load_config(args.calibration, args.tag)
        return [config]

    base, _ = calibration.load_config(args.calibration)
    configs = []
    for tag in calibration.TAG_DEFAULTS:
        if isinstance(base["calibration"].get(tag), dict):
            config, _ = calibration.load_config(args.calibration, tag)
            configs.append(config)
    if not configs:
        configs.append(base)

    walk_path = (args.calibration.parent / "config_walk.json").resolve()
    if walk_path.is_file() and walk_path != args.calibration.resolve():
        walk_config, _ = calibration.load_config(walk_path)
        configs.append(walk_config)

    return configs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ROOT.gROOT.SetBatch(True)

    try:
        configs = calibration_configs(args)
    except (calibration.CalibrationError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1

    fitting = []
    seen = set()
    if not args.skip_fitting:
        candidates = []
        if args.all_fits:
            candidates.extend(all_fitting_configs())
        elif not args.skip_calibration:
            for config in configs:
                candidates.extend(
                    str(Path(path).resolve())
                    for path in calibration.fitting_configs(config)
                )
        candidates.extend(str(Path(extra).resolve()) for extra in args.config)
        if not candidates:
            candidates.append(
                str((Path(__file__).parent / "config" / "config.json").resolve())
            )

        for key in candidates:
            if key in seen:
                continue
            seen.add(key)
            fitting.append(key)

    status = 0
    for path in fitting:
        print("=" * 60)
        print(f"Histogram fitting: {path}")
        print("=" * 60)
        status |= read.main(["--config", path])
        print()

    if not args.skip_calibration:
        for config in configs:
            print("=" * 60)
            print(f"Calibration: {config['calibration']['name']}")
            print("=" * 60)
            try:
                calibration.run(config)
            except (
                calibration.CalibrationError,
                read.ConfigError,
                read.ReaderError,
                read.FitError,
                OSError,
            ) as exc:
                print(f"ERROR: {exc}")
                status = 1
            print()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
