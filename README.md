
## Commands

### Everything (all fits + all calibrations + walk)

```bash
~/miniforge3/envs/rootenv/bin/python main.py --all-fits --all
```

### TAC calibration (default, ns)

```bash
~/miniforge3/envs/rootenv/bin/python main.py
```

### TAC cm calibration

```bash
~/miniforge3/envs/rootenv/bin/python main.py --tag tac_cm
```

### Walk plot (FWHM vs delay)

```bash
~/miniforge3/envs/rootenv/bin/python main.py --calibration config/config_walk.json
```

### All TAC tags + walk

```bash
~/miniforge3/envs/rootenv/bin/python main.py --all
```

### Fitting only — all configs

```bash
~/miniforge3/envs/rootenv/bin/python main.py --all-fits --skip-calibration
```

### Fitting only — one config

```bash
~/miniforge3/envs/rootenv/bin/python main.py --skip-calibration -c config/config_lab_energy_D1.json
```

### Several fitting configs

```bash
~/miniforge3/envs/rootenv/bin/python main.py --skip-calibration -c config/config.json -c config/config_lab_energy_D2.json
```

### Direct script usage

```bash
~/miniforge3/envs/rootenv/bin/python read.py --config config/config.json
~/miniforge3/envs/rootenv/bin/python calibration.py -c config/config_calibration_tac.json
~/miniforge3/envs/rootenv/bin/python calibration.py -c config/config_walk.json
```

## Notes

- Configs: `code/config/`
- Outputs: `code/output/` (always; `-o` takes a sub-folder name there)
- `main.py` options: `--all`, `--all-fits`, `--tag`, `--calibration`, `-c/--config`, `--skip-fitting`, `--skip-calibration`
