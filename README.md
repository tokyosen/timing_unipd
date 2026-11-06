
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

### All TAC tags + walk + digital CFD

```bash
~/miniforge3/envs/rootenv/bin/python main.py --all
```

### Digital CFD (section 4.7, waveform data 47waves)

```bash
~/miniforge3/envs/rootenv/bin/python main.py --waves --skip-fitting --skip-calibration
```

### Time resolution vs energy range (section 4.5, Co-60)

```bash
~/miniforge3/envs/rootenv/bin/python main.py --energy-scan --skip-fitting --skip-calibration
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
~/miniforge3/envs/rootenv/bin/python digital_cfd.py -c config/config_waves.json
```

## Notes

- Configs: `code/config/`
- Outputs: `code/output/` (always; `-o` takes a sub-folder name there)
- `main.py` options: `--all`, `--all-fits`, `--waves`, `--waves-config`, `--energy-scan`, `--energy-scan-config`, `--tag`, `--calibration`, `-c/--config`, `--skip-fitting`, `--skip-calibration`
