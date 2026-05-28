# Data

This directory contains the three spatio-temporal datasets used in the paper.

## Datasets

| Dataset | File | Shape | Description |
|---------|------|-------|-------------|
| Guangzhou Traffic | `tensor.mat` (root) | 214×61×144 | Urban traffic speed, 214 sensors, 61 days, 144 intervals/day |
| PeMSD7 Traffic | `PeMSD7/tensor.mat` | 228×96×44 | Highway traffic speed, 228 sensors, 44 days, 96 intervals/day |
| EPA CA Air Quality | `EPA_CA/tensor.mat` | 101×24×92 | PM2.5 concentration, 101 stations, 92 days, 24 hours/day |

## Download

The tensor `.mat` files are excluded from GitHub due to size. To obtain them:

### Guangzhou Traffic
The `tensor.mat` in the project root is the Guangzhou dataset. It is publicly available from the Guangzhou Open Data platform or from the HaLRTC paper's repository.

### PeMSD7 Traffic
Run the processing script:
```bash
cd data/external
python process_pemsd7.py
```
This downloads raw data from the [STGCN IJCAI-18 repository](https://github.com/VeritasYin/STGCN_IJCAI-18) and converts it to tensor format. The STGCN repo is included as a submodule in `data/external/STGCN_IJCAI-18/`.

### EPA CA Air Quality
Run the fetching script:
```bash
cd data/external
python fetch_epa_air_quality.py
```
This downloads hourly PM2.5 data from the EPA AQS API for California monitoring stations and reshapes it into a tensor. An EPA AQS API key may be required for large requests; see the script header for details.

## Precomputed Files

- `sensor_coords_mds.npy` — MDS-embedded 2D coordinates for Guangzhou sensors (computed by `src/kriging.py`)
- `variogram_params.npy.npz` — Fitted variogram parameters (computed by `src/kriging.py`)
- `PeMSD7/sensor_coords.npy` — GPS coordinates for PeMSD7 sensors
- `EPA_CA/sensor_coords.npy` — MDS coordinates for EPA CA stations
- `EPA_CA/metadata.json` — Station metadata for EPA CA

These are auto-generated on first run if not present.
