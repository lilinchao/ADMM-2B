# Data Preparation and Recovery Status

Processed tensors from the expired cloud server are **not included**. `reproduction_data/` contains saved predictions/statistics, not model inputs. Coordinate arrays tracked by the original repository are retained and must remain aligned with tensor sensor ordering.

| Dataset | Tensor | Coordinates | Archived shape |
|---|---|---|---|
| Guangzhou | `data/tensor.mat` | `data/sensor_coords_mds.npy` | `(214, 61, 144)` |
| PeMSD7 | `data/PeMSD7/tensor.mat` | `data/PeMSD7/sensor_coords.npy` | `(228, 96, 44)` |
| EPA CA | `data/EPA_CA/tensor.mat` | `data/EPA_CA/sensor_coords.npy` | `(101, 24, 92)` |

Each MAT file must contain a numeric array named `tensor`. Shapes are literal stored axis orders, not interchangeable day/time labels. PeMSD7's processing script produces `(sensor, slot, day)`; some historical comments use a different convention. Do not transpose files without revalidating the experiments.

## Guangzhou

Recover the **exact** processed tensor and matching sensor order from a backup or the authors. The old README did not identify a versioned downloadable source; this release does not invent one. A same-shaped file from another project does not prove identity. The saved value range is approximately 119.269, but shape/range alone do not establish equality. An original checksum was not recorded.

## PeMSD7

The existing processing script reads files from the [STGCN repository](https://github.com/VeritasYin/STGCN_IJCAI-18); it does not download them. A possible reconstruction route on the cloud is:

```bash
git clone https://github.com/VeritasYin/STGCN_IJCAI-18.git data/external/STGCN_IJCAI-18
python data/external/process_pemsd7.py \
  --source-dir data/external/STGCN_IJCAI-18/dataset \
  --output-dir data/PeMSD7 --agg-minutes 15
```

Required inputs: `PeMSD7_V_228.csv` and `PeMSD7_M_Station_Info.csv`. Record the upstream revision and input hashes. The historical revision was not retained, so this route is not yet verified to recreate identical inputs. Despite old README wording, this directory was not an actual Git submodule.

## EPA California

`data/external/fetch_epa_air_quality.py` and `data/EPA_CA/metadata.json` are historical acquisition/preprocessing artifacts. Live downloads can differ from the saved dataset. Preserve the station list, dates, filtering and filling choices in the metadata; review the script before use. Its public demo credentials are not production credentials. Never commit personal API credentials.

## Verification and Redistribution

```bash
python reproduction/reproduce.py preflight --data-dir /path/to/prepared/data
```

This checks presence, shapes, finite values and coordinate dimensions, and records new hashes. It cannot prove equality to missing original tensors. The wrapper rejects non-finite inputs because the legacy loader otherwise silently fills NaNs with zeros; resolve missing ground truth explicitly.

Third-party raw data are not redistributed without established provenance and permission. No private credentials or server address are included in this release.
