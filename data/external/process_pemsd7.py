"""
Process PeMSD7 traffic speed dataset into tensor format.

Source: STGCN IJCAI-18 repository
  https://github.com/VeritasYin/STGCN_IJCAI-18

Data: 228 traffic sensors on District 7 (Los Angeles area) freeways,
      5-minute interval speed data, 44 weekdays.

Output tensor shape: (228, 96, 44) = (sensors, time_slots_per_day, days)
  - Time slots are 15-minute aggregated (from original 5-min)
  - Coordinates are real (Latitude, Longitude) from station info

Usage:
  # First clone the source repo:
  git clone --depth 1 https://github.com/VeritasYin/STGCN_IJCAI-18.git
  # Then run:
  python process_pemsd7.py --source-dir STGCN_IJCAI-18/dataset --output-dir ../../data/PeMSD7
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description="Process PeMSD7 dataset")
    parser.add_argument("--source-dir", default="STGCN_IJCAI-18/dataset",
                        help="Path to STGCN repo dataset directory")
    parser.add_argument("--output-dir", default="../../data/PeMSD7")
    parser.add_argument("--agg-minutes", type=int, default=15,
                        choices=[5, 15, 60],
                        help="Time aggregation interval in minutes")
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load speed data: (12672 time_steps, 228 sensors)
    print("Loading PeMSD7_V_228.csv...")
    v = pd.read_csv(src / "PeMSD7_V_228.csv", header=None).values
    print(f"  Raw shape: {v.shape} (time_steps x sensors)")

    # Load station coordinates
    info = pd.read_csv(src / "PeMSD7_M_Station_Info.csv")
    lats = info["Latitude"].values
    lons = info["Longitude"].values
    coords = np.column_stack([lats, lons])
    print(f"  Coordinates: {coords.shape}, "
          f"lat=[{lats.min():.4f}, {lats.max():.4f}], "
          f"lon=[{lons.min():.4f}, {lons.max():.4f}]")

    # Reshape to (sensors, time_slots_per_day, days)
    I = 228
    J_raw = 288  # 24h * 12 (5-min intervals per day)
    K = v.shape[0] // J_raw  # = 44 days
    print(f"  Days: {K}, time_slots_per_day (5min): {J_raw}")

    # Aggregate time slots
    agg_factor = args.agg_minutes // 5  # 5min -> target
    J = J_raw // agg_factor
    print(f"  Aggregating from 5min to {args.agg_minutes}min: J={J_raw}->{J}")

    # Reshape: (time_steps, sensors) -> (days, J_raw, I) -> aggregate -> (I, J, K)
    v_3d = v.reshape(K, J_raw, I)
    v_agg = v_3d.reshape(K, J, agg_factor, I).mean(axis=2)  # (K, J, I)
    tensor = v_agg.transpose(2, 1, 0)  # (I, J, K)

    print(f"\nFinal tensor: {tensor.shape}")
    print(f"  Value range: [{tensor.min():.1f}, {tensor.max():.1f}]")
    print(f"  Mean: {tensor.mean():.1f}, Std: {tensor.std():.1f}")
    print(f"  Missing: {np.isnan(tensor).sum()}/{tensor.size}")
    print(f"  Memory: {tensor.nbytes / 1e6:.1f} MB")

    # Save
    scipy.io.savemat(str(out / "tensor.mat"),
                     {"tensor": tensor.astype(np.float64)})
    np.save(str(out / "sensor_coords.npy"), coords)

    # Save metadata
    meta = {
        "source": "PeMSD7 (Caltrans District 7, Los Angeles)",
        "url": "https://github.com/VeritasYin/STGCN_IJCAI-18",
        "parameter": "Traffic speed (mph)",
        "n_sensors": int(I),
        "n_time_slots_per_day": int(J),
        "agg_minutes": args.agg_minutes,
        "n_days": int(K),
        "value_range": [float(tensor.min()), float(tensor.max())],
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "missing_rate": float(np.isnan(tensor).sum() / tensor.size),
        "coord_format": "WGS84 (latitude, longitude)",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out}/")
    print(f"  tensor.mat: shape={tensor.shape}")
    print(f"  sensor_coords.npy: shape={coords.shape}")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
