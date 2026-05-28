"""
Fetch and process EPA AQS PM2.5 hourly data for California.

Downloads hourly PM2.5 readings from all monitoring stations in California
via the EPA AQS API, then reshapes into a tensor (I, J, K) where:
  I = number of sensors (stations with sufficient data)
  J = number of time slots per day (24 for hourly data)
  K = number of days

Usage:
  python fetch_epa_air_quality.py [--year 2023] [--state 06] [--output-dir ../../data/EPA_CA]

EPA AQS API docs: https://aqs.epa.gov/aqsweb/documents/data_api.html
Demo credentials: email=test@aqs.api, key=test
"""

import urllib.request
import json
import time
import argparse
import numpy as np
import scipy.io
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


EMAIL = "test@aqs.api"
KEY = "test"
PARAM = "88101"  # PM2.5


def api_get(url, retries=8, delay=5.0):
    """GET request with retries and 429 handling."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = delay * (2 ** attempt)  # exponential backoff: 5, 10, 20, 40, 80s...
                print(f"    Rate limited (429), waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


def get_monitors(state, year):
    """Get list of PM2.5 monitoring sites for a state."""
    bdate = f"{year}0101"
    edate = f"{year}1231"
    url = (
        f"https://aqs.epa.gov/data/api/monitors/byState?"
        f"email={EMAIL}&key={KEY}&param={PARAM}"
        f"&state={state}&bdate={bdate}&edate={edate}"
    )
    data = api_get(url)
    sites = {}
    for m in data.get("Data", []):
        sid = f"{m['state_code']}-{m['county_code']}-{m['site_number']}"
        if sid not in sites:
            lat = float(m.get("latitude", 0))
            lon = float(m.get("longitude", 0))
            if lat != 0 and lon != 0:
                sites[sid] = {
                    "lat": lat, "lon": lon,
                    "county_code": m["county_code"],
                    "site_number": m["site_number"],
                    "county": m.get("county_name", ""),
                }
    return sites


def fetch_hourly_data(state, county_code, site_number, bdate, edate):
    """Fetch hourly PM2.5 data for a single site."""
    url = (
        f"https://aqs.epa.gov/data/api/sampleData/bySite?"
        f"email={EMAIL}&key={KEY}&param={PARAM}"
        f"&state={state}&county={county_code}"
        f"&site={site_number}&bdate={bdate}&edate={edate}"
    )
    data = api_get(url)
    records = []
    for rec in data.get("Data", []):
        dt_str = rec.get("date_local", "") + " " + rec.get("time_local", "")
        val = rec.get("sample_measurement")
        if val is not None:
            try:
                dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
                records.append((dt, float(val)))
            except ValueError:
                pass
    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch EPA AQS PM2.5 data")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--state", default="06", help="State FIPS code (06=CA)")
    parser.add_argument("--output-dir", default="../../data/EPA_CA")
    parser.add_argument("--min-hours", type=int, default=500,
                        help="Minimum hourly readings per site to keep")
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8],
                        help="Months to download (default: Jun-Aug)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get site list
    print(f"Fetching PM2.5 monitor list for state {args.state} ({args.year})...")
    sites = get_monitors(args.state, args.year)
    print(f"  Found {len(sites)} sites with coordinates")

    # Step 2: Download hourly data
    print(f"Downloading hourly data for months {args.months}...")
    all_data = {}
    for i, (sid, info) in enumerate(sites.items()):
        site_records = []
        for month in args.months:
            bd = f"{args.year}{month:02d}01"
            # Compute end date
            if month == 12:
                ed = f"{args.year}1231"
            else:
                next_month = month + 1
                ed_dt = datetime(args.year, next_month, 1) - timedelta(days=1)
                ed = ed_dt.strftime("%Y%m%d")
            try:
                records = fetch_hourly_data(
                    args.state, info["county_code"], info["site_number"], bd, ed
                )
                site_records.extend(records)
            except Exception as e:
                print(f"  Warning: {sid} month={month} failed: {e}")
            time.sleep(1.0)  # Rate limit

        all_data[sid] = site_records
        if (i + 1) % 20 == 0 or i == len(sites) - 1:
            total = sum(len(v) for v in all_data.values())
            print(f"  Progress: {i+1}/{len(sites)} sites, {total} hourly records")

    # Step 3: Filter sites with sufficient data
    valid = {sid: recs for sid, recs in all_data.items() if len(recs) >= args.min_hours}
    print(f"\nSites with >= {args.min_hours} hours: {len(valid)}")

    if len(valid) < 10:
        print("WARNING: Too few valid sites. Consider reducing --min-hours or expanding --months.")

    # Step 4: Build tensor
    # Determine time range
    all_dts = set()
    for recs in valid.values():
        for dt, _ in recs:
            all_dts.add(dt)

    if not all_dts:
        print("ERROR: No data retrieved. Check network and API availability.")
        return

    min_dt = min(all_dts)
    max_dt = max(all_dts)
    print(f"Time range: {min_dt} to {max_dt}")

    # Build continuous hourly index
    start = min_dt.replace(minute=0, second=0, microsecond=0)
    # Align to day boundary
    start = start.replace(hour=0)
    end = max_dt.replace(minute=0, second=0, microsecond=0)
    end = end.replace(hour=23) + timedelta(hours=1)

    hours = []
    t = start
    while t < end:
        hours.append(t)
        t += timedelta(hours=1)
    n_hours = len(hours)
    hour_idx = {h: i for i, h in enumerate(hours)}

    n_days = (end.date() - start.date()).days
    J = 24  # hours per day
    K = n_days
    I = len(valid)
    print(f"Tensor shape: ({I}, {J}, {K}) = {I} sensors x {J} hours/day x {K} days")

    # Fill tensor
    tensor = np.full((I, J, K), np.nan)
    site_list = sorted(valid.keys())
    site_idx = {s: i for i, s in enumerate(site_list)}

    for sid, recs in valid.items():
        si = site_idx[sid]
        for dt, val in recs:
            if dt in hour_idx:
                hi = hour_idx[dt]
                day = (dt.date() - start.date()).days
                hour = dt.hour
                if 0 <= day < K and 0 <= hour < J:
                    tensor[si, hour, day] = val

    # Report missing rate
    missing_rate = np.isnan(tensor).sum() / tensor.size
    print(f"Missing rate: {missing_rate:.1%}")
    print(f"Value range: [{np.nanmin(tensor):.1f}, {np.nanmax(tensor):.1f}]")
    print(f"Mean: {np.nanmean(tensor):.1f}, Std: {np.nanstd(tensor):.1f}")

    # Step 5: Save
    coords = np.array([[valid[sid]["lat"], valid[sid]["lon"]] for sid in site_list]
                       if False else
                       [[sites[sid]["lat"], sites[sid]["lon"]] for sid in site_list])

    scipy.io.savemat(str(output_dir / "tensor.mat"),
                     {"tensor": tensor.astype(np.float64)})
    np.save(str(output_dir / "sensor_coords.npy"), coords)

    # Save metadata
    meta = {
        "source": "EPA AQS",
        "parameter": "PM2.5 (88101)",
        "state": args.state,
        "year": args.year,
        "months": args.months,
        "n_sites": I,
        "n_hours_per_day": J,
        "n_days": K,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": (end - timedelta(hours=1)).strftime("%Y-%m-%d"),
        "missing_rate": float(missing_rate),
        "site_list": site_list,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  tensor.mat: shape=({I}, {J}, {K})")
    print(f"  sensor_coords.npy: shape=({I}, 2)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
