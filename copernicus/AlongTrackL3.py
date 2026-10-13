"""Download Copernicus Marine Along Track - SSH reprocessed L3 Observations data"""
from datetime import datetime, date, timedelta
import copernicusmarine
import xarray as xr
import numpy as np
import xesmf as xe
import os
import sys
import gc
import argparse
import math

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
DEFAULT_OUTPUT_LOCATION = "/Odyssey/public/glonet"
# SLURM environment variable
SLURM_CHUNK_INDEX = "SLURM_ARRAY_TASK_ID"

def get_num_slurm_tasks():
    """Get number of SLURM array tasks. Returns None if not in SLURM environment."""
    if "SLURM_ARRAY_TASK_MAX" in os.environ:
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX"))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        return task_max - task_min + 1
    return None

def calculate_chunk_days(start_date: str, end_date: str, num_tasks: int = None) -> int:
    """Calculate chunk_days based on date range and number of SLURM tasks."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1
    
    # Use SLURM task count if available, otherwise calculate from 1 task
    if num_tasks is None:
        num_tasks = get_num_slurm_tasks()
    if num_tasks is None:
        num_tasks = 1
    
    chunk_days = math.ceil(total_days / num_tasks)
    print(f"Date range: {total_days} days, SLURM tasks: {num_tasks}, Chunk size: {chunk_days} days")
    return chunk_days

def iter_date_chunks(start_date: str, end_date: str, chunk_days: int):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer")
    if end_dt < start_dt:
        raise ValueError("end_date must be greater than or equal to start_date")

    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def get_all_chunks(start_date: str, end_date: str, chunk_days: int):
    """Return all chunks as a list instead of generator (useful for SLURM array jobs)."""
    return list(iter_date_chunks(start_date, end_date, chunk_days))

def get_data(start_date: str, 
              end_date: str) :
    
    id_list = [
        "cmems_obs-sl_glo_phy-ssh_my_c2-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_h2b-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_j3-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_alg-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S",
        "cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S"
    ]

    var_list = ["mdt", "sla_unfiltered"]

    ds_list = []
    for id in id_list:
        try:
            df = copernicusmarine.read_dataframe(
                dataset_id=id,
                variables=var_list,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_latitude=-80,
                maximum_latitude=90,
                start_datetime=start_date,
                end_datetime=end_date,
            )
            # Convert dataframe to xarray Dataset
            ds = df.to_xarray()
            ds_list.append(ds)
        except copernicusmarine.core_functions.exceptions.CoordinatesOutOfDatasetBounds:
            print(f"WARNING: Dataset {id} has no data for {start_date} to {end_date}, skipping...")
            continue
    
    # Combine all datasets
    ds_combined = xr.merge(ds_list, compat="override")
    
    ds_target = xr.Dataset(
        {
            "lat": (
                ["lat"],
                np.arange(ds_combined.latitude.min(), ds_combined.latitude.max(), 1 / 4),
            ),
            "lon": (
                ["lon"],
                np.arange(ds_combined.longitude.min(), ds_combined.longitude.max(), 1 / 4),
            ),
        }
    )
    
    regridder = xe.Regridder(ds_combined, ds_target, 'bilinear', periodic=True, reuse_weights=False)
    ds_regridded = regridder(ds_combined)
    del ds_combined
    
    return ds_regridded

def process_single_chunk(chunk_start: str,
                         chunk_end: str,
                         output_path: str) -> str:
    """Process a single chunk of data."""
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Processing chunk {chunk_start} to {chunk_end}")

    dataset = get_data(chunk_start, chunk_end)

    chunk_file = f"{output_path}/combined_obs_{chunk_start}_to_{chunk_end}.nc"
    dataset.to_netcdf(chunk_file)
    print(f"  Saved: {chunk_file}")

    dataset.close()
    del dataset
    gc.collect()
    
    return chunk_file


def is_slurm_environment():
    """Check if running under SLURM."""
    return SLURM_CHUNK_INDEX in os.environ


def process_slurm_chunk(start_date: str,
                        end_date: str,
                        output_path: str,
                        chunk_days: int = 30):
    """Process a single chunk assigned to this SLURM task."""
    chunk_index = int(os.environ.get(SLURM_CHUNK_INDEX, "0"))
    all_chunks = get_all_chunks(start_date, end_date, chunk_days)
    
    if chunk_index < 0 or chunk_index >= len(all_chunks):
        print(f"ERROR: Chunk index {chunk_index} out of range. Total chunks: {len(all_chunks)}")
        sys.exit(1)
    
    chunk_start, chunk_end = all_chunks[chunk_index]
    print(f"[Task {chunk_index + 1}/{len(all_chunks)}] {chunk_start} to {chunk_end}")
    
    return process_single_chunk(chunk_start, chunk_end, output_path)


def initiate_downloading(start_date: str,
                            end_date: str,
                            output_path: str = None,
                            chunk_days: int = 30,
                            combine_chunks: bool = False,
                            slurm_mode: bool = False) -> list:
    """Create initial states data from Copernicus Marine."""
    
    os.makedirs(output_path, exist_ok=True)
    
    # SLURM mode: process only the assigned chunk
    if slurm_mode or is_slurm_environment():
        chunk_file = process_slurm_chunk(start_date, end_date, output_path, chunk_days)
        return [chunk_file]
    
    # Sequential mode: process all chunks locally
    chunk_files = []
    for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_days):
        chunk_file = process_single_chunk(chunk_start, chunk_end, output_path)
        chunk_files.append(chunk_file)

    if combine_chunks:
        print(f"\nCombining {len(chunk_files)} chunks...")
        datasets_to_combine = [xr.open_dataset(path) for path in chunk_files]
        merged_all = xr.concat(datasets_to_combine, dim="time")
        output_file = f"{output_path}/combined_obs.nc"
        merged_all.to_netcdf(output_file)
        for ds in datasets_to_combine:
            ds.close()
        merged_all.close()
        gc.collect()
        print(f"Combined file: {output_file}")

    print(f"\nCompleted: {len(chunk_files)} chunks in {output_path}")
    return chunk_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine data Along Track satellite L3 observations."
    )
    parser.add_argument("--start_date", "-s",
                        required=True, 
                        help="Start date (YYYY-MM-DD)")
    
    parser.add_argument("--end_date", "-e", 
                        required=True, 
                        help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--out_path", "-o",
                        help="Output directory")

    parser.add_argument("--combine_chunks",
                        action="store_true",
                        help="Combine chunks into single file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.out_path:
        output_path = args.out_path
    else:
        output_path = DEFAULT_OUTPUT_LOCATION + f"/Alongtrack_SSH_{args.start_date}_to_{args.end_date}"
    
    # Calculate chunk_days based on date range and SLURM array size
    chunk_days = calculate_chunk_days(args.start_date, args.end_date)
    
    # Auto-detect SLURM and run accordingly
    initiate_downloading(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
        chunk_days=chunk_days,
        combine_chunks=args.combine_chunks,
        slurm_mode=is_slurm_environment()  # Auto-detect
    )


if __name__ == "__main__":
    main()