#!/usr/bin/env python3
"""
Script to average PSD data across multiple experiments/samples.

This script reads PSD data files (psd_ic_data.json and psd_forecast_data.json)
from multiple experiment outputs and computes the mean and standard deviation
of the power spectral density across all samples.

Usage:
    python average_psd_across_experiments.py --output-dir ../outputs/averaged_results
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import sys


def load_psd_ic_data(json_path: Path) -> Dict:
    """Load PSD initial condition data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_psd_forecast_data(json_path: Path) -> Tuple[Dict, Dict]:
    """Load PSD forecast data from JSON file, returns (metadata, data)."""
    with open(json_path, 'r') as f:
        content = json.load(f)
    
    if 'metadata' in content and 'data' in content:
        return content['metadata'], content['data']
    else:
        return {}, content


def average_psd_ic(psd_ic_files: List[Path], output_path: Path) -> Dict:
    """
    Average PSD initial condition data from multiple files.
    
    Args:
        psd_ic_files: List of paths to psd_ic_data.json files
        output_path: Path where averaged data will be saved
    
    Returns:
        Dictionary with averaged data
    """
    print(f"\n{'='*80}")
    print("AVERAGING PSD INITIAL CONDITION DATA")
    print(f"{'='*80}")
    print(f"Loading {len(psd_ic_files)} files...")
    
    all_data = []
    for fpath in psd_ic_files:
        if not fpath.exists():
            print(f"  ⚠ File not found: {fpath}")
            continue
        try:
            data = load_psd_ic_data(fpath)
            all_data.append(data)
            print(f"  ✓ Loaded: {fpath.parent.name}")
        except Exception as e:
            print(f"  ✗ Error loading {fpath}: {e}")
    
    if not all_data:
        print("  ERROR: No valid PSD IC files loaded!")
        return {}
    
    print(f"Successfully loaded {len(all_data)} files")
    
    # Average data
    averaged_data = {}
    
    for region_name in all_data[0].keys():
        averaged_data[region_name] = {}
        
        for channel_name in all_data[0][region_name].keys():
            averaged_data[region_name][channel_name] = {}
            
            for ic_type in ['reference_ic', 'optimized_ic']:
                k_values = []
                p_values = []
                
                for sample_data in all_data:
                    k = np.array(sample_data[region_name][channel_name][ic_type]['wavenumber'])
                    p = np.array(sample_data[region_name][channel_name][ic_type]['power'])
                    k_values.append(k)
                    p_values.append(p)
                
                # Use first sample's wavenumber grid as reference
                k_avg = k_values[0]
                p_mean = np.mean([p for p in p_values], axis=0)
                p_std = np.std([p for p in p_values], axis=0)
                p_min = np.min([p for p in p_values], axis=0)
                p_max = np.max([p for p in p_values], axis=0)
                
                averaged_data[region_name][channel_name][ic_type] = {
                    'wavenumber': k_avg.tolist(),
                    'power_mean': p_mean.tolist(),
                    'power_std': p_std.tolist(),
                    'power_min': p_min.tolist(),
                    'power_max': p_max.tolist(),
                    'n_samples': len(all_data)
                }
    
    # Save averaged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(averaged_data, f, indent=2)
    
    print(f"\n✓ Saved averaged IC PSD data to: {output_path}")
    print(f"  Regions: {list(averaged_data.keys())}")
    if averaged_data:
        first_region = list(averaged_data.keys())[0]
        print(f"  Channels: {list(averaged_data[first_region].keys())}")
    
    return averaged_data


def average_psd_forecast(psd_forecast_files: List[Path], output_path: Path) -> Dict:
    """
    Average PSD forecast data from multiple files.
    
    Args:
        psd_forecast_files: List of paths to psd_forecast_data.json files
        output_path: Path where averaged data will be saved
    
    Returns:
        Dictionary with averaged data
    """
    print(f"\n{'='*80}")
    print("AVERAGING PSD FORECAST DATA")
    print(f"{'='*80}")
    print(f"Loading {len(psd_forecast_files)} files...")
    
    all_data = []
    forecast_steps = []
    
    for fpath in psd_forecast_files:
        if not fpath.exists():
            print(f"  ⚠ File not found: {fpath}")
            continue
        try:
            metadata, data = load_psd_forecast_data(fpath)
            all_data.append(data)
            forecast_steps.append(metadata.get('forecast_step_index', -1))
            print(f"  ✓ Loaded: {fpath.parent.name} (step {metadata.get('forecast_step_number', '?')})")
        except Exception as e:
            print(f"  ✗ Error loading {fpath}: {e}")
    
    if not all_data:
        print("  ERROR: No valid PSD forecast files loaded!")
        return {}
    
    print(f"Successfully loaded {len(all_data)} files")
    
    # Check if all files have the same forecast step
    if len(set(forecast_steps)) > 1:
        print(f"  ⚠ Warning: Files contain different forecast steps: {set(forecast_steps)}")
    
    # Average data
    averaged_data = {}
    
    for region_name in all_data[0].keys():
        averaged_data[region_name] = {}
        
        for channel_name in all_data[0][region_name].keys():
            averaged_data[region_name][channel_name] = {}
            
            for fc_type in ['reference_forecast', 'control_forecast', 'optimized_forecast']:
                k_values = []
                p_values = []
                
                for sample_data in all_data:
                    k = np.array(sample_data[region_name][channel_name][fc_type]['wavenumber'])
                    p = np.array(sample_data[region_name][channel_name][fc_type]['power'])
                    k_values.append(k)
                    p_values.append(p)
                
                # Use first sample's wavenumber grid as reference
                k_avg = k_values[0]
                p_mean = np.mean([p for p in p_values], axis=0)
                p_std = np.std([p for p in p_values], axis=0)
                p_min = np.min([p for p in p_values], axis=0)
                p_max = np.max([p for p in p_values], axis=0)
                
                averaged_data[region_name][channel_name][fc_type] = {
                    'wavenumber': k_avg.tolist(),
                    'power_mean': p_mean.tolist(),
                    'power_std': p_std.tolist(),
                    'power_min': p_min.tolist(),
                    'power_max': p_max.tolist(),
                    'n_samples': len(all_data)
                }
    
    # Save averaged data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(averaged_data, f, indent=2)
    
    print(f"\n✓ Saved averaged forecast PSD data to: {output_path}")
    print(f"  Regions: {list(averaged_data.keys())}")
    if averaged_data:
        first_region = list(averaged_data.keys())[0]
        print(f"  Channels: {list(averaged_data[first_region].keys())}")
    
    return averaged_data


def find_psd_files(search_dir: Path, data_type: str = 'ic') -> List[Path]:
    """
    Recursively find all PSD data files in a directory tree.
    
    Args:
        search_dir: Directory to search in
        data_type: 'ic' for initial condition or 'forecast' for forecast
    
    Returns:
        List of Path objects for PSD data files
    """
    filename = f'psd_{data_type}_data.json'
    files = list(search_dir.rglob(filename))
    return sorted(files)

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Average PSD data across multiple experiments"
    )
    parser.add_argument(
        '--search-dir',
        type=Path,
        default=Path('../outputs/forecast_exp'),
        help='Directory to search for PSD data files (default: ../outputs/forecast_exp)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../outputs/averaged_psd'),
        help='Output directory for averaged PSD data (default: ../outputs/averaged_psd)'
    )
    parser.add_argument(
        '--ic-only',
        action='store_true',
        help='Average only initial condition PSD data'
    )
    parser.add_argument(
        '--forecast-only',
        action='store_true',
        help='Average only forecast PSD data'
    )
    return parser.parse_args()

def main():
    args = arg_parse()
    
    print(f"\n{'='*80}")
    print("PSD AVERAGING ACROSS EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Search directory: {args.search_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    
    # Find PSD files
    if not args.forecast_only:
        print(f"\nSearching for PSD IC data files...")
        ic_files = find_psd_files(args.search_dir, 'ic')
        print(f"  Found {len(ic_files)} PSD IC files")
        
        if ic_files:
            ic_output = args.output_dir / 'averaged_psd_ic.json'
            average_psd_ic(ic_files, ic_output)
    
    if not args.ic_only:
        print(f"\nSearching for PSD forecast data files...")
        forecast_files = find_psd_files(args.search_dir, 'forecast')
        print(f"  Found {len(forecast_files)} PSD forecast files")
        
        if forecast_files:
            forecast_output = args.output_dir / 'averaged_psd_forecast.json'
            average_psd_forecast(forecast_files, forecast_output)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
