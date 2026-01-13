#!/usr/bin/env python3
"""
Generate ensemble members using three perturbation methods:
- 40 members using one_vector perturbation
- 80 members using pointwise_1 perturbation  
- 80 members using pointwise_2 perturbation

Total: 200 ensemble members
"""

import os
import sys
import argparse
from pathlib import Path
import xarray as xr
import numpy as np
from typing import Union, List

# Import the perturbation classes (assuming they're in the same directory)
# If running as script, import from tmp.ipynb converted to .py or directly
from tmp import InitialConditionDataLoader, StochasticNoiseGenerator, CurrentBasedPerturbationGenerator


def generate_ensemble_members(
    data_path: Union[str, List[str]],
    target_idx: int,
    climato_path: str,
    rand_ds_path: str,
    output_dir: str,
    dt: float = 1.0,
    base_seed: int = 42
):
    """
    Generate ensemble members for all three perturbation methods.
    
    Parameters:
    -----------
    data_path : str or list of str
        Path(s) to the ocean state data
    target_idx : int
        Target time index
    climato_path : str
        Path to climatology data
    rand_ds_path : str
        Path to random dataset
    output_dir : str
        Directory to save ensemble members
    dt : float
        Time step for displacement
    base_seed : int
        Base seed for random number generation
    """
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onevector_dir = output_dir / "glonet_ensemble_one_vector"
    pointwise1_dir = output_dir / "glonet_ensemble_pointwise1"
    pointwise2_dir = output_dir / "glonet_ensemble_pointwise2"
    
    for dir_path in [onevector_dir, pointwise1_dir, pointwise2_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE GENERATION STARTING")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")
    print(f"Target index: {target_idx}")
    print(f"Base seed: {base_seed}")
    print()
    
    # ============================================================================
    # Method 1: One Vector Perturbation - 40 members
    # ============================================================================
    print("="*80)
    print("METHOD 1: ONE VECTOR PERTURBATION - 40 MEMBERS")
    print("="*80)
    print("Configuration:")
    print("  - 4 (u,v) combinations × 10 seeds = 40 members")
    print("  - Uses min/max ocean currents with regional Gaussian noise")
    print()
    
    member_count = 0
    for combo_idx in range(4):  # 4 combinations (min/max u,v)
        print(f"\nProcessing combination {combo_idx + 1}/4...")
        
        for seed_offset in range(10):  # 10 different seeds per combination
            seed = base_seed + combo_idx * 10 + seed_offset
            member_idx = member_count
            
            print(f"  Member {member_idx:03d}: seed={seed}", end="", flush=True)
            
            try:
                # Initialize generator
                generator = CurrentBasedPerturbationGenerator(
                    data_path=data_path,
                    target_idx=target_idx,
                    climato_path=climato_path,
                    rand_ds_path=rand_ds_path,
                    dt=dt,
                    seed=seed,
                    is_pointwise=False,
                    is_same_noise_in_time=True,
                    is_same_noise_for_component=True
                )
                
                # Generate perturbation (extraction_depth=0 for surface only)
                perturbed_state = generator.generate_one_vector_perturbation(
                    seed=seed,
                    extraction_depth=0
                )
                
                # Save to file
                output_path = onevector_dir / f"member_{member_idx:03d}" / "perturbed_initial_condition.nc"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to dataset and save
                ds_out = xr.Dataset({"data": perturbed_state})
                ds_out.to_netcdf(output_path)
                
                print(" ✓ Saved")
                member_count += 1
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                continue
    
    print(f"\nCompleted: {member_count} one_vector members generated")
    
    # ============================================================================
    # Method 2: Pointwise 1 Perturbation - 80 members
    # ============================================================================
    print("\n" + "="*80)
    print("METHOD 2: POINTWISE 1 PERTURBATION - 80 MEMBERS")
    print("="*80)
    print("Configuration:")
    print("  - Surface dynamics applied to all depths")
    print("  - Uses spectral correlated noise")
    print()
    
    member_count = 0
    for member_idx in range(80):
        seed = base_seed + 1000 + member_idx  # Offset to avoid collision
        
        print(f"Member {member_idx:03d}: seed={seed}", end="", flush=True)
        
        try:
            # Initialize generator
            generator = CurrentBasedPerturbationGenerator(
                data_path=data_path,
                target_idx=target_idx,
                climato_path=climato_path,
                rand_ds_path=rand_ds_path,
                dt=dt,
                seed=seed,
                is_pointwise=True,
                is_same_noise_in_time=False,
                is_same_noise_for_component=False
            )
            
            # Generate perturbation
            perturbed_state = generator.generate_pointwise_1_perturbation(seed=seed)
            
            # Save to file
            output_path = pointwise1_dir / f"member_{member_idx:03d}" / "perturbed_initial_condition.nc"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dataset and save
            ds_out = xr.Dataset({"data": perturbed_state})
            ds_out.to_netcdf(output_path)
            
            print(" ✓ Saved")
            member_count += 1
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
        
        # Progress indicator every 20 members
        if (member_idx + 1) % 20 == 0:
            print(f"  [{member_count}/80 completed]")
    
    print(f"\nCompleted: {member_count} pointwise_1 members generated")
    
    # ============================================================================
    # Method 3: Pointwise 2 Perturbation - 80 members
    # ============================================================================
    print("\n" + "="*80)
    print("METHOD 3: POINTWISE 2 PERTURBATION - 80 MEMBERS")
    print("="*80)
    print("Configuration:")
    print("  - Depth-specific dynamics (21 depth levels)")
    print("  - Uses spectral correlated noise at each depth")
    print()
    
    member_count = 0
    for member_idx in range(80):
        seed = base_seed + 2000 + member_idx  # Offset to avoid collision
        
        print(f"Member {member_idx:03d}: seed={seed}", end="", flush=True)
        
        try:
            # Initialize generator
            generator = CurrentBasedPerturbationGenerator(
                data_path=data_path,
                target_idx=target_idx,
                climato_path=climato_path,
                rand_ds_path=rand_ds_path,
                dt=dt,
                seed=seed,
                is_pointwise=True,
                is_same_noise_in_time=False,
                is_same_noise_for_component=False
            )
            
            # Generate perturbation
            perturbed_state = generator.generate_pointwise_2_perturbation(seed=seed)
            
            # Save to file
            output_path = pointwise2_dir / f"member_{member_idx:03d}" / "perturbed_initial_condition.nc"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dataset and save
            ds_out = xr.Dataset({"data": perturbed_state})
            ds_out.to_netcdf(output_path)
            
            print(" ✓ Saved")
            member_count += 1
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
        
        # Progress indicator every 20 members
        if (member_idx + 1) % 20 == 0:
            print(f"  [{member_count}/80 completed]")
    
    print(f"\nCompleted: {member_count} pointwise_2 members generated")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*80)
    print("ENSEMBLE GENERATION COMPLETE")
    print("="*80)
    print(f"Total ensemble members generated: 200")
    print(f"  - One vector:    40 members in {onevector_dir}")
    print(f"  - Pointwise 1:   80 members in {pointwise1_dir}")
    print(f"  - Pointwise 2:   80 members in {pointwise2_dir}")
    print()


def main():
    """Main function to parse arguments and run ensemble generation."""
    
    parser = argparse.ArgumentParser(
        description="Generate ocean ensemble members using three perturbation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_ensemble.py \\
    --data-path /path/to/ocean_data.nc \\
    --target-idx 10 \\
    --climato-path /path/to/climatology.nc \\
    --rand-ds-path /path/to/random_dataset.nc \\
    --output-dir ./ensemble_output

  python generate_ensemble.py \\
    --data-path /path/to/data1.nc /path/to/data2.nc \\
    --target-idx 5 \\
    --climato-path /path/to/climato.nc \\
    --rand-ds-path /path/to/rand.nc \\
    --output-dir ./output \\
    --dt 1.5 \\
    --base-seed 123
        """
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to ocean state NetCDF file(s)"
    )
    
    parser.add_argument(
        "--target-idx",
        type=int,
        required=True,
        help="Target time index for perturbation"
    )
    
    parser.add_argument(
        "--climato-path",
        type=str,
        required=True,
        help="Path to climatology NetCDF file"
    )
    
    parser.add_argument(
        "--rand-ds-path",
        type=str,
        required=True,
        help="Path to random dataset NetCDF file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ensemble_output",
        help="Output directory for ensemble members (default: ./ensemble_output)"
    )
    
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step for displacement (default: 1.0)"
    )
    
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for random number generation (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Convert data_path to single string if only one path provided
    data_path = args.data_path[0] if len(args.data_path) == 1 else args.data_path
    
    try:
        generate_ensemble_members(
            data_path=data_path,
            target_idx=args.target_idx,
            climato_path=args.climato_path,
            rand_ds_path=args.rand_ds_path,
            output_dir=args.output_dir,
            dt=args.dt,
            base_seed=args.base_seed
        )
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
