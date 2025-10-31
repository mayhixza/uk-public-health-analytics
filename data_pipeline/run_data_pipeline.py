"""
Data Pipeline Runner

Simply runs:
1. preprocess.py - Processes all raw datasets
2. merge_dataset.py - Merges processed datasets into master dataset

"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_directory_structure():
    base_dir = project_root / 'datasets'
    directories = [
        base_dir / 'raw',
        base_dir / 'processed', 
        base_dir / 'master'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")
    
    print()


def run_preprocessing():
    """Run preprocess.py"""
    print("=" * 80)
    print("STEP 1: RUNNING PREPROCESSING (preprocess.py)")
    print("=" * 80)
    print()
    
    preprocess_script = Path(__file__).parent / 'preprocess.py'
    
    if not preprocess_script.exists():
        print(f"✗ Error: preprocess.py not found at {preprocess_script}")
        return False
    
    try:
        # Run preprocess.py
        result = subprocess.run(
            [sys.executable, str(preprocess_script)],
            cwd=str(project_root),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ Preprocessing completed successfully!")
            return True
        else:
            print(f"\n✗ Preprocessing failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running preprocess.py: {e}")
        return False


def run_merging():
    """Run merge_dataset.py"""
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING DATASET MERGING (merge_dataset.py)")
    print("=" * 80)
    print()
    
    merge_script = Path(__file__).parent / 'merge_dataset.py'
    
    if not merge_script.exists():
        print(f"✗ Error: merge_dataset.py not found at {merge_script}")
        return False
    
    try:
        # Run merge_dataset.py
        result = subprocess.run(
            [sys.executable, str(merge_script)],
            cwd=str(project_root),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ Dataset merging completed successfully!")
            return True
        else:
            print(f"\n✗ Dataset merging failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running merge_dataset.py: {e}")
        return False


def main():
    print("Running data pipeline...")
    print()

    create_directory_structure()
    preprocessing_success = run_preprocessing()
    
    if not preprocessing_success:
        print("\nPipeline aborted: Preprocessing failed")
        return False
    
    merging_success = run_merging()
    
    if not merging_success:
        print("\nPipeline aborted: Merging failed")
        return False
    
    print("\n" + "=" * 80)
    print("Data pipeline completed")
    print("=" * 80)
    
    # outputs
    processed_dir = project_root / 'datasets' / 'processed'
    master_dir = project_root / 'datasets' / 'master'
    
    print("\n📁 Output locations:")
    print(f"  • Processed datasets: {processed_dir}")
    print(f"  • Master dataset: {master_dir}")
    
    master_file = master_dir / 'uk_health_master_dataset.csv'
    if master_file.exists():
        import pandas as pd
        master = pd.read_csv(master_file)
        print(f"\n📊 Master Dataset Summary:")
        print(f"  • Shape: {master.shape}")
        print(f"  • Features: {len(master.columns) - 1}")
        print(f"  • Local Authorities: {len(master)}")
    
    print("\n" + "=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)