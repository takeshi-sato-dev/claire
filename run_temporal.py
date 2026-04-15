#!/usr/bin/env python3
"""
CLAIRE v21 Stage 3: Temporal causal analysis

Determines causal direction between structural/compositional variables
using Granger causality, conditional Granger causality, and Transfer entropy.

Automatically detects available variables in the CSV and tests all
relevant pairs:
  - distance vs lipid counts (15A)
  - tilt vs lipid counts
  - local_scd vs lipid counts (both scd and scd_excl_fs if available)
  - first_shell counts vs lipid counts, S_CD, and other first_shell
  - cross first_shell pairs (e.g., fs_DIPC vs fs_DPSM)
  - distance vs first_shell, tilt, S_CD

Usage:
    python run_temporal.py composition_data.csv
    python run_temporal.py composition_data.csv --max-lag 20
    python run_temporal.py composition_data.csv --surrogates 200 --no-ccm
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.temporal_causal import run_temporal_causal, get_temporal_causal_table
from config import OUTPUT_DIR, DEFAULT_LIPID_TYPES, TARGET_LIPID


def main():
    parser = argparse.ArgumentParser(
        description='CLAIRE v21 Stage 3: Temporal causal analysis')
    parser.add_argument('csv', nargs='?',
                        default=os.path.join(OUTPUT_DIR, 'composition_data.csv'),
                        help='Path to composition_data.csv')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as CSV)')
    parser.add_argument('--lipid', nargs='+', default=None,
                        help='Lipid(s) to test (default: all composition lipids)')
    parser.add_argument('--max-lag', type=int, default=10,
                        help='Maximum lag in frames (default: 10)')
    parser.add_argument('--te-lags', nargs='+', type=int, default=None,
                        help='TE lags (default: 1 2 3 5 10)')
    parser.add_argument('--surrogates', type=int, default=100,
                        help='Number of surrogates (default: 100)')
    parser.add_argument('--ccm-E', type=int, default=4,
                        help='CCM embedding dimension (default: 4)')
    parser.add_argument('--no-ccm', action='store_true',
                        help='Skip CCM (faster)')
    parser.add_argument('--all-frames', action='store_true',
                        help='Use all frames (no bound filtering). '
                             'For GM3-free control systems.')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found")
        return

    df = pd.read_csv(args.csv)

    if not args.all_frames and 'target_lipid_min_distance' not in df.columns:
        print("ERROR: target_lipid_min_distance column not found.")
        print("  Re-run Stage 1 with run_extract.py to generate this column.")
        print("  Or use --all-frames for GM3-free systems.")
        return

    print(f"\n{'='*70}")
    print("CLAIRE v21 Stage 3: Temporal Causal Analysis")
    if args.all_frames:
        print("  MODE: ALL FRAMES (no bound filtering)")
    print(f"{'='*70}")
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Report available variables
    has_tilt = 'tm_tilt' in df.columns and df['tm_tilt'].notna().sum() > 100
    has_scd = 'local_scd' in df.columns and df['local_scd'].notna().sum() > 100
    has_scd_xfs = 'local_scd_excl_fs' in df.columns and df['local_scd_excl_fs'].notna().sum() > 100
    fs_cols = [c for c in df.columns if c.endswith('_first_shell') and df[c].notna().sum() > 100]

    print(f"\nVariables detected:")
    print(f"  distance:       yes")
    print(f"  tm_tilt:        {'yes' if has_tilt else 'no'}")
    print(f"  local_scd:      {'yes' if has_scd else 'no'}")
    print(f"  local_scd_xfs:  {'yes' if has_scd_xfs else 'no'}")
    print(f"  first_shell:    {fs_cols if fs_cols else 'none'}")

    # Per copy summary
    print(f"\nPer copy:")
    for prot in sorted(df['protein'].unique()):
        sub = df[df['protein'] == prot]
        info = f"  {prot}: {len(sub)} frames"
        if not args.all_frames and 'target_lipid_bound' in sub.columns:
            occ = sub['target_lipid_bound'].sum() / len(sub) * 100
            info += f", occ={occ:.0f}%"
        if has_scd:
            info += f", scd={sub['local_scd'].mean():.3f}"
        if has_scd_xfs:
            info += f", scd_xfs={sub['local_scd_excl_fs'].mean():.3f}"
        print(info)

    print(f"{'='*70}\n")

    output_dir = args.output or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    comp_lipids = args.lipid if args.lipid else DEFAULT_LIPID_TYPES
    te_lags = args.te_lags if args.te_lags else [1, 2, 3, 5, 10]

    bound_only = not args.all_frames

    results = run_temporal_causal(
        df, comp_lipids=comp_lipids,
        max_lag=args.max_lag,
        te_lags=te_lags,
        n_surrogates=args.surrogates,
        ccm_E=args.ccm_E,
        run_ccm=not args.no_ccm,
        bound_only=bound_only,
    )

    if results:
        table = get_temporal_causal_table(results)
        suffix = '_allframes' if args.all_frames else ''
        out_file = os.path.join(output_dir, f'claire_v5_temporal_causal{suffix}.csv')
        table.to_csv(out_file, index=False)
        print(f"\nSaved: {out_file}")
        print(f"Total pairs tested: {len(table['pair'].unique())}")

    print("\nDone.")


if __name__ == '__main__':
    main()
