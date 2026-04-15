#!/usr/bin/env python3
"""
CLAIRE v5 Stage 2: Static composition analysis

Hierarchical Bayesian analysis + count-based causal mediation.
No MDAnalysis dependency. Reads composition_data.csv from Stage 1.

If indirect pathway is detected with high target lipid occupancy,
advises the user to run run_temporal.py for causal direction analysis.

Usage:
    python run_analyze.py
    python run_analyze.py path/to/composition_data.csv
    python run_analyze.py --mediator CHOL
    python run_analyze.py --skip-mediation
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.composition import CompositionAnalyzer
from config import OUTPUT_DIR, DEFAULT_LIPID_TYPES, TARGET_LIPID


def _check_temporal_needed(df, mediation, analyzer):
    """Check if MERIT temporal analysis is recommended.

    Criteria:
    1. Indirect pathway detected (mediation > 20%)
    2. Mean occupancy > 80%
    """
    if mediation is None:
        return False, ""

    occupancies = []
    for prot in df['protein'].unique():
        sub = df[df['protein'] == prot]
        occ = sub['target_lipid_bound'].sum() / len(sub)
        occupancies.append(occ)
    mean_occ = sum(occupancies) / len(occupancies) if occupancies else 0

    max_med = 0
    for resp, r in mediation.items():
        med_pct = abs(r['mediation_fraction'] * 100)
        if med_pct > max_med:
            max_med = med_pct

    if max_med > 20 and mean_occ > 0.8:
        return True, (f"Indirect pathway detected (mediation {max_med:.0f}%) "
                      f"with high occupancy ({mean_occ:.0%})")
    return False, ""


def main():
    parser = argparse.ArgumentParser(
        description='CLAIRE v5 Stage 2: Hierarchical Bayesian composition '
                    'analysis + count-based causal mediation')
    parser.add_argument('csv', nargs='?',
                        default=os.path.join(OUTPUT_DIR, 'composition_data.csv'),
                        help='Path to composition_data.csv (default: %(default)s)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as CSV)')
    parser.add_argument('--lipids', nargs='+', default=DEFAULT_LIPID_TYPES,
                        help='Composition lipids (default: %(default)s)')
    parser.add_argument('--target', default=TARGET_LIPID,
                        help='Target lipid (default: %(default)s)')
    parser.add_argument('--mediator', default='auto',
                        help='Mediator lipid (default: auto)')
    parser.add_argument('--skip-mediation', action='store_true',
                        help='Skip mediation analysis')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy v1 frequentist analysis')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--tune', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found")
        return
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  Proteins: {sorted(df['protein'].unique())}")
    print(f"  Frames: {df['frame'].min()}--{df['frame'].max()}")

    for prot in sorted(df['protein'].unique()):
        sub = df[df['protein'] == prot]
        occ = sub['target_lipid_bound'].sum() / len(sub) * 100
        print(f"    {prot}: occupancy {occ:.0f}%")

    output_dir = args.output or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    all_lipids = args.lipids + [args.target]
    analyzer = CompositionAnalyzer(all_lipids, target_lipid=args.target)

    # ── Hierarchical analysis (fractions) ──────────────────────────────
    if args.legacy:
        results = analyzer.analyze_composition_changes(df, method='binary')
    else:
        results = analyzer.analyze_composition_hierarchical(
            df, n_chains=args.chains, n_tune=args.tune,
            n_samples=args.samples, random_seed=args.seed)

        summary = analyzer.get_summary_table(results)
        out_file = os.path.join(output_dir, 'claire_v5_hierarchical.csv')
        summary.to_csv(out_file, index=False)
        print(f"\nSaved: {out_file}")

    # ── Count-based mediation ─────────────────────────────────────────
    mediation = None
    if not args.skip_mediation and not args.legacy:
        mediation = analyzer.analyze_mediation(
            df, mediator_lipid=args.mediator,
            n_chains=args.chains, n_tune=args.tune,
            n_samples=args.samples, random_seed=args.seed)

        if mediation:
            response_lipids = set(mediation.keys())
            actual_mediator = [lt for lt in analyzer.comp_lipids
                               if lt not in response_lipids][0]

            med_table = analyzer.get_mediation_table(mediation, actual_mediator)
            out_file = os.path.join(output_dir, 'claire_v5_mediation.csv')
            med_table.to_csv(out_file, index=False)
            print(f"\nSaved: {out_file}")

    # ── Advisory: temporal analysis needed? ────────────────────────────
    if mediation is not None:
        needed, reason = _check_temporal_needed(df, mediation, analyzer)
        if needed:
            has_distance = 'target_lipid_min_distance' in df.columns
            print("\n" + "=" * 70)
            print(">>> TEMPORAL CAUSAL ANALYSIS RECOMMENDED")
            print(f"    {reason}")
            print(f"    Static mediation cannot distinguish:")
            print(f"      Active recruitment (binding drives remodeling)")
            print(f"      Selective binding  (lipid environment attracts binding)")
            if has_distance:
                print(f"\n    Run:  python run_temporal.py {args.csv}")
            else:
                print(f"\n    Re-run Stage 1 (run_extract.py v5) to add distance data,")
                print(f"    then:  python run_temporal.py composition_data.csv")
            print("=" * 70)
        else:
            print("\n  Static analysis is sufficient for this system.")

    print("\nDone.")


if __name__ == '__main__':
    main()
