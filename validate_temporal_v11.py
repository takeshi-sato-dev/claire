#!/usr/bin/env python3
"""
CLAIRE v11 Full Synthetic Validation

Part A: 4 basic scenarios (scd vs CHOL)
  1. ACTIVE:      S_CD ordering precedes CHOL arrival
  2. SELECTIVE:   CHOL arrival precedes ordering
  3. INDEPENDENT: no relationship
  4. DOMAIN:      simultaneous co-arrival (common driver)

Part B: Full causal chain
  dist -> fs_DIPC -> S_CD -> CHOL
  Tests all 6 pairs (3 direct + 3 transitive) + 2 null controls

Usage:
    python validate_temporal_v11.py
    python validate_temporal_v11.py --surrogates 100
"""
import os, sys, warnings, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from analysis.temporal_causal import test_causal_direction


def zscore(x):
    return (x - x.mean()) / x.std()


def run_all(n=2000, n_surr=50, output_dir='validation_v11'):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # ============================================================
    # PART A: 4 basic scenarios
    # ============================================================
    print("=" * 70)
    print("  PART A: Basic scenarios (S_CD vs CHOL)")
    print("=" * 70)

    # A1: ACTIVE (S_CD -> CHOL, lag 3)
    print("\n" + "-" * 60)
    print("  A1: ACTIVE (S_CD -> CHOL)")
    print("-" * 60)
    np.random.seed(42)
    scd1 = np.zeros(n); scd1[0] = 0.25
    for t in range(1, n):
        scd1[t] = 0.95*scd1[t-1] + 0.05*0.25 + np.random.randn()*0.008
    chol1 = np.zeros(n); chol1[0] = 9
    for t in range(3, n):
        chol1[t] = 0.85*chol1[t-1] + 0.15*9 + 8.0*(scd1[t-3]-0.25) + np.random.randn()*0.3
    dipc1 = 18 - chol1 + np.random.randn(n)*0.5
    r = test_causal_direction(zscore(scd1), zscore(chol1), max_lag=10,
                              n_surrogates=n_surr, run_ccm=False, condition=zscore(dipc1))
    all_results['A1_active'] = r

    # A2: SELECTIVE (CHOL -> S_CD, lag 3)
    print("\n" + "-" * 60)
    print("  A2: SELECTIVE (CHOL -> S_CD)")
    print("-" * 60)
    np.random.seed(99)
    chol2 = np.zeros(n); chol2[0] = 9
    for t in range(1, n):
        chol2[t] = 0.95*chol2[t-1] + 0.05*9 + np.random.randn()*0.1
    scd2 = np.zeros(n); scd2[0] = 0.25
    for t in range(3, n):
        scd2[t] = 0.9*scd2[t-1] + 0.1*0.25 + 0.01*(chol2[t-3]-9) + np.random.randn()*0.005
    dipc2 = 18 - chol2 + np.random.randn(n)*0.5
    r2 = test_causal_direction(zscore(scd2), zscore(chol2), max_lag=10,
                               n_surrogates=n_surr, run_ccm=False, condition=zscore(dipc2))
    all_results['A2_selective'] = r2

    # A3: INDEPENDENT
    print("\n" + "-" * 60)
    print("  A3: INDEPENDENT")
    print("-" * 60)
    np.random.seed(55)
    scd3 = np.zeros(n); scd3[0] = 0.25
    for t in range(1, n):
        scd3[t] = 0.95*scd3[t-1] + 0.05*0.25 + np.random.randn()*0.008
    chol3 = np.zeros(n); chol3[0] = 9
    for t in range(1, n):
        chol3[t] = 0.95*chol3[t-1] + 0.05*9 + np.random.randn()*0.1
    dipc3 = np.zeros(n); dipc3[0] = 7
    for t in range(1, n):
        dipc3[t] = 0.95*dipc3[t-1] + 0.05*7 + np.random.randn()*0.1
    r3 = test_causal_direction(zscore(scd3), zscore(chol3), max_lag=10,
                               n_surrogates=n_surr, run_ccm=False, condition=zscore(dipc3))
    all_results['A3_independent'] = r3

    # A4: DOMAIN (simultaneous co-arrival)
    print("\n" + "-" * 60)
    print("  A4: DOMAIN DYNAMICS (simultaneous)")
    print("-" * 60)
    np.random.seed(77)
    domain = np.cumsum(np.random.randn(n)*0.1)
    dist4 = 4.5 - 0.1*domain + np.random.randn(n)*0.1
    chol4 = 9 + 0.5*domain + np.random.randn(n)*0.3
    dipc4 = 7 - 0.5*domain + np.random.randn(n)*0.3
    r4 = test_causal_direction(zscore(dist4), zscore(chol4), max_lag=10,
                               n_surrogates=n_surr, run_ccm=False, condition=zscore(dipc4))
    all_results['A4_domain'] = r4

    # ============================================================
    # PART B: Full causal chain
    # ============================================================
    print("\n" + "=" * 70)
    print("  PART B: Full causal chain")
    print("  dist -> fs_DIPC -> S_CD -> CHOL")
    print("=" * 70)

    np.random.seed(500)
    dist = np.zeros(n); dist[0] = 4.5
    for t in range(1, n):
        dist[t] = 0.95*dist[t-1] + 0.05*4.5 + np.random.randn()*0.15
    dist = np.clip(dist, 2.5, 7.0)

    fs_dipc = np.zeros(n); fs_dipc[0] = 3
    for t in range(1, n):
        fs_dipc[t] = 0.8*fs_dipc[t-1] + 0.2*3 + 0.5*(dist[t-1]-4.5) + np.random.randn()*0.3
    fs_dipc = np.clip(fs_dipc, 0, 8)

    scd = np.zeros(n); scd[0] = 0.25
    for t in range(2, n):
        scd[t] = 0.85*scd[t-1] + 0.15*0.25 - 0.015*(fs_dipc[t-2]-3) + np.random.randn()*0.005

    chol = np.zeros(n); chol[0] = 9
    for t in range(3, n):
        chol[t] = 0.85*chol[t-1] + 0.15*9 + 5.0*(scd[t-3]-0.25) + np.random.randn()*0.3

    dipc = 18 - chol + np.random.randn(n)*0.5

    print(f"  Dist:    {dist.mean():.2f} +/- {dist.std():.2f} A")
    print(f"  fs_DIPC: {fs_dipc.mean():.1f} +/- {fs_dipc.std():.1f}")
    print(f"  S_CD:    {scd.mean():.3f} +/- {scd.std():.3f}")
    print(f"  CHOL:    {chol.mean():.1f} +/- {chol.std():.1f}")

    chain_pairs = [
        ('dist', 'fs_DIPC', dist, fs_dipc, None, 'direct'),
        ('fs_DIPC', 'S_CD', fs_dipc, scd, None, 'direct'),
        ('S_CD', 'CHOL', scd, chol, dipc, 'direct'),
        ('fs_DIPC', 'CHOL', fs_dipc, chol, dipc, 'transitive'),
        ('dist', 'CHOL', dist, chol, dipc, 'transitive'),
        ('dist', 'S_CD', dist, scd, None, 'transitive'),
    ]

    for name_x, name_y, x, y, cond, link_type in chain_pairs:
        pair_name = f'B_{name_x}_vs_{name_y}'
        print(f"\n{'-'*60}")
        print(f"  {name_x} -> {name_y}  ({link_type})")
        print(f"{'-'*60}")
        cond_z = zscore(cond) if cond is not None else None
        r = test_causal_direction(zscore(x), zscore(y), max_lag=10,
                                  n_surrogates=n_surr, run_ccm=False,
                                  condition=cond_z)
        all_results[pair_name] = r
        all_results[pair_name]['link_type'] = link_type

    # Null controls
    print(f"\n{'-'*60}")
    print(f"  NULL controls")
    print(f"{'-'*60}")
    np.random.seed(999)
    dist_n = 4.5 + np.cumsum(np.random.randn(n)*0.03)
    fs_n = 3.0 + np.cumsum(np.random.randn(n)*0.05)
    scd_n = 0.25 + np.cumsum(np.random.randn(n)*0.002)
    chol_n = 9 + np.cumsum(np.random.randn(n)*0.05)
    dipc_n = 7 + np.cumsum(np.random.randn(n)*0.05)

    r_n1 = test_causal_direction(zscore(fs_n), zscore(scd_n), max_lag=10,
                                  n_surrogates=n_surr, run_ccm=False, condition=None)
    all_results['B_null_fs_scd'] = r_n1
    all_results['B_null_fs_scd']['link_type'] = 'null'
    r_n2 = test_causal_direction(zscore(scd_n), zscore(chol_n), max_lag=10,
                                  n_surrogates=n_surr, run_ccm=False, condition=zscore(dipc_n))
    all_results['B_null_scd_chol'] = r_n2
    all_results['B_null_scd_chol']['link_type'] = 'null'

    # ============================================================
    # Save results
    # ============================================================
    rows = []
    for name, r in all_results.items():
        fwd = r['forward']
        rev = r['reverse']
        rows.append({
            'scenario': name,
            'classification': r['classification'],
            'link_type': r.get('link_type', ''),
            'gc_fwd_F': fwd['gc']['observed_statistic'],
            'gc_fwd_p': fwd['gc']['surrogate_p'],
            'gc_rev_F': rev['gc']['observed_statistic'],
            'gc_rev_p': rev['gc']['surrogate_p'],
            'cgc_fwd_p': fwd.get('cgc', {}).get('p_value', float('nan')),
            'cgc_rev_p': rev.get('cgc', {}).get('p_value', float('nan')),
            'te_fwd': fwd['te']['observed_statistic'],
            'te_fwd_p': fwd['te']['surrogate_p'],
            'te_rev': rev['te']['observed_statistic'],
            'te_rev_p': rev['te']['surrogate_p'],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'validation_full_results.csv')
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("  FULL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Scenario':<25s} {'Type':<12s} {'Result':<22s} {'GCf':>7s} {'p':>6s} {'GCr':>7s} {'p':>6s}")
    print("  " + "-" * 85)
    for _, row in df.iterrows():
        print(f"  {row['scenario']:<25s} {row['link_type']:<12s} {row['classification']:<22s} "
              f"{row['gc_fwd_F']:7.1f} {row['gc_fwd_p']:6.3f} {row['gc_rev_F']:7.1f} {row['gc_rev_p']:6.3f}")

    print(f"\n  Saved: {csv_path}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogates', type=int, default=50)
    parser.add_argument('--n-frames', type=int, default=2000)
    parser.add_argument('--output-dir', default='validation_v11')
    args = parser.parse_args()
    run_all(n=args.n_frames, n_surr=args.surrogates, output_dir=args.output_dir)
