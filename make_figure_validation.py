#!/usr/bin/env python3
"""
CLAIRE v11 Synthetic Validation Figure

Panel A: 4 basic scenarios (GC F-statistic, forward vs reverse)
Panel B: Causal chain F-statistics (direct + transitive + null)
Panel C: Chain pathway diagram with F-values

SVG output for Illustrator. Large text throughout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from validation ──

# Part A: 4 basic scenarios
a_labels = ['ACTIVE\n(S$_{CD}$$\\rightarrow$CHOL)',
            'SELECTIVE\n(CHOL$\\rightarrow$S$_{CD}$)',
            'INDEPENDENT',
            'DOMAIN\n(co-arrival)']
a_fwd = [156.8, 3.7, 2.0, 3.1]
a_rev = [0.9, 178.6, 2.8, 1.0]
a_fwd_sig = [True, False, False, False]
a_rev_sig = [False, True, False, False]

# Part B: Chain pairs
b_labels = ['dist\n$\\rightarrow$\nfs$_{DIPC}$',
            'fs$_{DIPC}$\n$\\rightarrow$\nS$_{CD}$',
            'S$_{CD}$\n$\\rightarrow$\nCHOL',
            'fs$_{DIPC}$\n$\\rightarrow$\nCHOL',
            'dist\n$\\rightarrow$\nCHOL',
            'dist\n$\\rightarrow$\nS$_{CD}$',
            'null\nfs$\\rightarrow$S$_{CD}$',
            'null\nS$_{CD}$$\\rightarrow$CHOL']
b_fwd = [439.5, 2462.3, 665.6, 233.2, 111.4, 522.7, 4.4, 0.6]
b_rev = [1.9, 3.6, 397.5, 0.8, 1.2, 2.1, 0.6, 0.7]
b_types = ['direct', 'direct', 'direct', 'trans.', 'trans.', 'trans.', 'null', 'null']
b_colors_fwd = ['#2166AC']*3 + ['#4393C3']*3 + ['#AAAAAA']*2
b_colors_rev = ['#B2182B']*3 + ['#D6604D']*3 + ['#AAAAAA']*2

# ── Figure ──
fig = plt.figure(figsize=(16, 14))

# ── Panel A ──
ax1 = fig.add_axes([0.07, 0.72, 0.90, 0.25])
x1 = np.arange(4)
w = 0.35

ax1.bar(x1 - w/2, a_fwd, w, color='#2166AC', edgecolor='black', lw=1.5, label='Forward')
ax1.bar(x1 + w/2, a_rev, w, color='#B2182B', edgecolor='black', lw=1.5, label='Reverse')

for i in range(4):
    if a_fwd_sig[i]:
        ax1.text(x1[i] - w/2, a_fwd[i] + 5, '***', ha='center', fontsize=16, fontweight='bold', color='#2166AC')
    if a_rev_sig[i]:
        ax1.text(x1[i] + w/2, a_rev[i] + 5, '***', ha='center', fontsize=16, fontweight='bold', color='#B2182B')

ax1.axhline(5.0, color='gray', ls='--', lw=1.5, alpha=0.7)
ax1.set_ylabel('GC F-statistic', fontsize=18, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(a_labels, fontsize=15, fontweight='bold')
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0, 220)
ax1.legend(fontsize=15, loc='upper center', ncol=2)
ax1.set_title('A   Four basic scenarios', fontsize=20, fontweight='bold', loc='left', pad=10)

# ── Panel B ──
ax2 = fig.add_axes([0.07, 0.38, 0.55, 0.28])
x2 = np.arange(8)

bars_f = ax2.bar(x2 - w/2, b_fwd, w, color=b_colors_fwd, edgecolor='black', lw=1.5, label='Forward')
bars_r = ax2.bar(x2 + w/2, b_rev, w, color=b_colors_rev, edgecolor='black', lw=1.5, label='Reverse')

# *** markers for significant forward
for i in range(6):
    ax2.text(x2[i] - w/2, min(b_fwd[i] + 30, 2500), '***', ha='center', fontsize=14, fontweight='bold', color='#2166AC')
# reverse significant for S_CD -> CHOL
ax2.text(x2[2] + w/2, b_rev[2] + 30, '***', ha='center', fontsize=14, fontweight='bold', color='#B2182B')

ax2.axhline(5.0, color='gray', ls='--', lw=1.5, alpha=0.7)

# Bracket labels
ax2.annotate('', xy=(0, -180), xytext=(2, -180), xycoords=('data', 'data'), textcoords=('data', 'data'),
             arrowprops=dict(arrowstyle='-', color='black', lw=2))
ax2.text(1, -280, 'direct', ha='center', fontsize=14, fontweight='bold')
ax2.annotate('', xy=(3, -180), xytext=(5, -180), xycoords=('data', 'data'), textcoords=('data', 'data'),
             arrowprops=dict(arrowstyle='-', color='black', lw=2))
ax2.text(4, -280, 'transitive', ha='center', fontsize=14, fontweight='bold')
ax2.annotate('', xy=(6, -180), xytext=(7, -180), xycoords=('data', 'data'), textcoords=('data', 'data'),
             arrowprops=dict(arrowstyle='-', color='black', lw=2))
ax2.text(6.5, -280, 'null', ha='center', fontsize=14, fontweight='bold')

ax2.set_ylabel('GC F-statistic', fontsize=18, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(b_labels, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(-400, 2700)
ax2.legend(fontsize=14, loc='upper right')
ax2.set_title('B   Causal chain validation', fontsize=20, fontweight='bold', loc='left', pad=10)

# ── Panel C: Pathway diagram ──
ax3 = fig.add_axes([0.66, 0.38, 0.32, 0.28])
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(-1, 11)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('C   Detected pathway', fontsize=20, fontweight='bold', loc='left', pad=10)

# Boxes
bw, bh = 3.0, 1.8
dist_b = mpatches.FancyBboxPatch((0, 7.5), bw, bh, boxstyle="round,pad=0.2",
                                  facecolor='#FDB863', edgecolor='black', lw=2.5)
fs_b = mpatches.FancyBboxPatch((7, 7.5), bw, bh, boxstyle="round,pad=0.2",
                                facecolor='#F4A582', edgecolor='black', lw=2.5)
scd_b = mpatches.FancyBboxPatch((0, 1), bw, bh, boxstyle="round,pad=0.2",
                                 facecolor='#92C5DE', edgecolor='black', lw=2.5)
chol_b = mpatches.FancyBboxPatch((7, 1), bw, bh, boxstyle="round,pad=0.2",
                                  facecolor='#B2DF8A', edgecolor='black', lw=2.5)
for p in [dist_b, fs_b, scd_b, chol_b]:
    ax3.add_patch(p)

ax3.text(1.5, 8.4, 'GM3\ndistance', ha='center', va='center', fontsize=14, fontweight='bold')
ax3.text(8.5, 8.4, 'first shell\nDIPC', ha='center', va='center', fontsize=14, fontweight='bold')
ax3.text(1.5, 1.9, 'local\nS$_{CD}$', ha='center', va='center', fontsize=14, fontweight='bold')
ax3.text(8.5, 1.9, 'CHOL\ncount', ha='center', va='center', fontsize=14, fontweight='bold')

# Arrows with F-values
# dist -> fs_DIPC
ax3.annotate('', xy=(7, 8.4), xytext=(3, 8.4),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=3.5))
ax3.text(5.0, 9.0, 'F = 440', ha='center', fontsize=14, fontweight='bold', color='#2166AC')

# fs_DIPC -> S_CD
ax3.annotate('', xy=(1.5, 2.8), xytext=(8.5, 7.5),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=3.5))
ax3.text(3.5, 5.5, 'F = 2462', ha='center', fontsize=14, fontweight='bold', color='#2166AC', rotation=33)

# S_CD -> CHOL
ax3.annotate('', xy=(7, 1.9), xytext=(3, 1.9),
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=3.5))
ax3.text(5.0, 1.0, 'F = 666', ha='center', fontsize=14, fontweight='bold', color='#2166AC')

# dist -> CHOL (transitive, dashed)
ax3.annotate('', xy=(7.5, 2.8), xytext=(2.0, 7.5),
             arrowprops=dict(arrowstyle='->', color='#999999', lw=2, ls='--'))
ax3.text(6.3, 5.5, 'F = 111\n(transitive)', ha='center', fontsize=12, color='#999999', fontstyle='italic', rotation=-45)

# ── Panel D: F-value decay ──
ax4 = fig.add_axes([0.07, 0.06, 0.42, 0.24])

chain_hops = [0, 1, 2]
chain_f = [2462, 233, 111]
chain_labels_d = ['direct\n(1 hop)', 'transitive\n(2 hops)', 'transitive\n(3 hops)']

ax4.bar(chain_hops, chain_f, 0.6, color=['#2166AC', '#4393C3', '#92C5DE'],
        edgecolor='black', lw=1.5)
for i, f in enumerate(chain_f):
    ax4.text(i, f + 50, f'F = {f}', ha='center', fontsize=15, fontweight='bold')

ax4.axhline(5.0, color='gray', ls='--', lw=1.5, alpha=0.7)
ax4.set_xticks(chain_hops)
ax4.set_xticklabels(chain_labels_d, fontsize=14, fontweight='bold')
ax4.set_ylabel('GC F-statistic', fontsize=18, fontweight='bold')
ax4.tick_params(axis='y', labelsize=15)
ax4.set_ylim(0, 2800)
ax4.set_title('D   F-statistic decay with causal distance', fontsize=20, fontweight='bold', loc='left', pad=10)

# ── Panel E: TE comparison ──
ax5 = fig.add_axes([0.58, 0.06, 0.38, 0.24])

te_scenarios = ['ACTIVE\n(GC ***)', 'SELECTIVE\n(GC ***)', 'INDEP.\n(GC n.s.)', 'chain\ndirect\n(GC ***)']
te_gc_p = [0.000, 0.000, 0.240, 0.000]
te_te_p = [0.680, 0.160, 0.280, 1.000]

x5 = np.arange(4)
ax5.bar(x5 - w/2, te_gc_p, w, color='#2166AC', edgecolor='black', lw=1.5, label='GC p-value')
ax5.bar(x5 + w/2, te_te_p, w, color='#FDB863', edgecolor='black', lw=1.5, label='TE p-value')
ax5.axhline(0.05, color='red', ls='--', lw=2, alpha=0.8)
ax5.text(3.7, 0.07, 'p = 0.05', fontsize=13, color='red', fontstyle='italic')

ax5.set_ylabel('p-value', fontsize=18, fontweight='bold')
ax5.set_xticks(x5)
ax5.set_xticklabels(te_scenarios, fontsize=13, fontweight='bold')
ax5.tick_params(axis='y', labelsize=15)
ax5.set_ylim(0, 1.15)
ax5.legend(fontsize=14, loc='upper left')
ax5.set_title('E   GC vs TE detection power', fontsize=20, fontweight='bold', loc='left', pad=10)

plt.savefig('/mnt/user-data/outputs/Figure_synthetic_validation.svg', format='svg', bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/Figure_synthetic_validation.png', dpi=300, bbox_inches='tight')
print("Saved Figure_synthetic_validation.svg/.png")
