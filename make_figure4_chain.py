#!/usr/bin/env python3
"""
Figure 4: The causal chain of lipid domain formation around EGFR
NM publication quality. SVG for Illustrator. Large text.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 11)
ax.set_aspect('equal')
ax.axis('off')

# Color scheme
c_gm3 = '#FDB863'
c_dipc = '#D6604D'
c_dpsm = '#92C5DE'
c_scd = '#4393C3'
c_chol = '#B2DF8A'

# Boxes
bw, bh = 2.8, 2.0

boxes = [
    (0.5, 7.0, 'GM3\nbinding', c_gm3),
    (4.0, 7.0, 'DIPC\nexclusion\n(first shell)', c_dipc),
    (7.5, 7.0, 'DPSM\narrival\n(first shell)', c_dpsm),
    (11.0, 7.0, 'Membrane\nordering\n(S$_{CD}$ $\\uparrow$)', c_scd),
    (11.0, 2.5, 'CHOL\narrival\n(stabilization)', c_chol),
]

for x, y, label, color in boxes:
    box = mpatches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.25",
                                   facecolor=color, edgecolor='black', lw=2.5)
    ax.add_patch(box)
    ax.text(x + bw/2, y + bh/2, label, ha='center', va='center',
            fontsize=14, fontweight='bold')

# Arrows between boxes
arrow_props = dict(arrowstyle='->', color='#333333', lw=3.5)

# GM3 -> DIPC
ax.annotate('', xy=(4.0, 8.0), xytext=(3.3, 8.0), arrowprops=arrow_props)
# DIPC -> DPSM
ax.annotate('', xy=(7.5, 8.0), xytext=(6.8, 8.0), arrowprops=arrow_props)
# DPSM -> ordering
ax.annotate('', xy=(11.0, 8.0), xytext=(10.3, 8.0), arrowprops=arrow_props)
# CHOL -> ordering (upward, stabilization)
ax.annotate('', xy=(12.4, 7.0), xytext=(12.4, 4.5),
            arrowprops=dict(arrowstyle='->', color='#4DAF4A', lw=3.5))

# Step labels
ax.text(3.65, 9.4, 'Step 1', ha='center', fontsize=13, fontweight='bold', color='#666666')
ax.text(7.15, 9.4, 'Step 2', ha='center', fontsize=13, fontweight='bold', color='#666666')
ax.text(10.65, 9.4, 'Step 3', ha='center', fontsize=13, fontweight='bold', color='#666666')
ax.text(13.8, 5.5, 'Step 4', ha='center', fontsize=13, fontweight='bold', color='#666666')

# Evidence labels
ax.text(3.65, 6.3, 'LIPAC\n(established)', ha='center', fontsize=11,
        fontstyle='italic', color='#888888')
ax.text(7.15, 6.3, 'GC: fs_DIPC\n$\\rightarrow$ fs_DPSM', ha='center', fontsize=11,
        fontstyle='italic', color='#2166AC')
ax.text(10.65, 6.3, 'GC: fs_DPSM\n$\\rightarrow$ S$_{CD}$\nF ratio 2-9x', ha='center', fontsize=11,
        fontstyle='italic', color='#2166AC')
ax.text(13.8, 3.2, 'GC: CHOL\n$\\rightarrow$ S$_{CD}$\n(selective)', ha='center', fontsize=11,
        fontstyle='italic', color='#4DAF4A')

# Replicate labels
ax.text(7.5, 0.5, 'Reproduced across 2 independent replicates (8 protein copies)',
        ha='center', fontsize=14, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#999999'))

# Notch annotation
ax.text(7.5, -0.5, 'Absent in Notch (direct displacement only, no DPSM replacement)',
        ha='center', fontsize=12, fontstyle='italic', color='#999999')

plt.savefig('/mnt/user-data/outputs/Figure4_causal_chain.svg', format='svg', bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/Figure4_causal_chain.png', dpi=300, bbox_inches='tight')
print("Saved Figure4_causal_chain.svg/.png")
