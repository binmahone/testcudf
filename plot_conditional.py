#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data: cast(coalesce(if(y==1, x, 0), 0) as double)
# 200 input cols (x1,y1,...,x100,y100) -> 100 output cols
strategy1_times = [53.261, 17.637, 17.602, 17.617, 17.609, 17.600, 17.611, 
                   17.622, 17.615, 17.621, 17.619, 17.621, 17.605, 17.604, 
                   17.605, 17.577, 17.530, 17.534, 17.585, 17.536, 17.553, 
                   17.552, 17.565, 17.547, 17.549]

strategy2_times = [90.948, 34.674, 34.689, 34.708, 34.672, 34.669, 34.692, 
                   34.648, 34.709, 34.698, 34.688, 34.652, 34.694, 34.659, 
                   34.636, 34.687, 34.677, 34.648, 34.656, 34.667, 34.698, 
                   34.634, 34.662, 34.649, 34.662]

num_warmup = 5
num_runs = 20

s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CUDF cast(coalesce(if(y==1,x,0),0) as double)\n' +
             '(200 input cols -> 100 output cols, 1GB->2GB, RMM Pool)', 
             fontsize=14, fontweight='bold')

# Plot 1: Full timeline
ax1 = axes[0, 0]
x_all = list(range(1, len(strategy1_times) + 1))
ax1.plot(x_all, strategy1_times, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6, color='blue')
ax1.plot(x_all, strategy2_times, 's-', 
         label='Strategy 2 (Concat-Process-Split)',
         linewidth=2, markersize=6, color='green')
ax1.axvline(x=num_warmup, color='red', linestyle='--', alpha=0.5, 
            label='Warmup end')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Time (ms)', fontsize=11)
ax1.set_title('All Iterations', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Timed runs only
ax2 = axes[0, 1]
x_timed = list(range(1, num_runs + 1))
ax2.plot(x_timed, s1_timed, 'o-', label='Strategy 1', 
         linewidth=2, markersize=6, color='blue')
ax2.plot(x_timed, s2_timed, 's-', label='Strategy 2',
         linewidth=2, markersize=6, color='green')
ax2.axhline(y=np.mean(s1_timed), color='blue', linestyle=':', alpha=0.5)
ax2.axhline(y=np.mean(s2_timed), color='green', linestyle=':', alpha=0.5)
ax2.set_xlabel('Run Number', fontsize=11)
ax2.set_ylabel('Time (ms)', fontsize=11)
ax2.set_title('Timed Runs (Perfect Stability)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax2.text(10, np.mean(s1_timed) - 0.5, 
         f'S1: {np.mean(s1_timed):.2f}ms (CV={np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%)', 
         color='blue', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(10, np.mean(s2_timed) + 0.2, 
         f'S2: {np.mean(s2_timed):.2f}ms (CV={np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%)', 
         color='green', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Box plot
ax3 = axes[1, 0]
bp = ax3.boxplot([s1_timed, s2_timed], 
                  tick_labels=['Strategy 1\nIndividual\n(100 pairs)', 
                               'Strategy 2\nConcat-Split'],
                  patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')

for box in bp['boxes']:
    box.set_linewidth(2)
for median in bp['medians']:
    median.set_linewidth(2)
    median.set_color('red')

ax3.set_ylabel('Time (ms)', fontsize=11)
ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

speedup = np.mean(s2_timed) / np.mean(s1_timed)
stats_text = (f'S1: {np.mean(s1_timed):.2f} ± {np.std(s1_timed):.3f}ms\n'
              f'S2: {np.mean(s2_timed):.2f} ± {np.std(s2_timed):.3f}ms\n'
              f'S1 is {speedup:.2f}x faster!')
ax3.text(0.5, 0.98, stats_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Performance breakdown estimate
ax4 = axes[1, 1]

# Estimate breakdown based on previous tests
s1_breakdown = ['Total']
s1_values = [np.mean(s1_timed)]

s2_breakdown = ['Concat\n(x2)', 'Process', 'Split', 'Total']
# Estimate: 2 concats ~9ms, process ~25ms, split ~0.2ms
s2_values = [9, 25.5, 0.2, np.mean(s2_timed)]

x_pos = np.arange(len(s2_breakdown))
bars = ax4.bar(x_pos, s2_values, color=['orange', 'red', 'yellow', 'green'],
               edgecolor='black', linewidth=1.5, alpha=0.7)

# Add S1 reference line
ax4.axhline(y=s1_values[0], color='blue', linestyle='--', linewidth=2,
            label=f'Strategy 1: {s1_values[0]:.1f}ms', alpha=0.7)

ax4.set_ylabel('Time (ms)', fontsize=11)
ax4.set_title('Strategy 2 Breakdown (Estimated)', fontsize=12, 
              fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(s2_breakdown)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/hongbin/code/testCUDF/conditional_coalesce.png', 
            dpi=150, bbox_inches='tight')
print("\nPlot saved to: /home/hongbin/code/testCUDF/conditional_coalesce.png")

# Analysis
print("\n" + "="*70)
print("ANALYSIS - cast(coalesce(if(y==1,x,0),0) as double)")
print("="*70)
print("\nWorkload:")
print("  Input: 200 columns (x1,y1, x2,y2, ..., x100,y100)")
print("  Output: 100 columns (DOUBLE)")
print("  Expression per pair: cast(coalesce(if(y==1, x, 0), 0) as double)")

print("\nStrategy 1 (Individual - 100 pairs):")
print(f"  Mean:   {np.mean(s1_timed):.3f} ms")
print(f"  Median: {np.median(s1_timed):.3f} ms")
print(f"  Min:    {np.min(s1_timed):.3f} ms")
print(f"  Max:    {np.max(s1_timed):.3f} ms")
print(f"  StdDev: {np.std(s1_timed):.3f} ms")
print(f"  CV:     {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%")

print("\nStrategy 2 (Concat x2 -> Process -> Split):")
print(f"  Mean:   {np.mean(s2_timed):.3f} ms")
print(f"  Median: {np.median(s2_timed):.3f} ms")
print(f"  Min:    {np.min(s2_timed):.3f} ms")
print(f"  Max:    {np.max(s2_timed):.3f} ms")
print(f"  StdDev: {np.std(s2_timed):.3f} ms")
print(f"  CV:     {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%")

ratio = np.mean(s2_timed) / np.mean(s1_timed)
print("\n" + "="*70)
print(f"RESULT: Strategy 1 is {ratio:.2f}x FASTER! ⭐")
print(f"Time saved: {np.mean(s2_timed) - np.mean(s1_timed):.3f} ms")
print(f"Improvement: {(ratio-1)*100:.1f}%")

print("\n" + "="*70)
print("WORKLOAD COMPARISON")
print("="*70)
print("\nSimple cast(coalesce(x,0) as double) [100 cols]:")
print("  Strategy 1: 10.07 ms")
print("  Strategy 2: 20.41 ms")
print("  S1 advantage: 2.03x")

print("\nConditional cast(coalesce(if(y==1,x,0),0) as double) [100 pairs]:")
print(f"  Strategy 1: {np.mean(s1_timed):.2f} ms")
print(f"  Strategy 2: {np.mean(s2_timed):.2f} ms")
print(f"  S1 advantage: {ratio:.2f}x")

print("\nConclusion:")
print("  - More complex expression takes longer (17.6ms vs 10.1ms)")
print("  - But S1 still wins!")
print("  - Concat overhead (2x ~9ms) hurts S2 significantly")

print("\n" + "="*70)

