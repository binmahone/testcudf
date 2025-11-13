#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data with pool_memory_resource
strategy1_times = [74.395, 8.410, 8.359, 8.380, 8.353, 8.384, 8.387, 8.390, 
                   8.423, 8.349, 8.380, 8.361, 8.361, 8.362, 8.364, 8.395, 
                   8.370, 8.354, 8.340, 8.323, 8.391, 8.406, 8.388, 8.335, 
                   8.352]

strategy2_times = [16.527, 13.939, 13.884, 13.913, 13.900, 13.883, 14.131, 
                   13.887, 14.137, 14.124, 14.147, 14.113, 14.092, 13.888, 
                   14.119, 13.890, 14.124, 14.133, 14.233, 14.106, 13.878, 
                   14.128, 14.134, 14.127, 14.125]

num_warmup = 5
num_runs = 20

# Split into warmup and timed runs
s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CUDF Coalesce Performance (200 cols, 1GB, with RMM Pool)', 
             fontsize=14, fontweight='bold')

# Plot 1: Full timeline
ax1 = axes[0, 0]
x_all = list(range(1, len(strategy1_times) + 1))
ax1.plot(x_all, strategy1_times, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6, color='blue')
ax1.plot(x_all, strategy2_times, 's-', 
         label='Strategy 2 (Concat-Coalesce-Split)',
         linewidth=2, markersize=6, color='green')
ax1.axvline(x=num_warmup, color='red', linestyle='--', alpha=0.5, 
            label='Warmup end')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Time (ms)', fontsize=11)
ax1.set_title('All Iterations (Warmup + Timed)', fontsize=12, 
              fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Timed runs only (zoomed in)
ax2 = axes[0, 1]
x_timed = list(range(1, num_runs + 1))
ax2.plot(x_timed, s1_timed, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6, color='blue')
ax2.plot(x_timed, s2_timed, 's-', 
         label='Strategy 2 (Concat-Coalesce-Split)',
         linewidth=2, markersize=6, color='green')
ax2.axhline(y=np.mean(s1_timed), color='blue', linestyle=':', 
            alpha=0.5, linewidth=1.5)
ax2.axhline(y=np.mean(s2_timed), color='green', linestyle=':', 
            alpha=0.5, linewidth=1.5)
ax2.set_xlabel('Run Number', fontsize=11)
ax2.set_ylabel('Time (ms)', fontsize=11)
ax2.set_title('Timed Runs Only (Stable Performance!)', fontsize=12, 
              fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add text annotations
ax2.text(10, np.mean(s1_timed) + 0.1, 
         f'S1 Avg: {np.mean(s1_timed):.2f}ms', 
         color='blue', fontsize=9, ha='center')
ax2.text(10, np.mean(s2_timed) + 0.3, 
         f'S2 Avg: {np.mean(s2_timed):.2f}ms', 
         color='green', fontsize=9, ha='center')

# Plot 3: Box plot comparison
ax3 = axes[1, 0]
data_to_plot = [s1_timed, s2_timed]
bp = ax3.boxplot(data_to_plot, tick_labels=['Strategy 1\n(Individual)', 
                                              'Strategy 2\n(Concat-Split)'],
                  patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')

# Make boxes more visible
for box in bp['boxes']:
    box.set_linewidth(2)
for whisker in bp['whiskers']:
    whisker.set_linewidth(1.5)
for cap in bp['caps']:
    cap.set_linewidth(1.5)
for median in bp['medians']:
    median.set_linewidth(2)
    median.set_color('red')

ax3.set_ylabel('Time (ms)', fontsize=11)
ax3.set_title('Distribution Comparison (Timed Runs)', fontsize=12, 
              fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats1 = f'Mean: {np.mean(s1_timed):.3f}ms\nStd:  {np.std(s1_timed):.3f}ms\nCV:   {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%'
stats2 = f'Mean: {np.mean(s2_timed):.3f}ms\nStd:  {np.std(s2_timed):.3f}ms\nCV:   {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%'

ax3.text(1, max(s1_timed) - 0.03, stats1, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         verticalalignment='top', horizontalalignment='center')
ax3.text(2, max(s2_timed) + 0.1, stats2, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         verticalalignment='bottom', horizontalalignment='center')

# Plot 4: Histogram
ax4 = axes[1, 1]
bins = np.linspace(min(min(s1_timed), min(s2_timed)), 
                   max(max(s1_timed), max(s2_timed)), 20)
ax4.hist(s1_timed, bins=bins, alpha=0.6, label='Strategy 1', 
         color='blue', edgecolor='black', linewidth=1.2)
ax4.hist(s2_timed, bins=bins, alpha=0.6, label='Strategy 2', 
         color='green', edgecolor='black', linewidth=1.2)
ax4.axvline(np.mean(s1_timed), color='blue', linestyle='--', 
            linewidth=2, alpha=0.7)
ax4.axvline(np.mean(s2_timed), color='green', linestyle='--', 
            linewidth=2, alpha=0.7)
ax4.set_xlabel('Time (ms)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Time Distribution Histogram', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/hongbin/code/testCUDF/coalesce_pool_comparison.png', 
            dpi=150, bbox_inches='tight')
print("\nPlot saved to: /home/hongbin/code/testCUDF/coalesce_pool_comparison.png")

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED ANALYSIS (WITH RMM POOL)")
print("="*70)
print("\nStrategy 1 (Individual coalesce):")
print(f"  Mean:   {np.mean(s1_timed):.3f} ms")
print(f"  Median: {np.median(s1_timed):.3f} ms")
print(f"  Min:    {np.min(s1_timed):.3f} ms")
print(f"  Max:    {np.max(s1_timed):.3f} ms")
print(f"  StdDev: {np.std(s1_timed):.3f} ms")
print(f"  CV:     {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%")
print(f"  Range:  {np.max(s1_timed) - np.min(s1_timed):.3f} ms")

print("\nStrategy 2 (Concat->Coalesce->Split view):")
print(f"  Mean:   {np.mean(s2_timed):.3f} ms")
print(f"  Median: {np.median(s2_timed):.3f} ms")
print(f"  Min:    {np.min(s2_timed):.3f} ms")
print(f"  Max:    {np.max(s2_timed):.3f} ms")
print(f"  StdDev: {np.std(s2_timed):.3f} ms")
print(f"  CV:     {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%")
print(f"  Range:  {np.max(s2_timed) - np.min(s2_timed):.3f} ms")

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

ratio = np.mean(s1_timed) / np.mean(s2_timed)
print(f"\nTime ratio (S1/S2): {ratio:.2f}")
if ratio < 1.0:
    print(f"Result: Strategy 1 is {1/ratio:.2f}x FASTER! ⭐")
    print(f"Strategy 1 wins by: {np.mean(s2_timed) - np.mean(s1_timed):.3f} ms")
else:
    print(f"Result: Strategy 2 is {ratio:.2f}x faster")
    print(f"Strategy 2 wins by: {np.mean(s1_timed) - np.mean(s2_timed):.3f} ms")

print("\n" + "="*70)
print("STABILITY ANALYSIS")
print("="*70)
print(f"\nStrategy 1 CV: {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}% ✓ Excellent!")
print(f"Strategy 2 CV: {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}% ✓ Excellent!")
print("\nBoth strategies show excellent stability with RMM Pool!")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("\n1. RMM Pool eliminates performance degradation")
print("2. Both strategies now show stable, predictable performance")
print("3. Strategy 1 is now FASTER with pool (8.4ms vs 14.1ms)")
print("4. The concat/split overhead (5.7ms) cannot be amortized")
print("   when individual operations are this fast")

print("\n" + "="*70)

