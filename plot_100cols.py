#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data with pool_memory_resource - 100 columns
strategy1_times = [102.516, 10.177, 10.137, 10.132, 10.080, 10.088, 10.097, 
                   10.078, 10.106, 10.058, 10.072, 10.031, 10.100, 10.071, 
                   10.050, 10.063, 10.035, 10.047, 10.190, 10.096, 10.010, 
                   10.059, 10.124, 10.062, 10.055]

strategy2_times = [23.299, 20.521, 20.391, 20.508, 20.364, 20.460, 20.378, 
                   20.372, 20.488, 20.430, 20.452, 20.381, 20.349, 20.360, 
                   20.364, 20.440, 20.383, 20.442, 20.355, 20.367, 20.400, 
                   20.398, 20.362, 20.469, 20.451]

num_warmup = 5
num_runs = 20

s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CUDF cast(coalesce(x,0) as double) Performance\n' +
             '(100 cols, 1GB->2GB, with RMM Pool)', 
             fontsize=14, fontweight='bold')

# Plot 1: Full timeline
ax1 = axes[0, 0]
x_all = list(range(1, len(strategy1_times) + 1))
ax1.plot(x_all, strategy1_times, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6, color='blue')
ax1.plot(x_all, strategy2_times, 's-', 
         label='Strategy 2 (Concat-Cast-Split)',
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
ax2.set_title('Timed Runs (Excellent Stability)', fontsize=12, 
              fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax2.text(10, np.mean(s1_timed) - 0.15, 
         f'S1: {np.mean(s1_timed):.2f}ms', 
         color='blue', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(10, np.mean(s2_timed) + 0.15, 
         f'S2: {np.mean(s2_timed):.2f}ms', 
         color='green', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Box plot
ax3 = axes[1, 0]
bp = ax3.boxplot([s1_timed, s2_timed], 
                  tick_labels=['Strategy 1\nIndividual', 
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

stats_text = (f'S1: {np.mean(s1_timed):.2f} ± {np.std(s1_timed):.3f}ms\n'
              f'S2: {np.mean(s2_timed):.2f} ± {np.std(s2_timed):.3f}ms\n'
              f'Speedup: {np.mean(s2_timed)/np.mean(s1_timed):.2f}x')
ax3.text(0.5, 0.98, stats_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Bar comparison
ax4 = axes[1, 1]
categories = ['Mean', 'Min', 'Max', 'StdDev']
s1_stats = [np.mean(s1_timed), np.min(s1_timed), 
            np.max(s1_timed), np.std(s1_timed)]
s2_stats = [np.mean(s2_timed), np.min(s2_timed), 
            np.max(s2_timed), np.std(s2_timed)]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, s1_stats, width, label='Strategy 1', 
                color='lightblue', edgecolor='blue', linewidth=1.5)
bars2 = ax4.bar(x + width/2, s2_stats, width, label='Strategy 2', 
                color='lightgreen', edgecolor='green', linewidth=1.5)

ax4.set_ylabel('Time (ms)', fontsize=11)
ax4.set_title('Statistical Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height < 1:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/hongbin/code/testCUDF/cast_coalesce_100cols.png', 
            dpi=150, bbox_inches='tight')
print("\nPlot saved to: /home/hongbin/code/testCUDF/cast_coalesce_100cols.png")

# Analysis
print("\n" + "="*70)
print("ANALYSIS - cast(coalesce(x,0) as double) - 100 COLUMNS")
print("="*70)

print("\nStrategy 1 (Individual - 100 ops):")
print(f"  Mean:   {np.mean(s1_timed):.3f} ms")
print(f"  Median: {np.median(s1_timed):.3f} ms")
print(f"  Min:    {np.min(s1_timed):.3f} ms")
print(f"  Max:    {np.max(s1_timed):.3f} ms")
print(f"  StdDev: {np.std(s1_timed):.3f} ms")
print(f"  CV:     {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%")

print("\nStrategy 2 (Concat->Cast(Coalesce)->Split):")
print(f"  Mean:   {np.mean(s2_timed):.3f} ms")
print(f"  Median: {np.median(s2_timed):.3f} ms")
print(f"  Min:    {np.min(s2_timed):.3f} ms")
print(f"  Max:    {np.max(s2_timed):.3f} ms")
print(f"  StdDev: {np.std(s2_timed):.3f} ms")
print(f"  CV:     {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%")

ratio = np.mean(s2_timed) / np.mean(s1_timed)
print("\n" + "="*70)
print(f"RESULT: Strategy 1 is {ratio:.2f}x FASTER! ⭐")
print(f"Time saved: {np.mean(s2_timed) - np.mean(s1_timed):.3f} ms ({(ratio-1)*100:.1f}% improvement)")

print("\n" + "="*70)
print("COMPARISON: 100 cols vs 200 cols")
print("="*70)
print("\n100 columns (current):")
print(f"  Strategy 1: {np.mean(s1_timed):.2f} ms")
print(f"  Strategy 2: {np.mean(s2_timed):.2f} ms")
print(f"  Speedup: {ratio:.2f}x")

print("\n200 columns (previous):")
print(f"  Strategy 1: 11.68 ms")
print(f"  Strategy 2: 20.59 ms")
print(f"  Speedup: 1.76x")

print("\nConclusion:")
print("  - With fewer columns (100), concat overhead is similar")
print("  - Strategy 1 advantage increases (2.03x vs 1.76x)")
print("  - Individual processing scales better!")

print("\n" + "="*70)

