#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data with pool_memory_resource - cast(coalesce(x, 0) as double)
strategy1_times = [105.261, 11.674, 11.684, 11.746, 11.717, 11.699, 11.662, 
                   11.669, 11.701, 11.706, 11.670, 11.679, 11.697, 11.698, 
                   11.695, 11.674, 11.670, 11.711, 11.676, 11.697, 11.639, 
                   11.669, 11.674, 11.706, 11.674]

strategy2_times = [22.898, 20.639, 20.535, 20.574, 20.626, 20.545, 20.602, 
                   20.622, 20.536, 20.534, 20.562, 20.529, 20.564, 20.710, 
                   20.602, 20.552, 20.535, 20.607, 20.599, 20.640, 20.520, 
                   20.527, 20.550, 20.818, 20.540]

num_warmup = 5
num_runs = 20

# Split into warmup and timed runs
s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CUDF cast(coalesce(x,0) as double) Performance\n' +
             '(200 cols, 1GB->2GB, with RMM Pool)', 
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
ax1.set_title('All Iterations (Warmup + Timed)', fontsize=12, 
              fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Timed runs only
ax2 = axes[0, 1]
x_timed = list(range(1, num_runs + 1))
ax2.plot(x_timed, s1_timed, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6, color='blue')
ax2.plot(x_timed, s2_timed, 's-', 
         label='Strategy 2 (Concat-Cast-Split)',
         linewidth=2, markersize=6, color='green')
ax2.axhline(y=np.mean(s1_timed), color='blue', linestyle=':', 
            alpha=0.5, linewidth=1.5)
ax2.axhline(y=np.mean(s2_timed), color='green', linestyle=':', 
            alpha=0.5, linewidth=1.5)
ax2.set_xlabel('Run Number', fontsize=11)
ax2.set_ylabel('Time (ms)', fontsize=11)
ax2.set_title('Timed Runs - Excellent Stability!', fontsize=12, 
              fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add text annotations
ax2.text(10, np.mean(s1_timed) - 0.15, 
         f'S1: {np.mean(s1_timed):.2f}ms (CV={np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%)', 
         color='blue', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(10, np.mean(s2_timed) + 0.25, 
         f'S2: {np.mean(s2_timed):.2f}ms (CV={np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%)', 
         color='green', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Box plot comparison
ax3 = axes[1, 0]
data_to_plot = [s1_timed, s2_timed]
bp = ax3.boxplot(data_to_plot, 
                  tick_labels=['Strategy 1\nIndividual\ncast(coalesce)', 
                               'Strategy 2\nConcat-Cast-Split'],
                  patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')

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
ax3.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics
stats_text = (f'Strategy 1: {np.mean(s1_timed):.2f} ± {np.std(s1_timed):.3f}ms\n'
              f'Strategy 2: {np.mean(s2_timed):.2f} ± {np.std(s2_timed):.3f}ms\n'
              f'Speedup: {np.mean(s2_timed)/np.mean(s1_timed):.2f}x (S1 faster)')
ax3.text(0.5, 0.98, stats_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Bar chart comparison
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

# Add value labels on bars
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
plt.savefig('/home/hongbin/code/testCUDF/cast_coalesce_comparison.png', 
            dpi=150, bbox_inches='tight')
print("\nPlot saved to: /home/hongbin/code/testCUDF/cast_coalesce_comparison.png")

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED ANALYSIS - cast(coalesce(x,0) as double)")
print("="*70)
print("\nStrategy 1 (Individual cast(coalesce) - 200 ops):")
print(f"  Mean:   {np.mean(s1_timed):.3f} ms")
print(f"  Median: {np.median(s1_timed):.3f} ms")
print(f"  Min:    {np.min(s1_timed):.3f} ms")
print(f"  Max:    {np.max(s1_timed):.3f} ms")
print(f"  StdDev: {np.std(s1_timed):.3f} ms")
print(f"  CV:     {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%")
print(f"  Range:  {np.max(s1_timed) - np.min(s1_timed):.3f} ms")

print("\nStrategy 2 (Concat->Cast(Coalesce)->Split view):")
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

ratio = np.mean(s2_timed) / np.mean(s1_timed)
print(f"\nStrategy 1 is {ratio:.2f}x FASTER! ⭐")
print(f"Time saved: {np.mean(s2_timed) - np.mean(s1_timed):.3f} ms")
print(f"Percentage: {(np.mean(s2_timed) - np.mean(s1_timed))/np.mean(s2_timed)*100:.1f}%")

print("\n" + "="*70)
print("STABILITY COMPARISON")
print("="*70)
print(f"\nStrategy 1 CV: {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}% ✓ Excellent!")
print(f"Strategy 2 CV: {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}% ✓ Excellent!")
print("\nBoth strategies show excellent stability with RMM Pool!")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nFor cast(coalesce(x, 0) as double) on 200 columns (1GB input):")
print(f"  ✓ Use Strategy 1 (Individual) with RMM Pool")
print(f"  ✓ Performance: {np.mean(s1_timed):.2f}ms, stable (CV={np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%)")
print(f"  ✓ {ratio:.2f}x faster than concat-split approach")
print(f"  ✓ Concat/split overhead (~8.9ms) not worth it")

print("\n" + "="*70)

