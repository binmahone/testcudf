#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from test output
strategy1_times = [190.242, 123.912, 124.523, 119.995, 120.202, 120.766, 
                   122.390, 119.977, 120.093, 119.840, 120.433, 119.854, 
                   120.480, 121.257, 120.317, 122.769, 120.208, 120.158, 
                   120.884, 120.186, 121.047, 120.784, 119.602, 120.584, 
                   120.104]

strategy2_times = [19.758, 25.413, 26.491, 25.428, 26.739, 25.430, 26.597, 
                   25.369, 26.337, 25.364, 26.481, 25.366, 26.484, 25.354, 
                   26.584, 25.375, 63.341, 54.470, 57.007, 57.038, 50.217, 
                   57.130, 39.991, 70.047, 54.523]

num_warmup = 5
num_runs = 20

# Split into warmup and timed runs
s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CUDF Coalesce Performance Comparison (200 cols, 1GB)', 
             fontsize=14, fontweight='bold')

# Plot 1: Full timeline (warmup + timed)
ax1 = axes[0, 0]
x_all = list(range(1, len(strategy1_times) + 1))
ax1.plot(x_all, strategy1_times, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6)
ax1.plot(x_all, strategy2_times, 's-', label='Strategy 2 (Concat-Coalesce-Split)',
         linewidth=2, markersize=6)
ax1.axvline(x=num_warmup, color='red', linestyle='--', alpha=0.5, 
            label='Warmup end')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Time (ms)')
ax1.set_title('All Iterations (Warmup + Timed)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Timed runs only
ax2 = axes[0, 1]
x_timed = list(range(1, num_runs + 1))
ax2.plot(x_timed, s1_timed, 'o-', label='Strategy 1 (Individual)', 
         linewidth=2, markersize=6)
ax2.plot(x_timed, s2_timed, 's-', label='Strategy 2 (Concat-Coalesce-Split)',
         linewidth=2, markersize=6)
ax2.set_xlabel('Run Number')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Timed Runs Only')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Highlight the anomaly in Strategy 2
ax2.axhline(y=np.mean(s2_timed), color='orange', linestyle=':', alpha=0.7,
            label='S2 Mean')
ax2.axhline(y=np.mean(s2_timed) + 2*np.std(s2_timed), color='red', 
            linestyle=':', alpha=0.5)

# Plot 3: Box plot comparison
ax3 = axes[1, 0]
data_to_plot = [s1_timed, s2_timed]
bp = ax3.boxplot(data_to_plot, labels=['Strategy 1', 'Strategy 2'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
ax3.set_ylabel('Time (ms)')
ax3.set_title('Distribution Comparison (Timed Runs)')
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats1 = f'Mean: {np.mean(s1_timed):.2f}ms\nStd: {np.std(s1_timed):.2f}ms\nCV: {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%'
stats2 = f'Mean: {np.mean(s2_timed):.2f}ms\nStd: {np.std(s2_timed):.2f}ms\nCV: {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%'
ax3.text(1, max(s1_timed)*0.95, stats1, fontsize=9, bbox=dict(boxstyle='round', 
         facecolor='lightblue', alpha=0.7))
ax3.text(2, max(s2_timed)*0.95, stats2, fontsize=9, bbox=dict(boxstyle='round',
         facecolor='lightgreen', alpha=0.7))

# Plot 4: Histogram
ax4 = axes[1, 1]
ax4.hist(s1_timed, bins=15, alpha=0.6, label='Strategy 1', color='blue', 
         edgecolor='black')
ax4.hist(s2_timed, bins=15, alpha=0.6, label='Strategy 2', color='green',
         edgecolor='black')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Frequency')
ax4.set_title('Time Distribution Histogram')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/hongbin/code/testCUDF/coalesce_comparison.png', dpi=150)
print("\nPlot saved to: /home/hongbin/code/testCUDF/coalesce_comparison.png")

# Print detailed analysis
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)
print("\nStrategy 1 (Individual coalesce):")
print(f"  Mean:   {np.mean(s1_timed):.3f} ms")
print(f"  Median: {np.median(s1_timed):.3f} ms")
print(f"  Min:    {np.min(s1_timed):.3f} ms")
print(f"  Max:    {np.max(s1_timed):.3f} ms")
print(f"  StdDev: {np.std(s1_timed):.3f} ms")
print(f"  CV:     {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%")

print("\nStrategy 2 (Concat->Coalesce->Split view):")
print(f"  Mean:   {np.mean(s2_timed):.3f} ms")
print(f"  Median: {np.median(s2_timed):.3f} ms")
print(f"  Min:    {np.min(s2_timed):.3f} ms")
print(f"  Max:    {np.max(s2_timed):.3f} ms")
print(f"  StdDev: {np.std(s2_timed):.3f} ms")
print(f"  CV:     {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%")

print(f"\nSpeedup: {np.mean(s1_timed)/np.mean(s2_timed):.2f}x")

# Identify anomalies in Strategy 2
mean_s2 = np.mean(s2_timed)
std_s2 = np.std(s2_timed)
threshold = mean_s2 + 2 * std_s2

print("\n" + "="*60)
print("ANOMALY DETECTION (Strategy 2)")
print("="*60)
print(f"Mean: {mean_s2:.3f} ms")
print(f"Threshold (mean + 2*std): {threshold:.3f} ms")
print("\nRuns exceeding threshold:")
for i, t in enumerate(s2_timed):
    if t > threshold:
        print(f"  Run {i+1}: {t:.3f} ms (deviation: +{((t-mean_s2)/mean_s2*100):.1f}%)")

# Find stable range
stable_s2 = [t for t in s2_timed if t < threshold]
if stable_s2:
    print(f"\nStable runs (n={len(stable_s2)}):")
    print(f"  Mean: {np.mean(stable_s2):.3f} ms")
    print(f"  StdDev: {np.std(stable_s2):.3f} ms")
    print(f"  CV: {np.std(stable_s2)/np.mean(stable_s2)*100:.1f}%")

print("\n" + "="*60)

