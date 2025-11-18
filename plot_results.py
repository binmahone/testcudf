#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from test output (Latest run)
strategy1_times = [105.842, 10.063, 10.084, 10.237, 10.305, 10.096, 10.088, 
                   10.140, 10.117, 10.196, 10.125, 10.041, 10.072, 10.121, 
                   10.219, 10.079, 9.986, 9.888, 9.904, 9.937, 9.913, 9.880, 
                   9.899, 9.911, 10.002]

strategy2_times = [22.505, 20.362, 20.456, 20.454, 20.387, 20.423, 20.474, 
                   20.361, 20.439, 20.386, 20.374, 20.466, 20.445, 20.359, 
                   20.475, 20.476, 20.589, 20.453, 20.356, 20.510, 20.369, 
                   20.491, 20.676, 20.472, 20.501]

concat_times = [5.949, 4.310, 4.302, 4.300, 4.296, 4.293, 4.300, 4.296, 4.299, 
                4.296, 4.296, 4.297, 4.298, 4.302, 4.297, 4.298, 4.291, 4.297, 
                4.297, 4.300, 4.294, 4.292, 4.299, 4.293, 4.309]

coalesce_times = [16.061, 15.931, 16.036, 16.036, 15.970, 16.011, 16.053, 
                  15.948, 16.018, 15.970, 15.958, 16.050, 16.031, 15.935, 
                  16.059, 16.062, 16.179, 16.038, 15.939, 16.092, 15.957, 
                  16.074, 16.257, 16.051, 16.071]

split_times = [0.493, 0.121, 0.116, 0.117, 0.120, 0.118, 0.120, 0.116, 0.121, 
               0.119, 0.118, 0.118, 0.115, 0.121, 0.118, 0.115, 0.118, 0.117, 
               0.119, 0.118, 0.116, 0.124, 0.118, 0.126, 0.120]

num_warmup = 5
num_runs = 20

# Split into warmup and timed runs
s1_warmup = strategy1_times[:num_warmup]
s1_timed = strategy1_times[num_warmup:]

s2_warmup = strategy2_times[:num_warmup]
s2_timed = strategy2_times[num_warmup:]

concat_warmup = concat_times[:num_warmup]
concat_timed = concat_times[num_warmup:]

coalesce_warmup = coalesce_times[:num_warmup]
coalesce_timed = coalesce_times[num_warmup:]

split_warmup = split_times[:num_warmup]
split_timed = split_times[num_warmup:]

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 13))
fig.suptitle('CUDF Coalesce Performance Comparison (100 cols, 1GB)', 
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

# Plot 5: Strategy 2 Breakdown - Stacked Area
ax5 = axes[2, 0]
x_timed = list(range(1, num_runs + 1))
ax5.stackplot(x_timed, concat_timed, coalesce_timed, split_timed,
              labels=['Concat', 'Coalesce', 'Split'],
              colors=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.7)
ax5.set_xlabel('Run Number')
ax5.set_ylabel('Time (ms)')
ax5.set_title('Strategy 2 Time Breakdown (Stacked)')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# Plot 6: Strategy 2 Breakdown - Pie Chart
ax6 = axes[2, 1]
avg_concat = np.mean(concat_timed)
avg_coalesce = np.mean(coalesce_timed)
avg_split = np.mean(split_timed)
breakdown_values = [avg_concat, avg_coalesce, avg_split]
breakdown_labels = [f'Concat\n{avg_concat:.2f}ms\n({avg_concat/np.sum(breakdown_values)*100:.1f}%)',
                    f'Coalesce\n{avg_coalesce:.2f}ms\n({avg_coalesce/np.sum(breakdown_values)*100:.1f}%)',
                    f'Split\n{avg_split:.2f}ms\n({avg_split/np.sum(breakdown_values)*100:.1f}%)']
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax6.pie(breakdown_values, labels=breakdown_labels, colors=colors, autopct='',
        startangle=90, textprops={'fontsize': 10})
ax6.set_title('Strategy 2 Time Breakdown (Average)')
ax6.axis('equal')

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

print("\nStrategy 2 Time Breakdown (Average):")
avg_concat = np.mean(concat_timed)
avg_coalesce = np.mean(coalesce_timed)
avg_split = np.mean(split_timed)
avg_total = np.mean(s2_timed)
print(f"  Concat:   {avg_concat:.3f} ms ({avg_concat/avg_total*100:.1f}%)")
print(f"  Coalesce: {avg_coalesce:.3f} ms ({avg_coalesce/avg_total*100:.1f}%)")
print(f"  Split:    {avg_split:.3f} ms ({avg_split/avg_total*100:.1f}%)")

if np.mean(s1_timed) < np.mean(s2_timed):
    print(f"\nSpeedup: Strategy 1 is {np.mean(s2_timed)/np.mean(s1_timed):.2f}x faster")
else:
    print(f"\nSpeedup: Strategy 2 is {np.mean(s1_timed)/np.mean(s2_timed):.2f}x faster")

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

