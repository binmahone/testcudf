#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from test output - 100 columns
data_100cols = {
    'num_columns': 100,
    'strategy1_times': [97.120, 10.100, 10.068, 10.069, 10.064, 10.055, 10.048, 
                        10.156, 10.054, 10.058, 10.067, 10.069, 10.093, 10.062, 
                        10.022, 10.032, 10.035, 10.146, 10.056, 10.030, 10.045, 
                        10.055, 10.018, 10.115, 10.029],
    'strategy2_times': [22.529, 20.432, 20.470, 20.438, 20.455, 20.348, 20.353, 
                        20.384, 20.450, 20.378, 20.492, 20.356, 20.427, 20.438, 
                        20.440, 20.426, 20.457, 20.366, 20.434, 20.390, 20.463, 
                        20.370, 20.532, 20.369, 20.458],
    'concat_times': [5.976, 4.308, 4.291, 4.342, 4.293, 4.291, 4.291, 4.294, 
                     4.286, 4.292, 4.292, 4.293, 4.283, 4.286, 4.287, 4.287, 
                     4.296, 4.288, 4.288, 4.295, 4.289, 4.294, 4.289, 4.297, 4.288],
    'coalesce_times': [16.052, 16.002, 16.058, 15.977, 16.043, 15.935, 15.945, 
                       15.972, 16.046, 15.969, 16.076, 15.947, 16.024, 16.035, 
                       16.035, 16.022, 16.044, 15.959, 16.025, 15.977, 16.056, 
                       15.959, 16.124, 15.955, 16.053],
    'split_times': [0.500, 0.121, 0.121, 0.118, 0.119, 0.121, 0.117, 0.117, 
                    0.117, 0.116, 0.123, 0.115, 0.118, 0.117, 0.117, 0.117, 
                    0.116, 0.117, 0.119, 0.118, 0.118, 0.116, 0.118, 0.117, 0.116]
}

# Data from test output - 400 columns
data_400cols = {
    'num_columns': 400,
    'strategy1_times': [108.335, 17.902, 17.646, 17.654, 17.633, 17.615, 17.772, 
                        17.671, 17.649, 17.566, 17.375, 17.023, 17.279, 17.304, 
                        17.291, 17.308, 17.316, 17.130, 17.502, 17.285, 17.363, 
                        17.291, 17.310, 17.250, 17.475],
    'strategy2_times': [21.585, 20.862, 20.823, 20.807, 20.872, 20.750, 20.990, 
                        20.951, 20.742, 20.861, 20.788, 20.901, 20.774, 20.841, 
                        20.765, 20.860, 20.835, 20.759, 20.858, 20.852, 20.841, 
                        20.868, 20.817, 20.846, 21.117],
    'concat_times': [5.013, 4.700, 4.680, 4.682, 4.685, 4.688, 4.682, 4.684, 
                     4.680, 4.682, 4.693, 4.682, 4.685, 4.683, 4.689, 4.685, 
                     4.684, 4.685, 4.688, 4.690, 4.686, 4.686, 4.684, 4.688, 4.692],
    'coalesce_times': [16.066, 16.034, 16.018, 16.000, 16.061, 15.935, 16.182, 
                       16.142, 15.931, 16.043, 15.969, 16.094, 15.961, 16.028, 
                       15.950, 16.050, 16.027, 15.949, 16.044, 16.036, 16.027, 
                       16.051, 16.010, 16.033, 16.301],
    'split_times': [0.506, 0.126, 0.124, 0.123, 0.125, 0.127, 0.125, 0.124, 
                    0.130, 0.135, 0.125, 0.124, 0.126, 0.129, 0.126, 0.123, 
                    0.123, 0.124, 0.125, 0.125, 0.127, 0.129, 0.122, 0.124, 0.123]
}

num_warmup = 5
num_runs = 20

def plot_comparison(data, output_file):
    """Generate comparison plots for given data"""
    num_cols = data['num_columns']
    
    # Split into warmup and timed runs
    s1_warmup = data['strategy1_times'][:num_warmup]
    s1_timed = data['strategy1_times'][num_warmup:]
    
    s2_warmup = data['strategy2_times'][:num_warmup]
    s2_timed = data['strategy2_times'][num_warmup:]
    
    concat_timed = data['concat_times'][num_warmup:]
    coalesce_timed = data['coalesce_times'][num_warmup:]
    split_timed = data['split_times'][num_warmup:]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(f'CUDF Coalesce Performance Comparison ({num_cols} cols, 1GB)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Full timeline (warmup + timed)
    ax1 = axes[0, 0]
    x_all = list(range(1, len(data['strategy1_times']) + 1))
    ax1.plot(x_all, data['strategy1_times'], 'o-', 
             label='Strategy 1 (Individual)', linewidth=2, markersize=6)
    ax1.plot(x_all, data['strategy2_times'], 's-', 
             label='Strategy 2 (Concat-Coalesce-Split)', linewidth=2, markersize=6)
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
    stats1 = (f'Mean: {np.mean(s1_timed):.2f}ms\n'
              f'Std: {np.std(s1_timed):.2f}ms\n'
              f'CV: {np.std(s1_timed)/np.mean(s1_timed)*100:.1f}%')
    stats2 = (f'Mean: {np.mean(s2_timed):.2f}ms\n'
              f'Std: {np.std(s2_timed):.2f}ms\n'
              f'CV: {np.std(s2_timed)/np.mean(s2_timed)*100:.1f}%')
    ax3.text(1, max(s1_timed)*0.95, stats1, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.text(2, max(s2_timed)*0.95, stats2, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
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
    total = np.sum(breakdown_values)
    breakdown_labels = [
        f'Concat\n{avg_concat:.2f}ms\n({avg_concat/total*100:.1f}%)',
        f'Coalesce\n{avg_coalesce:.2f}ms\n({avg_coalesce/total*100:.1f}%)',
        f'Split\n{avg_split:.2f}ms\n({avg_split/total*100:.1f}%)'
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax6.pie(breakdown_values, labels=breakdown_labels, colors=colors, 
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax6.set_title('Strategy 2 Time Breakdown (Average)')
    ax6.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    
    # Print detailed analysis
    print("\n" + "="*60)
    print(f"DETAILED ANALYSIS - {num_cols} Columns")
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
    avg_total = np.mean(s2_timed)
    print(f"  Concat:   {avg_concat:.3f} ms ({avg_concat/avg_total*100:.1f}%)")
    print(f"  Coalesce: {avg_coalesce:.3f} ms ({avg_coalesce/avg_total*100:.1f}%)")
    print(f"  Split:    {avg_split:.3f} ms ({avg_split/avg_total*100:.1f}%)")
    
    if np.mean(s1_timed) < np.mean(s2_timed):
        speedup = np.mean(s2_timed) / np.mean(s1_timed)
        print(f"\nSpeedup: Strategy 1 is {speedup:.2f}x faster")
    else:
        speedup = np.mean(s1_timed) / np.mean(s2_timed)
        print(f"\nSpeedup: Strategy 2 is {speedup:.2f}x faster")
    
    print("="*60)

# Generate plots for both configurations
print("\n" + "="*70)
print("Generating comparison plots for different column counts")
print("="*70)

plot_comparison(data_100cols, 
                '/home/hongbin/code/testCUDF/coalesce_comparison_100cols.png')
plot_comparison(data_400cols, 
                '/home/hongbin/code/testCUDF/coalesce_comparison_400cols.png')

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"\n100 Columns:")
print(f"  Strategy 1 Avg: {np.mean(data_100cols['strategy1_times'][num_warmup:]):.3f} ms")
print(f"  Strategy 2 Avg: {np.mean(data_100cols['strategy2_times'][num_warmup:]):.3f} ms")
print(f"  Strategy 1 is {np.mean(data_100cols['strategy2_times'][num_warmup:])/np.mean(data_100cols['strategy1_times'][num_warmup:]):.2f}x faster")

print(f"\n400 Columns:")
print(f"  Strategy 1 Avg: {np.mean(data_400cols['strategy1_times'][num_warmup:]):.3f} ms")
print(f"  Strategy 2 Avg: {np.mean(data_400cols['strategy2_times'][num_warmup:]):.3f} ms")
s1_avg_400 = np.mean(data_400cols['strategy1_times'][num_warmup:])
s2_avg_400 = np.mean(data_400cols['strategy2_times'][num_warmup:])
if s1_avg_400 < s2_avg_400:
    print(f"  Strategy 1 is {s2_avg_400/s1_avg_400:.2f}x faster")
else:
    print(f"  Strategy 2 is {s1_avg_400/s2_avg_400:.2f}x faster")

print("\nConclusion:")
print("  As the number of columns increases, the performance gap narrows.")
print("  At 100 columns, concat overhead dominates.")
print("  At 400 columns, multiple kernel launches become more expensive.")
print("="*70)

