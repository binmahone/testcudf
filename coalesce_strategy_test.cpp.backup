#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>

/**
 * Performance comparison of two coalesce strategies:
 * Strategy 1: Coalesce each column individually (200 separate operations)
 * Strategy 2: Concatenate -> Coalesce once -> Slice back (3 operations)
 */

// Helper function to format bytes
std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " 
        << units[unit_index];
    return oss.str();
}

/**
 * Generate a column with random data and NULL values
 */
std::unique_ptr<cudf::column> generate_column(
    size_t num_rows,
    double null_probability,
    std::mt19937& gen,
    std::uniform_int_distribution<int32_t>& data_dist,
    std::uniform_real_distribution<double>& null_dist) {
    
    std::vector<int32_t> host_data(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        host_data[i] = data_dist(gen);
    }
    
    size_t mask_size = cudf::bitmask_allocation_size_bytes(num_rows) / 
                       sizeof(cudf::bitmask_type);
    std::vector<cudf::bitmask_type> null_mask(mask_size, 0);
    size_t null_count = 0;
    
    for (size_t i = 0; i < num_rows; ++i) {
        if (null_dist(gen) > null_probability) {
            cudf::set_bit_unsafe(null_mask.data(), i);
        } else {
            null_count++;
        }
    }
    
    rmm::device_buffer data_buffer(
        host_data.data(), 
        num_rows * sizeof(int32_t), 
        rmm::cuda_stream_default
    );
    rmm::device_buffer mask_buffer(
        null_mask.data(), 
        cudf::bitmask_allocation_size_bytes(num_rows),
        rmm::cuda_stream_default
    );
    
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        std::move(data_buffer),
        std::move(mask_buffer),
        null_count
    );
}

/**
 * Perform coalesce operation on a single column using copy_if_else
 */
std::unique_ptr<cudf::column> coalesce_column(
    const cudf::column_view& input_column) {
    
    size_t num_rows = input_column.size();
    
    auto zero_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        rmm::cuda_stream_default
    );
    
    cudaMemset(zero_column->mutable_view().data<int32_t>(), 
               0, 
               num_rows * sizeof(int32_t));
    
    auto is_valid_mask = cudf::is_valid(input_column);
    
    return cudf::copy_if_else(
        input_column,
        zero_column->view(),
        is_valid_mask->view()
    );
}

/**
 * Strategy 1: Coalesce each column individually
 */
std::vector<double> strategy1_individual_coalesce(
    const std::vector<std::unique_ptr<cudf::column>>& input_columns,
    std::vector<std::unique_ptr<cudf::column>>& result_columns,
    size_t num_warmup,
    size_t num_runs) {
    
    size_t num_columns = input_columns.size();
    
    std::cout << "Strategy 1: Individual column coalesce" << std::endl;
    std::cout << "  Processing " << num_columns 
              << " columns separately..." << std::endl;
    
    // Warm-up
    for (size_t i = 0; i < 3; ++i) {
        for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
            auto temp = coalesce_column(input_columns[col_idx]->view());
        }
        cudaDeviceSynchronize();
    }
    
    // Timed runs
    std::vector<double> run_times;
    run_times.reserve(num_runs);
    
    for (size_t run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
            auto temp = coalesce_column(input_columns[col_idx]->view());
            if (run == num_runs - 1) {
                result_columns.push_back(std::move(temp));
            }
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);
        run_times.push_back(duration.count() / 1000.0);
    }
    
    double avg_time = 0.0;
    for (double t : run_times) {
        avg_time += t;
    }
    avg_time /= run_times.size();
    
    double min_time = *std::min_element(run_times.begin(), 
                                         run_times.end());
    double max_time = *std::max_element(run_times.begin(), 
                                         run_times.end());
    
    std::cout << "  Avg time: " << avg_time << " ms" << std::endl;
    std::cout << "  Min time: " << min_time << " ms" << std::endl;
    std::cout << "  Max time: " << max_time << " ms" << std::endl;
    
    return avg_time;
}

/**
 * Strategy 2: Concat -> Coalesce -> Split (returns column_view, zero-copy)
 */
std::vector<double> strategy2_concat_coalesce_split_view(
    const std::vector<std::unique_ptr<cudf::column>>& input_columns,
    size_t num_warmup,
    size_t num_runs) {
    
    size_t num_columns = input_columns.size();
    size_t rows_per_column = input_columns[0]->size();
    
    std::cout << "Strategy 2: Concat->Coalesce->Split (column_view only)" 
              << std::endl;
    std::cout << "  Warmup runs: " << num_warmup << std::endl;
    std::cout << "  Timed runs: " << num_runs << std::endl;
    std::cout << std::endl;
    
    std::vector<cudf::column_view> column_views;
    column_views.reserve(num_columns);
    for (const auto& col : input_columns) {
        column_views.push_back(col->view());
    }
    
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(num_columns - 1);
    for (size_t i = 1; i < num_columns; ++i) {
        split_indices.push_back(i * rows_per_column);
    }
    
    std::vector<double> all_times;
    all_times.reserve(num_warmup + num_runs);
    
    std::vector<double> all_concat_times;
    std::vector<double> all_coalesce_times;
    std::vector<double> all_split_times;
    
    // Warm-up runs
    std::cout << "  Warmup iterations:" << std::endl;
    for (size_t i = 0; i < num_warmup; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto concat_col = cudf::concatenate(column_views);
        auto coalesced = coalesce_column(concat_col->view());
        auto split_result = cudf::split(coalesced->view(), split_indices);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;
        all_times.push_back(time_ms);
        
        std::cout << "    Warmup " << (i+1) << ": " 
                  << std::fixed << std::setprecision(3) 
                  << time_ms << " ms" << std::endl;
    }
    
    // Timed runs
    std::cout << std::endl;
    std::cout << "  Timed iterations:" << std::endl;
    std::vector<double> timed_runs;
    timed_runs.reserve(num_runs);
    
    for (size_t run = 0; run < num_runs; ++run) {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        auto start_concat = std::chrono::high_resolution_clock::now();
        auto concatenated_column = cudf::concatenate(column_views);
        cudaDeviceSynchronize();
        auto end_concat = std::chrono::high_resolution_clock::now();
        
        auto start_coalesce = std::chrono::high_resolution_clock::now();
        auto coalesced_column = coalesce_column(
            concatenated_column->view()
        );
        cudaDeviceSynchronize();
        auto end_coalesce = std::chrono::high_resolution_clock::now();
        
        auto start_split = std::chrono::high_resolution_clock::now();
        auto split_result = cudf::split(coalesced_column->view(), 
                                         split_indices);
        cudaDeviceSynchronize();
        auto end_split = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        auto concat_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_concat - start_concat);
        auto coalesce_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_coalesce - start_coalesce);
        auto split_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_split - start_split);
        auto total_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_total - start_total);
        
        double concat_ms = concat_dur.count() / 1000.0;
        double coalesce_ms = coalesce_dur.count() / 1000.0;
        double split_ms = split_dur.count() / 1000.0;
        double total_ms = total_dur.count() / 1000.0;
        
        all_concat_times.push_back(concat_ms);
        all_coalesce_times.push_back(coalesce_ms);
        all_split_times.push_back(split_ms);
        all_times.push_back(total_ms);
        timed_runs.push_back(total_ms);
        
        std::cout << "    Run " << (run+1) << ": " 
                  << std::fixed << std::setprecision(3) 
                  << total_ms << " ms  (concat: " << concat_ms 
                  << ", coalesce: " << coalesce_ms 
                  << ", split: " << split_ms << ")" << std::endl;
    }
    
    // Calculate statistics
    auto calc_avg = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double val : v) sum += val;
        return sum / v.size();
    };
    
    auto calc_stddev = [&calc_avg](const std::vector<double>& v) {
        double avg = calc_avg(v);
        double variance = 0.0;
        for (double val : v) {
            variance += (val - avg) * (val - avg);
        }
        variance /= v.size();
        return std::sqrt(variance);
    };
    
    double avg_total = calc_avg(timed_runs);
    double stddev_total = calc_stddev(timed_runs);
    double min_time = *std::min_element(timed_runs.begin(), 
                                         timed_runs.end());
    double max_time = *std::max_element(timed_runs.begin(), 
                                         timed_runs.end());
    
    std::cout << std::endl;
    std::cout << "  Statistics (timed runs only):" << std::endl;
    std::cout << "    Average:     " << std::fixed << std::setprecision(3) 
              << avg_total << " ms" << std::endl;
    std::cout << "    Min:         " << min_time << " ms" << std::endl;
    std::cout << "    Max:         " << max_time << " ms" << std::endl;
    std::cout << "    Std Dev:     " << stddev_total << " ms" << std::endl;
    std::cout << "    CV:          " << std::fixed << std::setprecision(2) 
              << (stddev_total / avg_total * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "    Avg Concat:  " << calc_avg(all_concat_times) << " ms" 
              << std::endl;
    std::cout << "    Avg Coalesce:" << calc_avg(all_coalesce_times) << " ms" 
              << std::endl;
    std::cout << "    Avg Split:   " << calc_avg(all_split_times) << " ms" 
              << std::endl;
    
    return all_times;
}

/**
 * Strategy 3: Concat -> Coalesce -> Split (materialize to column, with copy)
 */
double strategy3_concat_coalesce_split_column(
    const std::vector<std::unique_ptr<cudf::column>>& input_columns,
    std::vector<std::unique_ptr<cudf::column>>& result_columns,
    size_t num_runs) {
    
    size_t num_columns = input_columns.size();
    size_t rows_per_column = input_columns[0]->size();
    
    std::cout << "Strategy 3: Concat->Coalesce->Split (materialize column)" 
              << std::endl;
    
    std::vector<cudf::column_view> column_views;
    column_views.reserve(num_columns);
    for (const auto& col : input_columns) {
        column_views.push_back(col->view());
    }
    
    // Prepare split indices
    std::vector<cudf::size_type> split_indices;
    split_indices.reserve(num_columns - 1);
    for (size_t i = 1; i < num_columns; ++i) {
        split_indices.push_back(i * rows_per_column);
    }
    
    // Warm-up
    for (size_t i = 0; i < 3; ++i) {
        auto concat_col = cudf::concatenate(column_views);
        auto coalesced = coalesce_column(concat_col->view());
        cudaDeviceSynchronize();
    }
    
    // Timed runs
    std::vector<double> run_times;
    std::vector<double> concat_times;
    std::vector<double> coalesce_times;
    std::vector<double> split_times;
    std::vector<double> materialize_times;
    
    run_times.reserve(num_runs);
    concat_times.reserve(num_runs);
    coalesce_times.reserve(num_runs);
    split_times.reserve(num_runs);
    materialize_times.reserve(num_runs);
    
    for (size_t run = 0; run < num_runs; ++run) {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Step 1: Concatenate
        auto start_concat = std::chrono::high_resolution_clock::now();
        auto concatenated_column = cudf::concatenate(column_views);
        cudaDeviceSynchronize();
        auto end_concat = std::chrono::high_resolution_clock::now();
        
        // Step 2: Coalesce
        auto start_coalesce = std::chrono::high_resolution_clock::now();
        auto coalesced_column = coalesce_column(
            concatenated_column->view()
        );
        cudaDeviceSynchronize();
        auto end_coalesce = std::chrono::high_resolution_clock::now();
        
        // Step 3: Split (get views)
        auto start_split = std::chrono::high_resolution_clock::now();
        auto split_result = cudf::split(coalesced_column->view(), 
                                         split_indices);
        cudaDeviceSynchronize();
        auto end_split = std::chrono::high_resolution_clock::now();
        
        // Step 4: Materialize views to columns
        auto start_materialize = std::chrono::high_resolution_clock::now();
        if (run == num_runs - 1) {
            for (auto& col_view : split_result) {
                result_columns.push_back(
                    std::make_unique<cudf::column>(col_view)
                );
            }
        } else {
            // Still need to materialize for accurate timing
            std::vector<std::unique_ptr<cudf::column>> temp_columns;
            for (auto& col_view : split_result) {
                temp_columns.push_back(
                    std::make_unique<cudf::column>(col_view)
                );
            }
        }
        cudaDeviceSynchronize();
        auto end_materialize = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        auto concat_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_concat - start_concat);
        auto coalesce_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_coalesce - start_coalesce);
        auto split_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_split - start_split);
        auto materialize_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_materialize - start_materialize);
        auto total_dur = std::chrono::duration_cast<
            std::chrono::microseconds>(end_total - start_total);
        
        concat_times.push_back(concat_dur.count() / 1000.0);
        coalesce_times.push_back(coalesce_dur.count() / 1000.0);
        split_times.push_back(split_dur.count() / 1000.0);
        materialize_times.push_back(materialize_dur.count() / 1000.0);
        run_times.push_back(total_dur.count() / 1000.0);
    }
    
    auto calc_avg = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double val : v) sum += val;
        return sum / v.size();
    };
    
    double avg_concat = calc_avg(concat_times);
    double avg_coalesce = calc_avg(coalesce_times);
    double avg_split = calc_avg(split_times);
    double avg_materialize = calc_avg(materialize_times);
    double avg_total = calc_avg(run_times);
    
    std::cout << "  Breakdown (average):" << std::endl;
    std::cout << "    Concatenate:   " << avg_concat << " ms" << std::endl;
    std::cout << "    Coalesce:      " << avg_coalesce << " ms" 
              << std::endl;
    std::cout << "    Split(view):   " << avg_split << " ms (zero-copy)" 
              << std::endl;
    std::cout << "    Materialize:   " << avg_materialize 
              << " ms (GPU copy)" << std::endl;
    std::cout << "    Total:         " << avg_total << " ms" << std::endl;
    
    return avg_total;
}

int main() {
    // Initialize CUDA memory resource
    rmm::mr::cuda_memory_resource mr;
    rmm::mr::set_current_device_resource(&mr);
    
    constexpr size_t TOTAL_DATA_SIZE = 1073741824;  // 1GB
    constexpr size_t NUM_COLUMNS = 200;
    constexpr double NULL_PROBABILITY = 0.2;         // 20% NULL
    constexpr size_t NUM_WARMUP = 5;
    constexpr size_t NUM_RUNS = 20;
    
    size_t total_rows = TOTAL_DATA_SIZE / sizeof(int32_t);
    size_t rows_per_column = total_rows / NUM_COLUMNS;
    
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << "CUDF Coalesce Strategy Comparison" << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Total data size: " 
              << format_bytes(TOTAL_DATA_SIZE) << std::endl;
    std::cout << "  Number of columns: " << NUM_COLUMNS << std::endl;
    std::cout << "  Rows per column: " << rows_per_column << std::endl;
    std::cout << "  NULL probability: " 
              << (NULL_PROBABILITY * 100) << "%" << std::endl;
    std::cout << "  Warmup runs: " << NUM_WARMUP << std::endl;
    std::cout << "  Timed runs: " << NUM_RUNS << std::endl;
    std::cout << std::endl;
    
    // Generate input data (simulating table1 already on GPU)
    std::cout << "Generating input table (simulating table already on GPU)..." 
              << std::endl;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> data_dist(1, 1000);
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.reserve(NUM_COLUMNS);
    
    for (size_t col_idx = 0; col_idx < NUM_COLUMNS; ++col_idx) {
        input_columns.push_back(
            generate_column(rows_per_column, NULL_PROBABILITY, 
                           gen, data_dist, null_dist)
        );
    }
    
    cudaDeviceSynchronize();
    std::cout << "Input table generated (Table1)." << std::endl;
    std::cout << std::endl;
    
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Strategy 1: Individual coalesce
    std::vector<std::unique_ptr<cudf::column>> result_columns_s1;
    result_columns_s1.reserve(NUM_COLUMNS);
    
    std::vector<double> times_strategy1 = strategy1_individual_coalesce(
        input_columns, 
        result_columns_s1,
        NUM_WARMUP,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "Strategy 1 completed." << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Strategy 2: Concat -> Coalesce -> Split (view only, zero-copy)
    std::vector<double> times_strategy2 = strategy2_concat_coalesce_split_view(
        input_columns,
        NUM_WARMUP,
        NUM_RUNS
    );
    
    std::cout << std::endl;
    std::cout << "Strategy 2 completed." << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << "=== Final Comparison ===" << std::endl;
    std::cout << "========================================"
              << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Calculate averages from timed runs only (skip warmup)
    auto calc_avg_timed = [NUM_WARMUP](const std::vector<double>& times) {
        double sum = 0.0;
        for (size_t i = NUM_WARMUP; i < times.size(); ++i) {
            sum += times[i];
        }
        return sum / (times.size() - NUM_WARMUP);
    };
    
    double avg1 = calc_avg_timed(times_strategy1);
    double avg2 = calc_avg_timed(times_strategy2);
    
    std::cout << std::left << std::setw(50) << "Strategy"
              << std::right << std::setw(18) << "Avg Time (ms)"
              << std::setw(18) << "Speedup"
              << std::endl;
    std::cout << std::string(86, '-') << std::endl;
    
    std::cout << std::left << std::setw(50) 
              << "1. Individual coalesce (200 ops)"
              << std::right << std::setw(18) << std::fixed 
              << std::setprecision(3) << avg1
              << std::setw(18) << "1.00x"
              << std::endl;
    
    double speedup = avg1 / avg2;
    std::cout << std::left << std::setw(50) 
              << "2. Concat->Coalesce->Split (view, zero-copy)"
              << std::right << std::setw(18) << std::fixed 
              << std::setprecision(3) << avg2
              << std::setw(18) << std::fixed << std::setprecision(2) 
              << speedup << "x"
              << std::endl;
    
    std::cout << std::endl;
    std::cout << "Performance gain: " 
              << std::fixed << std::setprecision(2) 
              << speedup << "x faster!" << std::endl;
    std::cout << "Time saved: " 
              << std::fixed << std::setprecision(3)
              << (avg1 - avg2) << " ms (" 
              << std::fixed << std::setprecision(1)
              << ((avg1 - avg2) / avg1 * 100.0) << "%)" << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Data for Plotting ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy 1 (Individual coalesce):" << std::endl;
    std::cout << "  All times (warmup + timed): ";
    for (size_t i = 0; i < times_strategy1.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) 
                  << times_strategy1[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Strategy 2 (Concat->Coalesce->Split view):" << std::endl;
    std::cout << "  All times (warmup + timed): ";
    for (size_t i = 0; i < times_strategy2.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) 
                  << times_strategy2[i];
    }
    std::cout << std::endl;
    
    std::cout << std::endl 
              << "=== Test completed successfully ===" << std::endl;
    
    return 0;
}

