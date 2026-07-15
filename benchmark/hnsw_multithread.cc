/**
 * @file hnsw_multithread.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief HNSW Multi-threaded Query Benchmark
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 */

// Define this to disable setEf() calls inside rangeFilteringSearchOutBound
// (we set it once before launching threads instead)
#define HNSW_MULTITHREAD_BENCHMARK

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "utils.h"

#include "../../src/baselines/knn_first_hnsw.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

// ==================== Thread-safe Result Accumulator ====================

struct ThreadLocalResult {
  std::map<int, std::pair<float, float>> result_recorder;
  std::map<int, float> comparison_recorder;
  int query_count;

  ThreadLocalResult() : query_count(0) {}

  void merge(const ThreadLocalResult& other) {
    for (const auto& [range, val] : other.result_recorder) {
      result_recorder[range].first += val.first;
      result_recorder[range].second += val.second;
    }
    for (const auto& [range, comps] : other.comparison_recorder) {
      comparison_recorder[range] += comps;
    }
    query_count += other.query_count;
  }
};

// ==================== Batch Query Worker ====================

struct QueryBatch {
  size_t start_idx;
  size_t end_idx;
  int thread_id;
};

void process_hnsw_query_batch(
    const QueryBatch& batch,
    KnnFirstWrapper* hnsw_index,  // 移除 const
    const DataWrapper* data_wrapper,
    const BaseIndex::SearchParams& s_params,
    ThreadLocalResult& local_result) {

  // Each thread gets its own SearchInfo to avoid race conditions
  BaseIndex::IndexParams i_params(16, 200, 200, 500);  // Default params
  BaseIndex::SearchInfo local_search_info(data_wrapper, &i_params, "HNSW_Baseline_MT", "benchmark");

  for (size_t idx = batch.start_idx; idx < batch.end_idx; idx++) {
    int one_id = data_wrapper->query_ids.at(idx);

    // Calculate query range
    int query_range = data_wrapper->query_ranges.at(idx).second -
                      data_wrapper->query_ranges.at(idx).first + 1;

    // Perform search
    auto res = hnsw_index->rangeFilteringSearchOutBound(
        &s_params, &local_search_info,
        data_wrapper->querys.at(one_id),
        data_wrapper->query_ranges.at(idx));

    // Calculate precision
    float precision = countPrecision(data_wrapper->groundtruth.at(idx), res);

    // Accumulate local results
    local_result.result_recorder[query_range].first += precision;
    local_result.result_recorder[query_range].second += local_search_info.internal_search_time;
    local_result.comparison_recorder[query_range] += local_search_info.total_comparison;
    local_result.query_count++;
  }
}

// ==================== Multi-threaded Query Runner ====================
// 设计理念：每个range串行执行，32个线程并发处理该range的所有查询
// 这样可以准确计算每个range的wall clock time和QPS

void run_multithreaded_hnsw_queries(
    KnnFirstWrapper* hnsw_index,  // 移除 const
    const DataWrapper* data_wrapper,
    const BaseIndex::SearchParams& s_params,
    int num_threads,
    int batch_size) {

  const size_t total_queries = data_wrapper->query_ids.size();

  cout << "Multi-threaded Query Configuration (SERIAL RANGE MODE):" << endl;
  cout << "  Total queries: " << total_queries << endl;
  cout << "  Worker threads: " << num_threads << endl;
  cout << "  search_ef: " << s_params.search_ef << endl;
  cout << "  Strategy: Execute each range serially with " << num_threads << " threads" << endl;
  cout << endl;

  // CRITICAL: Set ef BEFORE launching threads to avoid race condition
  // (32 threads calling setEf() simultaneously causes SIGSEGV)
  hnsw_index->setSearchEf(s_params.search_ef);

  // Step 1: Group query indices by range
  std::map<int, vector<size_t>> queries_by_range;
  for (size_t idx = 0; idx < total_queries; idx++) {
    int query_range = data_wrapper->query_ranges.at(idx).second -
                      data_wrapper->query_ranges.at(idx).first + 1;
    queries_by_range[query_range].push_back(idx);
  }

  cout << "Found " << queries_by_range.size() << " distinct ranges:" << endl;
  for (const auto& [range, indices] : queries_by_range) {
    cout << "  Range " << range << ": " << indices.size() << " queries" << endl;
  }
  cout << endl;

  // Step 2: Execute each range independently
  for (const auto& [range, query_indices] : queries_by_range) {
    const size_t range_query_count = query_indices.size();

    cout << "========================================" << endl;
    cout << "Processing Range: " << range << " (" << range_query_count << " queries)" << endl;
    cout << "========================================" << endl;

    // Reset thread-local results for this range
    vector<ThreadLocalResult> thread_results(num_threads);

    // Atomic counter for this range's queries
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> completed_queries{0};

    timeval range_start, range_end;
    gettimeofday(&range_start, NULL);

    // Launch worker threads for this range
    vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
      workers.emplace_back([&, t]() {
        ThreadLocalResult& local_result = thread_results[t];

        // 每个线程有自己的 SearchInfo
        BaseIndex::IndexParams local_i_params(16, 200, 200, 500);
        BaseIndex::SearchInfo local_search_info(data_wrapper, &local_i_params, "HNSW_Baseline_MT", "benchmark");

        // 线程循环抢查询
        while (true) {
          // 原子地获取下一个查询索引（在当前range的查询列表中）
          size_t local_idx = next_query_idx.fetch_add(1);

          // 检查是否当前range的所有查询都已处理
          if (local_idx >= range_query_count) {
            break;
          }

          // 获取实际的查询索引
          size_t query_idx = query_indices[local_idx];
          int one_id = data_wrapper->query_ids.at(query_idx);

          // 执行搜索
          auto res = hnsw_index->rangeFilteringSearchOutBound(
              &s_params, &local_search_info,
              data_wrapper->querys.at(one_id),
              data_wrapper->query_ranges.at(query_idx));

          // 计算精度
          float precision = countPrecision(data_wrapper->groundtruth.at(query_idx), res);

          // 累积到本地结果（无锁）
          local_result.result_recorder[range].first += precision;
          local_result.comparison_recorder[range] += local_search_info.total_comparison;
          local_result.query_count++;

          // 进度显示（每100个查询）
          size_t completed = completed_queries.fetch_add(1) + 1;
          if (t == 0 && completed % 100 == 0) {
            cout << "  Progress: " << completed << "/" << range_query_count << " queries\r" << std::flush;
          }
        }
      });
    }

    // Wait for all threads to complete for this range
    for (auto& worker : workers) {
      worker.join();
    }

    gettimeofday(&range_end, NULL);

    cout << endl;  // 换行

    // Calculate wall clock time for this range (in seconds)
    double range_time = (range_end.tv_sec - range_start.tv_sec) +
                       (range_end.tv_usec - range_start.tv_usec) / 1000000.0;

    // Merge all thread results for this range
    ThreadLocalResult range_result;
    for (int t = 0; t < num_threads; t++) {
      range_result.merge(thread_results[t]);
    }

    // Output results for this range
    float total_recall = range_result.result_recorder[range].first;
    float avg_comps = range_result.comparison_recorder[range] / range_query_count;
    float avg_recall = total_recall / range_query_count;
    float qps = range_query_count / range_time;

    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << range
         << "\t recall: " << avg_recall
         << "\t QPS: " << std::setprecision(0)
         << qps
         << "\t Comps: " << std::setprecision(2)
         << avg_comps
         << "\t Time: " << std::setprecision(4)
         << range_time << "s"
         << endl;
    cout << endl;
  }

  cout << "========================================" << endl;
  cout << "All ranges completed!" << endl;
  cout << "========================================" << endl;
}

// ==================== Main Function ====================

void log_result_recorder(
    const std::map<int, std::pair<float, float>>& result_recorder,
    const std::map<int, float>& comparison_recorder,
    const int amount) {
  for (auto it : result_recorder) {
    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << it.first
         << "\t recall: " << it.second.first / (amount / result_recorder.size())
         << "\t QPS: " << std::setprecision(0)
         << (amount / result_recorder.size()) / it.second.second
         << "\t Comps: " << comparison_recorder.at(it.first) / (amount / result_recorder.size())
         << endl;
  }
}

int main(int argc, char** argv) {
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif
#ifdef USE_AVX
  cout << "Use AVX" << endl;
#endif
#ifdef USE_AVX512
  cout << "Use AVX512" << endl;
#endif
#ifndef NO_PARALLEL_BUILD
  cout << "Index Construct Parallelly" << endl;
#endif

  // Parameters
  string dataset = "deep";
  int data_size = 1000000;
  string dataset_path = "";
  string query_path = "";
  string groundtruth_path = "";
  int index_k = 16;
  int ef_construction = 200;
  int ef_max = 500;
  int search_ef = 200;
  vector<int> search_ef_list;
  int query_num = 1000;
  int query_k = 10;

  string save_index_path = "";
  string load_index_path = "";
  string version = "HNSW_MultiThread";
  bool full_range = false;
  bool generate_gt_only = false;
  bool build_only = false;

  // Multi-threading parameters
  int num_threads = 8;
  int batch_size = 32;
  bool use_multithread = true;

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
    if (arg == "-index_k") index_k = atoi(argv[i + 1]);
    if (arg == "-ef_con") ef_construction = atoi(argv[i + 1]);
    if (arg == "-ef_max") ef_max = atoi(argv[i + 1]);
    if (arg == "-ef_search") search_ef = atoi(argv[i + 1]);
    if (arg == "-ef_search_list") {
      string list_str = string(argv[i + 1]);
      std::stringstream ss(list_str);
      string item;
      while (std::getline(ss, item, ',')) {
        search_ef_list.push_back(std::stoi(item));
      }
    }
    if (arg == "-save_index") save_index_path = string(argv[i + 1]);
    if (arg == "-load_index") load_index_path = string(argv[i + 1]);
    if (arg == "-full_range") full_range = true;
    if (arg == "-generate_gt_only") generate_gt_only = true;
    if (arg == "-build_only") build_only = true;
    if (arg == "-threads") num_threads = atoi(argv[i + 1]);
    if (arg == "-batch_size") batch_size = atoi(argv[i + 1]);
    if (arg == "-single_thread") use_multithread = false;
  }

  if (search_ef_list.empty()) {
    search_ef_list.push_back(search_ef);
  }

  if (build_only) {
    query_num = 0;
  }

  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
  data_wrapper.readData(dataset_path, query_path);

  // Mode 1: Generate groundtruth only
  if (generate_gt_only) {
    cout << "=== Generating Groundtruth Only ===" << endl;
    if (full_range)
      data_wrapper.generateRangeFilteringQueriesAndGroundtruth(!groundtruth_path.empty(), groundtruth_path);
    else
      data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(!groundtruth_path.empty(), groundtruth_path);

    cout << "Groundtruth generated: " << data_wrapper.query_ids.size() << " queries" << endl;
    if (!groundtruth_path.empty()) {
      cout << "Saved groundtruth to: " << groundtruth_path << endl;
      string query_file = groundtruth_path;
      size_t pos = query_file.find(".csv");
      if (pos != string::npos) {
        query_file = query_file.substr(0, pos) + ".fvecs";
      } else {
        query_file += ".fvecs";
      }
      cout << "Saving queries to: " << query_file << endl;
      SaveQueriesToFile(query_file, data_wrapper.querys);
      cout << "Query file saved: " << query_file << endl;
    }
    cout << "=== Done. Exiting (no index built). ===" << endl;
    return 0;
  }

  // Mode 1.5: Build index only (no groundtruth, no queries)
  if (build_only) {
    cout << "=== Build Only Mode (no groundtruth, no queries) ===" << endl;
    cout << "Parameters: index_k=" << index_k << ", ef_con=" << ef_construction
         << ", ef_max=" << ef_max << endl;

    data_wrapper.version = version;

    BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);

    KnnFirstWrapper hnsw_index(&data_wrapper);

    timeval t1, t2;
    cout << "=== Building Index ===" << endl;
    gettimeofday(&t1, NULL);
    hnsw_index.buildIndex(&i_params);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Build Index Time");
    cout << "Total # of Neighbors: " << hnsw_index.index_info->nodes_amount << endl;

    if (!save_index_path.empty()) {
      cout << "=== Saving Index to: " << save_index_path << " ===" << endl;
      hnsw_index.saveIndex(save_index_path);
    }
    cout << "=== Done. Exiting (build only). ===" << endl;
    return 0;
  }

  // Mode 2: Load existing groundtruth or generate new one
  if (groundtruth_path != "") {
    string query_file = "";
    if (!query_path.empty()) {
      query_file = query_path;
    } else {
      query_file = groundtruth_path;
      size_t pos = query_file.find(".csv");
      if (pos != string::npos) {
        query_file = query_file.substr(0, pos) + ".fvecs";
      } else {
        query_file += ".fvecs";
      }
    }
    cout << "Loading groundtruth from: " << groundtruth_path << endl;
    cout << "Loading queries from: " << query_file << endl;
    data_wrapper.LoadGroundtruth(groundtruth_path, query_file);
  } else {
    cout << "========================================" << endl;
    cout << "Generating Groundtruth (SINGLE THREADED)" << endl;
    cout << "========================================" << endl;
    cout << "This may take 10-60 minutes depending on dataset..." << endl;
    cout << endl;

    timeval gt_start, gt_end;
    gettimeofday(&gt_start, NULL);

    if (full_range)
      data_wrapper.generateRangeFilteringQueriesAndGroundtruth(false);
    else
      data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(false);

    gettimeofday(&gt_end, NULL);

    cout << endl;
    cout << "========================================" << endl;
    cout << "Groundtruth Generation Complete!" << endl;
    cout << "========================================" << endl;
    logTime(gt_start, gt_end, "Groundtruth Generation Time");
    cout << endl;
  }

  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  cout << "Parameters: index_k=" << index_k << ", ef_con=" << ef_construction
       << ", ef_max=" << ef_max << ", ef_search=" << search_ef << endl;

  data_wrapper.version = version;
  base_hnsw::L2Space ss(data_wrapper.data_dim);

  BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);
  BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "HNSW_Baseline", "benchmark");

  timeval t1, t2;

  KnnFirstWrapper hnsw_index(&data_wrapper);

  // Mode 3: Load existing index and test only
  if (!load_index_path.empty()) {
    cout << "=== Loading Index from: " << load_index_path << " ===" << endl;
    gettimeofday(&t1, NULL);
    hnsw_index.loadIndex(load_index_path);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Load Index Time");
  } else {
    // Mode 4: Build new index
    cout << "=== Building Index ===" << endl;
    gettimeofday(&t1, NULL);
    hnsw_index.buildIndex(&i_params);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Build Index Time");
    cout << "Total # of Neighbors: " << hnsw_index.index_info->nodes_amount << endl;

    if (!save_index_path.empty()) {
      cout << "=== Saving Index to: " << save_index_path << " ===" << endl;
      hnsw_index.saveIndex(save_index_path);
    }
  }

  // Run queries with multiple search_ef values
  cout << "=== Running Queries with " << search_ef_list.size() << " search_ef values ===" << endl;
  for (int current_search_ef : search_ef_list) {
    cout << endl;
    cout << "========================================" << endl;
    cout << "Testing with search_ef=" << current_search_ef << endl;
    cout << "========================================" << endl;

    BaseIndex::SearchParams s_params;
    s_params.query_K = data_wrapper.query_k;
    s_params.search_ef = current_search_ef;

    if (use_multithread) {
      run_multithreaded_hnsw_queries(&hnsw_index, &data_wrapper, s_params, num_threads, batch_size);
    } else {
      // Single-threaded execution (original code)
      cout << "Running single-threaded queries..." << endl;

      timeval tt3, tt4;
      std::map<int, std::pair<float, float>> result_recorder;
      std::map<int, float> comparison_recorder;

      gettimeofday(&tt3, NULL);
      for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
        int one_id = data_wrapper.query_ids.at(idx);
        s_params.query_range =
            data_wrapper.query_ranges.at(idx).second -
            data_wrapper.query_ranges.at(idx).first + 1;
        auto res = hnsw_index.rangeFilteringSearchOutBound(
            &s_params, &search_info, data_wrapper.querys.at(one_id),
            data_wrapper.query_ranges.at(idx));
        search_info.precision =
            countPrecision(data_wrapper.groundtruth.at(idx), res);
        result_recorder[s_params.query_range].first +=
            search_info.precision;
        result_recorder[s_params.query_range].second +=
            search_info.internal_search_time;
        comparison_recorder[s_params.query_range] +=
            search_info.total_comparison;
      }
      gettimeofday(&tt4, NULL);

      log_result_recorder(result_recorder, comparison_recorder,
                          data_wrapper.query_ids.size());
      logTime(tt3, tt4, "total query time");
      cout << endl;
    }
  }

  return 0;
}

