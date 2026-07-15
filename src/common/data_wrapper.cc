#include "data_wrapper.h"

#include "reader.h"
#include "utils.h"
#include <random>
#include <algorithm>
#include <omp.h>
#include <atomic>

void SynthesizeQuerys(const vector<vector<float>> &nodes,
                      vector<vector<float>> &querys, const int query_num) {
  int dim = nodes.front().size();
  std::default_random_engine e(42);  // Fixed seed for reproducibility
  std::uniform_int_distribution<int> u(0, nodes.size() - 1);
  querys.clear();
  querys.resize(query_num);

  for (unsigned n = 0; n < query_num; n++) {
    for (unsigned i = 0; i < dim; i++) {
      int select_idx = u(e);
      querys[n].emplace_back(nodes[select_idx][i]);
    }
  }
}

void DataWrapper::readData(string &dataset_path, string &query_path) {
  ReadDataWrapper(dataset, dataset_path, this->nodes, data_size, query_path,
                  this->querys, query_num, this->nodes_keys);
  cout << "Load vecs from: " << dataset_path << endl;
  cout << "# of vecs: " << nodes.size() << endl;

  // already sort data in sorted_data
  // if (dataset != "wiki-image" && dataset != "yt8m") {
  //   nodes_keys.resize(nodes.size());
  //   iota(nodes_keys.begin(), nodes_keys.end(), 0);
  // }

  if (querys.empty()) {
    cout << "Synthesizing querys..." << endl;
    SynthesizeQuerys(nodes, querys, query_num);
  }

  this->real_keys = false;
  vector<size_t> index_permutation;  // already sort data ahead

  // if (dataset == "wiki-image" || dataset == "yt8m") {
  //   cout << "first search_key before sorting: " << nodes_keys.front() <<
  //   endl; cout << "sorting dataset: " << dataset << endl; index_permutation =
  //   sort_permutation(nodes_keys); apply_permutation_in_place(nodes,
  //   index_permutation); apply_permutation_in_place(nodes_keys,
  //   index_permutation); cout << "Dimension: " << nodes.front().size() <<
  //   endl; cout << "first search_key: " << nodes_keys.front() << endl;
  //   this->real_keys = true;
  // }
  this->data_dim = this->nodes.front().size();
}

void SaveToCSVRow(const string &path, const int idx, const int l_bound,
                  const int r_bound, const int pos_range,
                  const int real_search_key_range, const int K_neighbor,
                  const double &search_time, const vector<int> &gt,
                  const vector<float> &pts) {
  std::ofstream file;
  file.open(path, std::ios_base::app);
  if (file) {
    file << idx << "," << l_bound << "," << r_bound << "," << pos_range << ","
         << real_search_key_range << "," << K_neighbor << "," << search_time
         << ",";
    for (auto ele : gt) {
      file << ele << " ";
    }
    // file << ",";
    // for (auto ele : pts) {
    //   file << ele << " ";
    // }
    file << "\n";
  }
  file.close();
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruth(
    bool is_save_to_file, const string save_path) {
  std::default_random_engine e;
  timeval t1, t2;
  double accu_time = 0.0;

  if (is_save_to_file) {
    std::ofstream(save_path, std::ios_base::trunc).close();
  }

  vector<int> query_range_list;
  float scale = 0.05;
  if (this->dataset == "local") {
    scale = 0.1;
  }
  int init_range = this->data_size * scale;
  while (init_range <= this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
#ifdef LOG_DEBUG_MODE
  if (this->dataset == "local") {
    query_range_list.erase(
        query_range_list.begin(),
        query_range_list.begin() + 4 * query_range_list.size() / 9);
  }
#endif
  if (0.01 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);

  if (0.001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);

  if (0.0001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);
  if (this->data_size == 1000000) {
    query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,
                        7000,   8000,   9000,   10000,  20000,  30000,
                        40000,  50000,  60000,  70000,  80000,  90000,
                        100000, 200000, 300000, 400000, 500000, 600000,
                        700000, 800000, 900000, 1000000};
  }

  int num_threads = omp_get_max_threads();
  cout << "Generating Groundtruth using " << num_threads << " OMP threads...\nRanges: ";
  print_set(query_range_list);
  // generate groundtruth

  for (auto range : query_range_list) {
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);

    // Pre-allocate random bounds for all queries to ensure thread safety
    vector<std::pair<int, int>> query_bounds(this->querys.size());
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range - 1;
      if (range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      query_bounds[i] = std::make_pair(l_bound, r_bound);
    }

    // Pre-allocate results vectors
    vector<vector<int>> gt_results(this->querys.size());
    vector<int> query_ids_local(this->querys.size());
    vector<std::pair<int, int>> query_ranges_local(this->querys.size());
    vector<double> greedy_times(this->querys.size());

    // Atomic counter for progress tracking
    std::atomic<int> progress_counter{0};
    int total_queries = this->querys.size();
    int last_progress = -1;

    // Generate groundtruth for this range using OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = query_bounds[i].first;
      int r_bound = query_bounds[i].second;

      timeval gt_start, gt_end;
      gettimeofday(&gt_start, NULL);

      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);

      gettimeofday(&gt_end, NULL);
      double greedy_time;
      CountTime(gt_start, gt_end, greedy_time);

      gt_results[i] = gt;
      query_ids_local[i] = i;
      query_ranges_local[i] = std::make_pair(l_bound, r_bound);
      greedy_times[i] = greedy_time;

      // Update progress (only in master thread to avoid interleaved output)
      int current_progress = ++progress_counter;
      if (omp_get_thread_num() == 0) {
        int progress_pct = (current_progress * 100) / total_queries;
        if (progress_pct >= last_progress + 25 || current_progress == total_queries) {
          #pragma omp critical
          {
            if (progress_pct >= last_progress + 25 || current_progress == total_queries) {
              cout << "    Progress: " << progress_pct << "% (" << current_progress
                   << "/" << total_queries << " queries)" << endl;
              last_progress = progress_pct;
            }
          }
        }
      }
    }

    // Merge results (must be done sequentially for vector order)
    for (int i = 0; i < this->querys.size(); i++) {
      groundtruth.emplace_back(gt_results[i]);
      query_ids.emplace_back(query_ids_local[i]);
      query_ranges.emplace_back(query_ranges_local[i]);
      accu_time += greedy_times[i];
      if (is_save_to_file) {
        int l_bound = query_ranges_local[i].first;
        int r_bound = query_ranges_local[i].second;
        int search_key_range = r_bound - l_bound + 1;
        if (this->real_keys) {
          search_key_range =
              this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
        }
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_times[i], gt_results[i],
                     this->querys.at(i));
      }
    }
  }

  cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// Get Groundtruth for half bounded query
void DataWrapper::generateHalfBoundedQueriesAndGroundtruth(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  int init_range = this->data_size * 0.05;
  while (init_range <= this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * 0.05;
  }
  if (0.01 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);

  if (0.001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);

  if (0.0001 * this->data_size > 100)
    query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);

  if (this->data_size == 1000000) {
    query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,
                        7000,   8000,   9000,   10000,  20000,  30000,
                        40000,  50000,  60000,  70000,  80000,  90000,
                        100000, 200000, 300000, 400000, 500000, 600000,
                        700000, 800000, 900000, 1000000};
  }

  if (this->data_size == 100000) {
    query_range_list = {1000,  2000,  3000,  4000,  5000,  6000,  7000,
                        8000,  9000,  10000, 20000, 30000, 40000, 50000,
                        60000, 70000, 80000, 90000, 100000};
  }

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

void SaveQueriesToFile(const string &query_path, const vector<vector<float>> &querys) {
  // Save queries in fvecs format
  std::ofstream file(query_path, std::ios::binary);
  if (!file.is_open()) {
    cout << "Failed to open query file for writing: " << query_path << endl;
    return;
  }

  int dim = querys.front().size();
  for (const auto &query : querys) {
    file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(query.data()), dim * sizeof(float));
  }
  file.close();
}

void DataWrapper::LoadGroundtruth(const string &gt_path, const string &query_path) {
  this->groundtruth.clear();
  this->query_ranges.clear();
  this->query_ids.clear();
  cout << "Loading Groundtruth from " << gt_path << "...";
  ReadGroundtruthQuery(this->groundtruth, this->query_ranges, this->query_ids,
                       gt_path);
  cout << " Done!" << endl;

  // Load query vectors if path is provided
  if (!query_path.empty()) {
    this->querys.clear();
    cout << "Loading Queries from " << query_path << "...";
    this->querys = ReadTopN(query_path, "fvecs", -1);
    cout << " Done! Loaded " << this->querys.size() << " queries" << endl;
  }
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthScalability(
    bool is_save_to_file, const string save_path) {
  std::default_random_engine e(42);  // Fixed seed for reproducibility
  timeval t1, t2;
  double accu_time = 0.0;

  vector<int> query_range_list;
  float scale = 0.001;
  int init_range = this->data_size * scale;
  while (init_range < 0.01 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
  scale = 0.01;
  init_range = this->data_size * scale;
  while (init_range < 0.1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  scale = 0.1;
  init_range = this->data_size * scale;
  while (init_range < 1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  query_range_list.emplace_back(this->data_size);

  cout << "Generating Groundtruth...\nRanges: ";
  print_set(query_range_list);
  cout << "sample size:" << this->nodes.size() << endl;
  // generate groundtruth

  for (auto range : query_range_list) {
    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - range - 80);
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + range - 1;
      if (range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      int search_key_range = r_bound - l_bound + 1;
      // if (this->real_keys)
      //   search_key_range =
      //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      accu_time += greedy_time;
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }

  cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// Get Groundtruth for half bounded query
void DataWrapper::generateHalfBoundedQueriesAndGroundtruthScalability(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  float scale = 0.001;
  int init_range = this->data_size * scale;
  while (init_range < 0.01 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }
  scale = 0.01;
  init_range = this->data_size * scale;
  while (init_range < 0.1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  scale = 0.1;
  init_range = this->data_size * scale;
  while (init_range < 1 * this->data_size) {
    query_range_list.emplace_back(init_range);
    init_range += this->data_size * scale;
  }

  query_range_list.emplace_back(this->data_size);

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// For evaluating in Benchmark, use only 7 points in the benchmark: 0.1% 0.5% 1%
// 5% 10% 50% 100%

void DataWrapper::generateHalfBoundedQueriesAndGroundtruthBenchmark(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  vector<int> query_range_list;
  query_range_list.emplace_back(this->data_size * 0.001);
  query_range_list.emplace_back(this->data_size * 0.005);
  query_range_list.emplace_back(this->data_size * 0.01);
  query_range_list.emplace_back(this->data_size * 0.05);
  query_range_list.emplace_back(this->data_size * 0.1);
  query_range_list.emplace_back(this->data_size * 0.2);
  query_range_list.emplace_back(this->data_size * 0.5);
  query_range_list.emplace_back(this->data_size);

  cout << "Generating Half Bounded Groundtruth...";
  cout << endl << "Ranges: " << endl;
  print_set(query_range_list);
  // generate groundtruth
  for (auto range : query_range_list) {
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = 0;
      int r_bound = range - 1;

      int search_key_range = r_bound - l_bound + 1;
      if (this->real_keys)
        search_key_range =
            this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
      query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
      double greedy_time;
      gettimeofday(&t1, NULL);
      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);
      gettimeofday(&t2, NULL);
      CountTime(t1, t2, greedy_time);
      groundtruth.emplace_back(gt);
      query_ids.emplace_back(i);
      if (is_save_to_file) {
        SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                     this->query_k, greedy_time, gt, this->querys.at(i));
      }
    }
  }
  cout << "  Done!" << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthBenchmark(
    bool is_save_to_file, const string save_path) {
  timeval t1, t2;

  // Fixed range percentages for benchmark testing
  vector<double> range_pcts = {1, 10, 20, 100};  // 1%, 10%, 20%, 100%

  cout << "Generating Range Filtering Groundtruth for benchmark ranges..." << endl;
  int num_threads = omp_get_max_threads();
  cout << "Using " << num_threads << " OMP threads for groundtruth generation" << endl;

  std::default_random_engine e(42);  // Fixed seed for reproducibility
  double accu_time = 0.0;

  if (is_save_to_file) {
    std::ofstream(save_path, std::ios_base::trunc).close();
  }

  // Process each range percentage
  for (double target_range_pct : range_pcts) {
    int target_range = (target_range_pct == 100) ? this->data_size : (int)(this->data_size * target_range_pct / 100);

    cout << "  Range: " << target_range_pct << "% (size=" << target_range << ")" << endl;

    std::uniform_int_distribution<int> u_lbound(0,
                                                this->data_size - target_range - 80);

    // Pre-allocate random bounds for all queries to ensure thread safety
    vector<std::pair<int, int>> query_bounds(this->querys.size());
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = u_lbound(e);
      int r_bound = l_bound + target_range - 1;
      if (target_range == this->data_size) {
        l_bound = 0;
        r_bound = this->data_size - 1;
      }
      query_bounds[i] = std::make_pair(l_bound, r_bound);
    }

    // Pre-allocate results vectors
    vector<vector<int>> gt_results(this->querys.size());
    vector<int> query_ids_local(this->querys.size());
    vector<std::pair<int, int>> query_ranges_local(this->querys.size());
    vector<double> greedy_times(this->querys.size());

    // Atomic counter for progress tracking
    std::atomic<int> progress_counter{0};
    int total_queries = this->querys.size();
    int last_progress = -1;

    // Generate groundtruth for this range using OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < this->querys.size(); i++) {
      int l_bound = query_bounds[i].first;
      int r_bound = query_bounds[i].second;

      timeval gt_start, gt_end;
      gettimeofday(&gt_start, NULL);

      auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                              this->query_k);

      gettimeofday(&gt_end, NULL);
      double greedy_time;
      CountTime(gt_start, gt_end, greedy_time);

      gt_results[i] = gt;
      query_ids_local[i] = i;
      query_ranges_local[i] = std::make_pair(l_bound, r_bound);
      greedy_times[i] = greedy_time;

      // Update progress (only in master thread to avoid interleaved output)
      int current_progress = ++progress_counter;
      if (omp_get_thread_num() == 0) {
        int progress_pct = (current_progress * 100) / total_queries;
        if (progress_pct >= last_progress + 25 || current_progress == total_queries) {
          #pragma omp critical
          {
            if (progress_pct >= last_progress + 25 || current_progress == total_queries) {
              cout << "    Progress: " << progress_pct << "% (" << current_progress
                   << "/" << total_queries << " queries)" << endl;
              last_progress = progress_pct;
            }
          }
        }
      }
    }

    // Merge results (must be done sequentially for vector order)
    for (int i = 0; i < this->querys.size(); i++) {
      groundtruth.emplace_back(gt_results[i]);
      query_ids.emplace_back(query_ids_local[i]);
      query_ranges.emplace_back(query_ranges_local[i]);
      accu_time += greedy_times[i];
      if (is_save_to_file) {
        int l_bound = query_ranges_local[i].first;
        int r_bound = query_ranges_local[i].second;
        int search_key_range = r_bound - l_bound + 1;
        SaveToCSVRow(save_path, i, l_bound, r_bound, target_range,
                     search_key_range, this->query_k, greedy_times[i],
                     gt_results[i], this->querys.at(i));
      }
    }
  }

  cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
  if (is_save_to_file) {
    cout << "Save GroundTruth to path: " << save_path << endl;
  }
}

// 生成不相交的固定大小范围查询（用于bucket-based场景）
// 将整个数据集 [0, data_size-1] 分成 num_partitions 个相等的区间
// 使用 querys 文件中的独立查询向量
// OPTIMIZED: Uses batched multi-threaded groundtruth generation
void DataWrapper::generatePartitionedQueriesAndGroundtruth(int num_partitions,
                                                           int queries_per_partition,
                                                           int window_size) {
  cout << "Generating Partitioned Queries..." << endl;

  std::default_random_engine generator(42);  // Fixed seed for reproducibility

  cout << "Data size: " << this->data_size << endl;
  cout << "Number of partitions: " << num_partitions << endl;
  cout << "Queries per partition: " << queries_per_partition << endl;
  cout << "Available querys: " << this->querys.size() << endl;

  // 计算每个分区的大小（尽量相等）
  int partition_size = this->data_size / num_partitions;

  cout << "Partition size: " << partition_size << endl;

  // 计算总共需要多少个查询
  int total_queries_needed = num_partitions * queries_per_partition;

  if (this->querys.size() < total_queries_needed) {
    cout << "Warning: Not enough querys! Need " << total_queries_needed
         << ", but only have " << this->querys.size() << endl;
    cout << "Will use all available querys." << endl;
  }

  // Calculate queries per partition (evenly distribute all available queries)
  int actual_queries_per_partition = this->querys.size() / num_partitions;
  if (actual_queries_per_partition == 0 && !this->querys.empty()) {
    actual_queries_per_partition = 1;
  }
  int remaining_queries = this->querys.size() % num_partitions;

  cout << "Actual queries per partition: " << actual_queries_per_partition
       << " (with " << remaining_queries << " extra)" << endl;

  int query_idx = 0;  // 当前使用的 query 索引

  for (int i = 0; i < num_partitions; i++) {
    int l_bound = i * partition_size;
    int r_bound = (i == num_partitions - 1) ? (this->data_size - 1) : (l_bound + partition_size - 1);

    // Collect queries for this partition (evenly distributed)
    vector<vector<float>> partition_queries;
    vector<int> partition_query_ids;

    // First 'remaining_queries' partitions get one extra query
    int queries_for_this_partition = actual_queries_per_partition;
    if (i < remaining_queries) {
      queries_for_this_partition++;
    }

    for (int q = 0; q < queries_for_this_partition; q++) {
      if (query_idx >= this->querys.size()) {
        break;
      }
      partition_queries.push_back(this->querys[query_idx]);
      partition_query_ids.push_back(query_idx);
      query_idx++;
    }

    if (partition_queries.empty()) {
      break;
    }

    // Print partition info and groundtruth generation on one line
    cout << "Partition " << i << ": [" << l_bound << ", " << r_bound
         << "] (size: " << (r_bound - l_bound + 1) << ") - "
         << partition_queries.size() << " queries, " << omp_get_max_threads() << " threads" << endl;
    auto batched_gt = greedyNearestBatched(this->nodes, partition_queries,
                                           l_bound, r_bound, this->query_k);

    // Store results
    for (size_t j = 0; j < batched_gt.size(); j++) {
      this->query_ids.push_back(partition_query_ids[j]);
      this->query_ranges.emplace_back(l_bound, r_bound);
      this->groundtruth.emplace_back(batched_gt[j]);
    }

    if (query_idx >= this->querys.size()) {
      break;
    }
  }

  cout << "Generated " << this->query_ids.size() << " query-range pairs" << endl;
}
