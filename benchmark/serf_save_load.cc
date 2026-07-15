/**
 * @file serf_save_load.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief SeRF Benchmark with Save/Load Index Support
 * @date 2025-01-04
 *
 * @copyright Copyright (c) 2025
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

void log_result_recorder(
    const std::map<int, std::pair<float, float>> &result_recorder,
    const std::map<int, float> &comparison_recorder, const int amount) {
  for (auto it : result_recorder) {
    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << it.first
         << "\t recall: " << it.second.first / (amount / result_recorder.size())
         << "\t QPS: " << std::setprecision(0)
         << (amount / result_recorder.size()) / it.second.second << "\t Comps: "
         << comparison_recorder.at(it.first) / (amount / result_recorder.size())
         << endl;
  }
}

int main(int argc, char **argv) {
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
  int index_k = 8;
  int ef_construction = 100;
  int ef_max = 500;
  int search_ef = 100;
  vector<int> search_ef_list;  // Support multiple search_ef values
  int query_num = 1000;
  int query_k = 10;

  string save_index_path = "";
  string load_index_path = "";
  string recursion_type_str = "MAX_POS";
  string version = "SeRF_SaveLoad";
  bool full_range = false;
  bool generate_gt_only = false;

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
      // Parse comma-separated list: e.g., "100,200,300,400"
      string list_str = string(argv[i + 1]);
      std::stringstream ss(list_str);
      string item;
      while (std::getline(ss, item, ',')) {
        search_ef_list.push_back(std::stoi(item));
      }
    }
    if (arg == "-save_index") save_index_path = string(argv[i + 1]);
    if (arg == "-load_index") load_index_path = string(argv[i + 1]);
    if (arg == "-recursion_type") recursion_type_str = string(argv[i + 1]);
    if (arg == "-full_range") full_range = true;
    if (arg == "-generate_gt_only") generate_gt_only = true;
  }

  // If no search_ef_list provided, use single search_ef
  if (search_ef_list.empty()) {
    search_ef_list.push_back(search_ef);
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
    cout << "Generating groundtruth..." << endl;
    if (full_range)
      data_wrapper.generateRangeFilteringQueriesAndGroundtruth(false);
    else
      data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(false);
  }

  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  cout << "Parameters: index_k=" << index_k << ", ef_con=" << ef_construction
       << ", ef_max=" << ef_max << ", ef_search=" << search_ef << endl;

  data_wrapper.version = version;
  base_hnsw::L2Space ss(data_wrapper.data_dim);

  BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);

  // Set recursion type
  if (recursion_type_str == "MIN_POS") {
    i_params.recursion_type = BaseIndex::IndexParams::MIN_POS;
  } else if (recursion_type_str == "MID_POS") {
    i_params.recursion_type = BaseIndex::IndexParams::MID_POS;
  } else {
    i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
  }

  SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
  BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D", "benchmark");

  timeval t1, t2;

  // Mode 3: Load existing index and test only
  if (!load_index_path.empty()) {
    cout << "=== Loading Index from: " << load_index_path << " ===" << endl;
    gettimeofday(&t1, NULL);
    index.load(load_index_path);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Load Index Time");
  } else {
    // Mode 4: Build new index
    cout << "=== Building Index ===" << endl;
    gettimeofday(&t1, NULL);
    index.buildIndex(&i_params);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Build Index Time");
    cout << "Total # of Neighbors: " << index.index_info->nodes_amount << endl;

    // Save index if path specified
    if (!save_index_path.empty()) {
      cout << "=== Saving Index to: " << save_index_path << " ===" << endl;
      index.save(save_index_path);
    }
  }

  // Run range queries with multiple search_ef values
  cout << "=== Running Range Queries with " << search_ef_list.size() << " search_ef values ===" << endl;
  for (int current_search_ef : search_ef_list) {
    cout << "--- Testing with search_ef=" << current_search_ef << " ---" << endl;

    timeval tt3, tt4;
    BaseIndex::SearchParams s_params;
    s_params.query_K = data_wrapper.query_k;
    s_params.search_ef = current_search_ef;

    std::map<int, std::pair<float, float>> result_recorder;
    std::map<int, float> comparison_recorder;

    gettimeofday(&tt3, NULL);
    for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
      int one_id = data_wrapper.query_ids.at(idx);
      s_params.query_range =
          data_wrapper.query_ranges.at(idx).second -
          data_wrapper.query_ranges.at(idx).first + 1;
      auto res = index.rangeFilteringSearchOutBound(
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

  return 0;
}

