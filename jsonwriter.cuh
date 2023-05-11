//jsonwriter.cuh

#ifndef JSON_WRITER_CUH
#define JSON_WRITER_CUH

#include "cliqueMerging.cuh"
#include <gunrock/gunrock.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filewritestream.h>
#include <iostream>

typedef typename gunrock::app::TestGraph<unsigned int, unsigned int, unsigned int,
                 gunrock::graph::HAS_CSR>GraphT;     //VertexT, SizeT, ValueT are uint
typedef typename GraphT::CsrT CsrT;

void outputJSON(std::string test_name, CsrT graph, gunrock::util::Parameters parameters, struct time_breakdown runtime_breakdown, struct bfs_loop_breakdown bfs_timings, struct dfs_loop_breakdown dfs_timings, struct clique_node* cliques, std::deque<float> loopRuntimes, std::deque<long long unsigned int> dfsCumulativeCliques, std::deque<struct time_breakdown> allRuntimes) {
    std::cout << "Start JSON writer" << std::endl;

    bool json = parameters.Get<bool>("json");
    if (!json) {
        return;
    }

    std::string schema = "8 May 2023";

    //Get date and time, including milliseconds, for timestamp
    long ms; // milliseconds
    struct timespec spec;
    time_t timer = time(NULL);
    std::string time_str = std::string(ctime(&timer));  //timestamp without milliseconds
    clock_gettime(CLOCK_REALTIME, &spec);
    ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
    if (ms > 999) {
        ms = 0;
    }
    std::string time_ms_str = std::to_string(ms);


    //Generate filename for the JSON
    std::string time_str_filename = time_str.substr(0, time_str.size() - 5) + time_ms_str + ' ' + time_str.substr(time_str.length() - 5);
    std::string dataset = parameters.Get<std::string>("dataset");
    std::string dir = parameters.Get<std::string>("jsondir");
    if (dir == "") {
        dir = "./eval";
    }
    else {
        utsname uts;
        uname(&uts);
        dir = dir + "_" + std::string(uts.nodename);
    }
    std::string heuristic_flag = parameters.Get<std::string>("heuristic");
    std::string pruning = parameters.Get<std::string>("pruning");
    bool bfs_flag = parameters.Get<bool>("bfs");
    bool dfs_flag = parameters.Get<bool>("windowing");
    std::string fileLabel = parameters.Get<std::string>("json_label");
    std::string json_filename = dir + "/" + "maxClique_" + ((dataset != "") ? (dataset + "_") : "") + (bfs_flag ? "bfs_" : "") + (dfs_flag ? "dfs_" : "") + heuristic_flag + "_" + pruning + "_" + time_str_filename + "_" + fileLabel + ".json";
    char bad_chars[] = ":\n";
    for (unsigned int i = 0; i < strlen(bad_chars); ++i) {
        json_filename.erase(std::remove(json_filename.begin(), json_filename.end(), bad_chars[i]), json_filename.end());
    }

    //std::cout << "JSON filename:" << json_filename.c_str() << std::endl;
    FILE* fp = fopen(json_filename.c_str(), "w");
    char writeBuffer[65536];
    rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);

    //std::cout << "Start JSON object" << std::endl;
    writer.StartObject();

    writer.Key("64bit-SizeT");
    writer.Bool(parameters.Get<bool>("64bit-SizeT"));

    writer.Key("64bit-ValueT");
    writer.Bool(parameters.Get<bool>("64bit-ValueT"));

    writer.Key("64bit-VertexT");
    writer.Bool(parameters.Get<bool>("64bit-VertexT"));

    writer.Key("all_runtime_breakdowns");
    writer.StartArray();
    for (time_breakdown run : allRuntimes) {
        writer.StartObject();
            writer.Key("total_time");
            writer.Double(run.total);
            writer.Key("kcore_time");
            writer.Double(run.kcore);
            writer.Key("heuristic_time");
            writer.Double(run.heuristic);
            writer.Key("presort_time");
            writer.Double(run.presort);
            writer.Key("two_cliques_time");
            writer.Double(run.two_cliques);
            writer.Key("postsort_time");
            writer.Double(run.postsort);
            writer.Key("preproc_total_time");
            writer.Double(run.total_preproc);
            writer.Key("dfs_time");
            writer.Double(run.dfs);
            writer.Key("bfs_time");
            writer.Double(run.bfs);
        writer.EndObject();
    }
    writer.EndArray();

    writer.Key("bfs_clique_counts");
    struct clique_node* currCliques = cliques;
    int cliqueSize = cliques->k;
    std::vector<long long unsigned int> cliqueVec = {};
    auto it = cliqueVec.begin();
    while ((cliqueSize > 1) && (currCliques != NULL)) {
        it = cliqueVec.insert(it, currCliques->numVertices);
        cliqueSize--;
        currCliques = currCliques->previous;
    }
    writer.StartArray();
    for (int i : cliqueVec) {
        writer.Uint64(i);
    }
    writer.EndArray();

    writer.Key("bfs_flag");
    writer.Bool(bfs_flag);

    writer.Key("bfs_loop_timings");
    writer.StartObject();
        writer.Key("count_cliques_time");
        writer.Double(bfs_timings.count);
        writer.Key("scan_alloc_time");
        writer.Double(bfs_timings.scan_alloc);
        writer.Key("merge_time");
        writer.Double(bfs_timings.merge);
    writer.EndObject();

    writer.Key("block_size");
    writer.Uint(BLOCK_SIZE);

    writer.Key("window_size");
    writer.Uint(parameters.Get<unsigned int>("window_size"));

    writer.Key("command-line");
    writer.String(parameters.Get_CommandLine());

    writer.Key("compiler-version");
    writer.Int((__GNUC__ % 100) * 10000000 + (__GNUC_MINOR__ % 100) * 100000 + (__GNUC_PATCHLEVEL__ % 100000));

    writer.Key("dataset");
    writer.String(dataset);

    writer.Key("device");
    writer.Int(parameters.Get<int>("device"));

    writer.Key("dfs_clique_counts");
    writer.StartArray();
    for (float i : dfsCumulativeCliques) {
        writer.Uint64(i);
    }
    writer.EndArray();

    writer.Key("dfs_flag");
    writer.Bool(dfs_flag);

    writer.Key("dfs_loop_timings");
    writer.StartObject();
        writer.Key("find_window_time");
        writer.Double(dfs_timings.find_window);
        writer.Key("count_cliques_time");
        writer.Double(dfs_timings.count);
        writer.Key("scan_alloc_time");
        writer.Double(dfs_timings.scan_alloc);
        writer.Key("merge_time");
        writer.Double(dfs_timings.merge);
    writer.EndObject();

    writer.Key("expected_clique_size");
    writer.Uint(parameters.Get<unsigned int>("expected_max"));

    writer.Key("frac_seeds");
    writer.Double(parameters.Get<float>("frac_seeds"));

    writer.Key("git-commit-sha");
    writer.String(GIT_SHA1);

    writer.Key("gunrock-git-commit-sha");
    writer.String(GUNROCK_GIT_SHA1);

    cudaDeviceProp devProps;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount != 0) {
        writer.Key("gpuinfo");
        writer.StartObject();
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&devProps, dev);
        writer.Key("name");
        writer.String(devProps.name);
        writer.Key("total_global_mem");
        writer.Int64(int64_t(devProps.totalGlobalMem));
        writer.Key("major");
        writer.String(std::to_string(devProps.major));
        writer.Key("minor");
        writer.String(std::to_string(devProps.minor));
        writer.Key("clock_rate");
        writer.Int(devProps.clockRate);
        writer.Key("multi_processor_count");
        writer.Int(devProps.multiProcessorCount);
        int runtimeVersion, driverVersion;
        cudaRuntimeGetVersion(&runtimeVersion);
        cudaDriverGetVersion(&driverVersion);
        writer.Key("driver_api");
        writer.String(std::to_string(CUDA_VERSION));
        writer.Key("driver_version");
        writer.String(std::to_string(driverVersion));
        writer.Key("runtime_version");
        writer.String(std::to_string(runtimeVersion));
        writer.Key("compute_version");
        writer.String(std::to_string(devProps.major * 10 + devProps.minor));
        writer.EndObject();
    }

    writer.Key("graph-file");
    writer.String(parameters.Get<std::string>("graph-file"));

    writer.Key("graph-type");
    writer.String(parameters.Get<std::string>("graph-type"));

    writer.Key("help");
    writer.Bool(parameters.Get<bool>("help"));

    writer.Key("heuristic_clique_size");
    writer.Int(parameters.Get<int>("heuristic_clique_size"));

    writer.Key("heuristic_flag");
    writer.String(heuristic_flag);

    writer.Key("json");
    writer.Bool(json);

    writer.Key("json_label");
    writer.String(parameters.Get<std::string>("json_label"));

    writer.Key("json-schema");
    writer.String(schema);

    writer.Key("jsondir");
    writer.String(dir);

    writer.Key("jsonfile");
    writer.String(json_filename);

    writer.Key("kcore");
    writer.Bool(parameters.Get<bool>("kcore"));

    writer.Key("kcore_average");
    writer.Uint(parameters.Get<unsigned int>("kcore_average"));

    writer.Key("kcore_max");
    writer.Uint(parameters.Get<unsigned int>("kcore_max"));

    writer.Key("load-time");
    writer.Double(parameters.Get<float>("load-time"));

    writer.Key("loop_runtimes");
    writer.StartArray();
    for (float i : loopRuntimes) {
        writer.Double(i);
    }
    writer.EndArray();

    writer.Key("lowerbound_in");
    writer.Uint(parameters.Get<unsigned int>("lowerbound"));

    writer.Key("maximum_clique_size");
    if (parameters.Get<bool>("preempt_main")) {
        writer.Int(parameters.Get<int>("heuristic_clique_size"));
    }
    else {
        writer.Int(cliques->k);
    }

    writer.Key("num_cliques");
    writer.Bool(parameters.Get<bool>("num_cliques"));

    writer.Key("num-edges");
    writer.Uint(graph.edges);

    writer.Key("num_runs");
    writer.Uint(parameters.Get<unsigned int>("num_runs"));

    writer.Key("num_seeds");
    writer.Uint(parameters.Get<unsigned int>("num_seeds"));

    writer.Key("num-vertices");
    writer.Uint(graph.nodes);

    writer.Key("oom");
    writer.Bool(parameters.Get<bool>("oom"));

    writer.Key("overall_perf");
    writer.Bool(parameters.Get<bool>("overall_perf"));

    writer.Key("order_candidates");
    writer.Bool(parameters.Get<bool>("order_candidates"));

    writer.Key("orientation");
    writer.String(parameters.Get<std::string>("orientation"));

    writer.Key("peak_mem_use");
    writer.Uint(parameters.Get<unsigned int>("peak_mem_use"));

    writer.Key("preempt_main");
    writer.Bool(parameters.Get<bool>("preempt_main"));

    writer.Key("primitive");
    writer.String("max_clique");

    writer.Key("pruning");
    writer.String(pruning);

    writer.Key("quick");
    writer.Bool(parameters.Get<bool>("quick"));

    writer.Key("quiet");
    writer.Bool(parameters.Get<bool>("quiet"));

    writer.Key("remove-duplicate-edges");
    writer.Bool(parameters.Get<bool>("remove-duplicate-edges"));

    writer.Key("remove-self-loops");
    writer.Bool(parameters.Get<bool>("remove-self-loops"));

    writer.Key("runtime_breakdown");
    writer.StartObject();
        writer.Key("total_time");
        writer.Double(runtime_breakdown.total);
        writer.Key("kcore_time");
        writer.Double(runtime_breakdown.kcore);
        writer.Key("heuristic_time");
        writer.Double(runtime_breakdown.heuristic);
        writer.Key("presort_time");
        writer.Double(runtime_breakdown.presort);
        writer.Key("two_cliques_time");
        writer.Double(runtime_breakdown.two_cliques);
        writer.Key("postsort_time");
        writer.Double(runtime_breakdown.postsort);
        writer.Key("preproc_total_time");
        writer.Double(runtime_breakdown.total_preproc);
        writer.Key("dfs_time");
        writer.Double(runtime_breakdown.dfs);
        writer.Key("bfs_time");
        writer.Double(runtime_breakdown.bfs);
    writer.EndObject();

    writer.Key("sort-csr");
    writer.Bool(parameters.Get<bool>("sort-csr"));

    writer.Key("sort_sublists");
    writer.Bool(parameters.Get<bool>("sort_sublists"));

    writer.Key("stddev-degree");
    writer.Double(gunrock::graph::GetStddevDegree(graph));

    writer.Key("sublist_descend");
    writer.Bool(parameters.Get<bool>("sublist_descend"));

    writer.Key("sysinfo");
    writer.StartObject();
        utsname uts;
        uname(&uts);
        writer.Key("sysname");
        writer.String(std::string(uts.sysname));
        writer.Key("release");
        writer.String(std::string(uts.release));
        writer.Key("version");
        writer.String(std::string(uts.version));
        writer.Key("machine");
        writer.String(std::string(uts.machine));
        writer.Key("nodename");
        writer.String(std::string(uts.nodename));
    writer.EndObject();

    writer.Key("tag");
    std::vector<std::string> param_tags = parameters.Get<std::vector<std::string>>("tag");
    writer.StartArray();
    for (std::string item : param_tags) {
        writer.String(item);
    }
    writer.EndArray();

    writer.Key("test_name");
    writer.String(test_name);

    writer.Key("time");
    writer.String(time_str);

    writer.Key("timing");
    writer.String(parameters.Get<std::string>("timing"));

    writer.Key("undirected");
    writer.Bool(parameters.Get<bool>("undirected"));

    writer.Key("userinfo");
    writer.StartObject();
        writer.Key("login");
        if (getpwuid(getuid())) {
            writer.String(getpwuid(getuid())->pw_name);
        }
        else {
            writer.String("Not Found");
        }
    writer.EndObject();

    writer.Key("v");
    writer.Bool(parameters.Get<bool>("v"));

    writer.Key("vertex-start-from-zero");
    writer.Bool(parameters.Get<bool>("vertex-start-from-zero"));

    writer.EndObject();

    fclose(fp);

    return;
}

#endif
