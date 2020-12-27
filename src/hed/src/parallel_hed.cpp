#include <iostream>
#include <getopt.h>
#include <thread>
#include <mutex>
#include <cmath>

#include "ged_env.hpp"

std::once_flag init_flag;

std::vector<ged::GEDGraph::GraphID> setup_environment(const std::string & graph_dir, const std::string & collection_file, ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> & env, double tv, double te, double alpha, double beta) {
	std::unordered_set<std::string> irrelevant_attributes;
	std::string options("");
	std::vector<double> edit_cost_constants {tv, te, alpha, beta};

	std::vector<ged::GEDGraph::GraphID> graph_ids(env.load_gxl_graphs(graph_dir, collection_file, ged::Options::GXLNodeEdgeType::LABELED, ged::Options::GXLNodeEdgeType::UNLABELED, irrelevant_attributes, irrelevant_attributes));
	env.set_edit_costs(ged::Options::EditCosts::LETTER, edit_cost_constants);
	env.init(ged::Options::InitType::LAZY_WITHOUT_SHUFFLED_COPIES);
	env.set_method(ged::Options::GEDMethod::HED, options);

	return graph_ids;
}

void init_matrix(double *&distances, int graph_num) {
	distances = new double[graph_num * graph_num];
}

void compute_indices(int thread_id, int thread_num, int graph_num,
			int &start_x, int &start_y,
			int &end_x, int &end_y) {

	int work_items, block_x, block_y, thread_x, thread_y;

	work_items = (int) std::ceil(std::pow(graph_num, 2.0)/thread_num);

	block_x = (int) std::ceil(std::sqrt(work_items));
	block_y = (int) std::floor(std::sqrt(work_items));
	thread_x = (int) std::floor(std::sqrt(thread_num));
	thread_y = (int) std::ceil(std::sqrt(thread_num));

	start_x = (thread_id % thread_x) * block_x;
	if (start_x + block_x < graph_num) {
		/* roud up work for last thread in thread row */
		if ((thread_id % thread_x) == thread_x -1) {
			end_x = graph_num;
		} else {
			end_x = start_x + block_x;
		}
	} else {
		end_x = graph_num;
	}

	start_y = (thread_id / thread_x) * block_y;
	if (start_y + block_y < graph_num) {
		if ((thread_id / thread_x) == thread_y - 1) {
			end_y = graph_num;
		} else {
			end_y = start_y + block_y;
		}
	} else {
		end_y = graph_num;
	}

}

int* compute_end_points(int thread_num, int graph_num) {
	int *end_points = new int[thread_num];

	float pairs = (graph_num - 1) * graph_num;
	pairs = pairs / 2;
	int limit = (int) std::ceil(pairs / thread_num);

	int idx = 0;
	int tmp = 0;
	for (int i = 1; i < graph_num; i++) {
		tmp = tmp + i;
		if (tmp >= limit) {
			end_points[idx] = i + 1;
			tmp = 0;
			idx++;
		}
	}
	end_points[thread_num-1] = graph_num;

	return end_points;
}

void compute_hed(int thread_id, int *&end_points, double *&distances
		, const std::string &graph_dir, const std::string &collection_file, double tv, double te, double alpha, double beta) {

	ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> env;
	int start;

	setup_environment(graph_dir, collection_file, env, tv, te, alpha, beta);
	std::call_once(init_flag, init_matrix, std::ref(distances), env.graph_ids().second);
//	std::cout << "tv: " << tv << " te: " << te << " alpha: " << alpha << " beta: " << beta << " graph_dir: " << graph_dir << " collection: " << collection_file << "\n";
	if (thread_id == 0) {
		start = 0;
	} else {
		start = end_points[thread_id - 1];
	}

	for (int g_id = start; g_id != end_points[thread_id]; g_id++) {
		for (int h_id = 0; h_id != g_id; h_id++) {
			env.run_method(g_id, h_id);
			distances[int(g_id*env.graph_ids().second + h_id)] = env.get_lower_bound(g_id, h_id);
			distances[int(h_id*env.graph_ids().second + g_id)] = distances[int(g_id*env.graph_ids().second + h_id)];
			env.delete_results(g_id, h_id);
		}
	}

}


void compute_hed_parallel(std::string graph_dir, std::string collection_file, double tv, double te, double alpha, double beta, int thread_num) {

	std::thread threads[thread_num];
	double *distances;
	int *end_points;

	ged::GEDEnv<ged::GXLNodeID, ged::GXLLabel, ged::GXLLabel> env;
	setup_environment(graph_dir, collection_file, env, tv, te, alpha, beta);
	end_points = compute_end_points(thread_num, env.graph_ids().second);

	for (int i = 0; i < thread_num; i++) {
		threads[i] = std::thread(compute_hed, i, std::ref(end_points), std::ref(distances), std::ref(graph_dir), std::ref(collection_file), tv, te, alpha, beta);
	}

	for (std::thread &thread : threads) {
		thread.join();
	}

	// first line of CSV
	std::cout << " , ";
	for (ged::GEDGraph::GraphID g_id = env.graph_ids().first; g_id != env.graph_ids().second; g_id++) {
		std::cout << env.get_graph_name(g_id) << ", ";
	}
	std::cout << "\n";
	for (ged::GEDGraph::GraphID g_id = env.graph_ids().first; g_id != env.graph_ids().second; g_id++) {
		std::cout << env.get_graph_name(g_id) << ", ";
		for (ged::GEDGraph::GraphID h_id = env.graph_ids().first; h_id != env.graph_ids().second; h_id++) {
			std::cout << distances[int(g_id*env.graph_ids().second + h_id)] << ", ";
		}
		std::cout << "\n";
	}
}

void print_usage(char *file) {
	std::cerr << "usage: " << file << " [-h] [--tv TV] [--te TE] [--alpha ALPHA] [--beta BETA] [--threads THREADS] graph_dir collection_file\n\n";
	std::cerr << "Compute graph edit distances in parallel.\n\n";
	std::cerr << "positional arguments:\n";
	std::cerr << "  graph_dir\t\tRoot folder for the collection XML file.\n";
	std::cerr << "  collection_file\tXML file containing the pathes to the GXL files.\n\n";
	std::cerr << "optional arguments:\n";
	std::cerr << "  -h, --help\t\tshow this help message and exit.\n";
	std::cerr << "  --tv TV\t\tCost for node insertion/deletion (default: 1).\n";
	std::cerr << "  --te TE\t\tCost for edge insertion/deletion (default: 1).\n";
	std::cerr << "  --alpha ALPHA\t\tWeight indicating importance of node substitution cost over edge substitution cost when computing overall substituion cost (default=0.5).\n";
	std::cerr << "  --beta BETA\t\tWeight indicating importance of x coordinate over y coordinate when computing node substitution costs (default=0.5).\n";
	std::cerr << "  --threads THREADS\tNumber of threads to use for computing the edit distances (default: 20).\n";

	exit(1);
}

int main(int argc, char* argv[]) {

	int opt= 0;
	int threads = 20;
	double tv = 1.0, te = 1.0, alpha = 0.5, beta = 0.5;

	static struct option long_options[] = {
		{"tv",   required_argument, 0,  'v' },
		{"te",   required_argument, 0,  'e' },
		{"alpha",   required_argument, 0,  'a' },
		{"beta",   required_argument, 0,  'b' },
		{"threads",   required_argument, 0,  't' },
		{"help",   no_argument, 0,  'h' },
		{0,           0,                 0,  0   }
	};

	int long_index =0;
	while ((opt = getopt_long(argc, argv,"hv:e:a:b:t:", long_options, &long_index )) != -1) {
		switch (opt) {
			case 'h' :
				print_usage(argv[0]);
				break;
			case 'v':
				tv = std::stod(optarg);
				break;
			case 'e':
				te = std::stod(optarg);
				break;
			case 'a':
				alpha = std::stod(optarg);
				break;
			case 'b':
				beta = std::stod(optarg);
				break;
			case 't':
				threads = atoi(optarg);
				break;
			case '?':
				break;
			default:
				print_usage(argv[0]);
		}
	 }

	if (argc - optind == 2) {
		//std::cout << "tv: " << tv << " te: " << te << " alpha: " << alpha << " beta: " << beta << " threads: " << threads << " graph_dir: " << argv[optind++] << " collection: " << argv[optind++] << "\n";
		compute_hed_parallel(std::string(argv[optind]), std::string(argv[optind+1]), tv, te, alpha, beta, threads);
	} else {
		print_usage(argv[0]);
	}
}
