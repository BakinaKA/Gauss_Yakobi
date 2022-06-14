#pragma once
#include <vector>
#include <thread>

std::vector<std::vector<double>> get_special_matrix(int n);

std::vector<double> get_random_vector(int n);

bool check_eq_vectors(std::vector<double>& v1, std::vector<double>& v2);

static void calculate_x(std::vector<std::vector<double>>& a,
	std::vector<double>& f, std::vector<double>& x_cur,
	std::vector<double>& x_new, int left, int right, double eps,
	std::atomic<bool>& achieve_accuracy);

std::vector<double> Gauss_Yakobi_serial(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps = 1e-15);

std::vector<double> Gauss_Yakobi_on_thread(std::vector<std::vector<double>>& a,
	std::vector<double>& f, std::size_t n_threads = 8, double eps = 1e-15);

std::vector<double> Gauss_Yakobi_omp(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps = 1e-15);

std::vector<double> Gauss_Yakobi_tbb(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps = 1e-15);
