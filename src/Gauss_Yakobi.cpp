#include <omp.h>
#include <iostream>
#include <tbb/tbb.h>
#include "Gauss_Yakobi.h"

std::vector<std::vector<double>> get_special_matrix(int n) {
	if (n < 0) {
		throw("Wrong matrix size for generation");
	}
	std::srand(time(0));
	std::vector<std::vector<double>> matrix(n);
	for (int i = 0; i < n; ++i) {
		matrix[i].resize(n);
		for (int j = 0; j < n; ++j) {
			matrix[i][j] = (static_cast<double>(std::rand() % 100 + ((i == j) ? 100 * (n - 1) : 0)) +
				static_cast<double>(std::rand()) / RAND_MAX)
				* ((std::rand() % 2 == 0) ? -1 : 1);
		}
	}
	return matrix;
}

std::vector<double> get_random_vector(int n) {
	if (n < 0) {
		throw("Wrong vector size for generation");
	}
	std::srand(time(0));
	std::vector<double> vector(n);
	for (int i = 0; i < n; ++i) {
		vector[i] = (static_cast<double>(std::rand() % 100 * n) +
			static_cast<double>(std::rand()) / RAND_MAX)
			* ((std::rand() % 2 == 0) ? -1 : 1);
	}
	return vector;
}

bool check_eq_vectors(std::vector<double>& v1, std::vector<double>& v2) {
	for (int i = 0; i < v1.size(); ++i) {
		if (std::fabs(v1[i] - v2[i]) > 1e-1)
			return false;
	}
	return true;
}

std::vector<double> Gauss_Yakobi_serial(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps) {
	int n = a.size();
	std::vector<double> x_cur(n, 0);
	std::vector<double> x_new(n);
	bool achieve_accuracy;
	do
	{
		achieve_accuracy = true;
		for (int i = 0; i < n; ++i) {
			double sum = 0;
			for (int j = 0; j < n; ++j) {
				sum += a[i][j] * x_cur[j];
			}
			x_new[i] = (f[i] - sum + a[i][i] * x_cur[i]) / a[i][i];
			if (achieve_accuracy)
				achieve_accuracy &= (std::abs(x_new[i] - x_cur[i]) < eps);
		}
		std::swap(x_new, x_cur);
	} while (!achieve_accuracy);
	return x_new;
}

void calculate_x(std::vector<std::vector<double>>& a, std::vector<double>& f,
	std::vector<double>& x_cur, std::vector<double>& x_new,
	int left, int right, double eps, std::atomic<bool>& achieve_accuracy) {
	int n = a.size();
	for (int i = left; i < right; ++i) {
		double sum = 0;
		for (int j = 0; j < n; ++j) {
			sum += a[i][j] * x_cur[j];
		}
		x_new[i] = (f[i] - sum + a[i][i] * x_cur[i]) / a[i][i];
		if (achieve_accuracy)
			achieve_accuracy = achieve_accuracy & (std::abs(x_new[i] - x_cur[i]) < eps);
	}
}

std::vector<double> Gauss_Yakobi_on_thread(std::vector<std::vector<double>>& a,
	std::vector<double>& f, std::size_t n_threads, double eps) {
	int n = a.size();
	std::vector<double> x_cur(n, 0);
	std::vector<double> x_new(n);
	int block_size = n / n_threads;
	int offset = n % n_threads;
	std::vector<std::thread> threads(n_threads);
	std::atomic<bool> achieve_accuracy;
	do
	{
		achieve_accuracy = true;
		for (int i = 0; i < n_threads; ++i) {
			threads[i] = std::thread(calculate_x, std::ref(a), std::ref(f),
				std::ref(x_cur), std::ref(x_new),
				block_size * i, block_size * (i + 1) + ((i == 0) ? offset : 0),
				eps, std::ref(achieve_accuracy));
		}

		for (auto& th : threads) {
			th.join();
		}

		std::swap(x_new, x_cur);
	} while (!achieve_accuracy);
	return x_new;
}

std::vector<double> Gauss_Yakobi_omp(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps) {
	int n = a.size();
	std::vector<double> x_cur(n, 0);
	std::vector<double> x_new(n);
	bool achieve_accuracy;
	do
	{
		achieve_accuracy = true;
#pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			double sum = 0;
			for (int j = 0; j < n; ++j) {
				sum += a[i][j] * x_cur[j];
			}
			x_new[i] = (f[i] - sum + a[i][i] * x_cur[i]) / a[i][i];
#pragma omp critical 
			{
				if (achieve_accuracy)
					achieve_accuracy &= (std::abs(x_new[i] - x_cur[i]) < eps);
			}
		}
		std::swap(x_new, x_cur);
	} while (!achieve_accuracy);
	return x_new;
}

std::vector<double> Gauss_Yakobi_tbb(std::vector<std::vector<double>>& a,
	std::vector<double>& f, double eps) {
	int n = a.size();
	std::vector<double> x_cur(n, 0);
	std::vector<double> x_new(n);
	tbb::atomic<bool> achieve_accuracy;
	do
	{
		achieve_accuracy = true;
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
			[&](const tbb::blocked_range<size_t>& r) {
				int begin = r.begin(), end = r.end();
				for (int i = 0; i < n; ++i) {
					double sum = 0;
					for (int j = 0; j < n; ++j) {
						sum += a[i][j] * x_cur[j];
					}
					x_new[i] = (f[i] - sum + a[i][i] * x_cur[i]) / a[i][i];
					if (achieve_accuracy)
						achieve_accuracy = achieve_accuracy
						& (std::abs(x_new[i] - x_cur[i]) < eps);
				}
			});
		std::swap(x_new, x_cur);
	} while (!achieve_accuracy);
	return x_new;
}