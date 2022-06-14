#include <ctime> 
#include <intrin.h>
#include <gtest/gtest.h>
#include "../src/Gauss_Yakobi.h"

TEST(Gauss_Yakobi_testsuit, test_data_generation) {
    int n = 150;
    std::vector<std::vector<double>> matrix = get_special_matrix(n);
	bool f = true;
	for (int i = 0; i < n; ++i) {
		double sum = 0;
		for (int j = 0; j < n; ++j) {
			sum += std::fabs(matrix[i][j]);
		}
		f = sum > 2 * std::fabs(matrix[i][i]);
		if (f) {
			break;
		}
	}
	EXPECT_FALSE(f);
}

typedef testing::TestWithParam<int> Params;
TEST_P(Params, test_perfomance)
{
	int n = GetParam();
	std::cout << "Data size: " << n << "\n";
	std::vector<std::vector<double>>  a = get_special_matrix(n);
	std::vector<double> f = get_random_vector(n);

	auto start = __rdtsc();
	std::vector<double> x_s = Gauss_Yakobi_serial(a, f);
	auto serial_time = __rdtsc() - start;
	std::cout << "Serial time: " << serial_time << "\n";

	start = __rdtsc();
	std::vector<double> x_th = Gauss_Yakobi_on_thread(a, f);
	auto thread_time = __rdtsc() - start;

	start = __rdtsc();
	std::vector<double> x_omp = Gauss_Yakobi_omp(a, f);
	auto omp_time = __rdtsc() - start;

	start = __rdtsc();
	std::vector<double> x_tbb = Gauss_Yakobi_omp(a, f);
	auto tbb_time = __rdtsc() - start;

	std::cout << "Thread time: " << thread_time << "\n";
	std::cout << "OpenMP time: " << omp_time << "\n";
	std::cout << "TBB time: " << tbb_time << "\n";

	std::cout << "Serial/Thread: " << (double)serial_time / thread_time << "\n";
	std::cout << "Serial/OpenMP: " << (double)serial_time / omp_time << "\n";
	std::cout << "Serial/TBB: " << (double)serial_time / tbb_time << "\n";

	EXPECT_TRUE(check_eq_vectors(x_s, x_th));
	EXPECT_TRUE(check_eq_vectors(x_s, x_omp));
	EXPECT_TRUE(check_eq_vectors(x_s, x_tbb));
}

INSTANTIATE_TEST_SUITE_P(GaussYakobiTestsuite, Params,
	testing::Values(200, 1000, 5000, 10000, 15000, 20000, 25000, 30000));

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}