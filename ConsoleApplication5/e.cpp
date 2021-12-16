#include <omp.h>
#include <thread>

static unsigned g_num_threads = std::thread::hardware_concurrency();

void set_num_threads(unsigned T)
{
	g_num_threads = T;
	omp_set_num_threads(T);
}

unsigned get_num_threads()
{
	return g_num_threads;
}