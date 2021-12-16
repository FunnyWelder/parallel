#include <cstdlib>
#include <thread>
#include <vector>
#include <barrier>

#if defined(__GNUC__) && __GNUC__ <= 10
namespace std {
    constexpr size_t hardware_constructive_interference_size = 64u;
    constexpr size_t hardware_destructive_interference_size = 64u;
}
#endif

auto ceil_div(auto x, auto y)
{
    return (x + x - 1) / y;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero)
{
	unsigned T = get_num_threads();
	struct reduction_partial_result_t
	{
		alignas(std::hardware_destructive_interference_size) ElementType value;
	};
	static auto reduction_partial_results =
		std::vector<reduction_partial_result_t>(T, reduction_partial_result_t{ zero });
	constexpr std::size_t k = 2;
	std::barrier<> bar{ T };

	auto thread_proc = [=, &bar](unsigned t)
	{
		auto K = ceil_div(n, k);
		std::size_t Mt = K / T, it1 = K % T;
		if (t < it1)
			it1 = ++Mt * t;
		else
			it1 += Mt * t;
		it1 *= k;
		std::size_t mt = Mt * k;
		auto it2 = it1 + mt;
		ElementType accum = zero;
		for (std::size_t i = it1; i < it2; ++i)
			accum = f(accum, V[i]);
		reduction_partial_results[t].value = accum;

		std::size_t s = 1;
		while (s < T)
		{
			bar.arrive_and_wait();
			if ((t % (s * k)) == 0 && s + t < T)
			{
				reduction_partial_results[t].value = f(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
				s *= k;
			}
		}
	};

	std::vector<std::thread> threads;
	for (unsigned t = 1; t < T; ++t)
		threads.emplace_back(thread_proc, t);
	thread_proc(0);
	for (auto& thread : threads)
		thread.join();

	return reduction_partial_results[0].value;
}


#include <type_traits>

template <class ElementType, class UnaryFn, class BinaryFn>
requires (
	std::is_invocable_r_v<ElementType, UnaryFn, ElementType>&&
	std::is_invocable_r_v<ElementType, BinaryFn, ElementType, ElementType>
	)
	ElementType reduce_range(ElementType a, ElementType b, std::size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
	unsigned T = get_num_threads();

	struct reduction_partial_result_t
	{
		alignas(std::hardware_destructive_interference_size) ElementType value;
	};
	static auto reduction_partial_results =
		std::vector<reduction_partial_result_t>(std::thread::hardware_concurrency(), reduction_partial_result_t{ zero });
	constexpr std::size_t k = 2;
	std::barrier<> bar{ (std::ptrdiff_t)T };

	auto thread_proc = [=, &bar](unsigned t)
	{
		auto K = ceil_div(n, k);
		double dx = (b - a) / n;
		std::size_t Mt = K / T, it1 = K % T;
		if (t < it1)
			it1 = ++Mt * t;
		else
			it1 += Mt * t;
		it1 *= k;
		std::size_t mt = Mt * k;
		auto it2 = it1 + mt;
		ElementType accum = zero;
		for (std::size_t i = it1; i < it2; ++i)
			accum = reduce_2(accum, get(a + i * dx));
		reduction_partial_results[t].value = accum;
		for (std::size_t s = 1u, s_next = 2u; s < T; s = s_next, s_next += s_next) //assume: k = 2
		{
			bar.arrive_and_wait();
			if ((t % s_next) == 0 && s + t < T)
				reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
		}
	};

	std::vector<std::thread> threads;
	for (unsigned t = 1; t < T; ++t)
		threads.emplace_back(thread_proc, t);
	thread_proc(0);
	for (auto& thread : threads)
		thread.join();

	return reduction_partial_results[0].value;
}