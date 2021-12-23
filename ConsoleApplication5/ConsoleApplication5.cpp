#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <iterator> 
#include <algorithm>

#include "struct_mapping/struct_mapping.h"

void set_num_threads(unsigned T);

unsigned get_num_threads();

struct Schet {
    std::string name;
    std::vector<double> time_ms;
    std::vector<double> speed;
};

#ifndef __cplusplus
#ifndef _MSC_VER
#include <cstdalign>
#define _aligned_free free
#else
#define alignas(x) __declspec(align(x))
#define aligned_alloc(al, sz) _aligned_malloc((sz),(al))
#endif
#else
#define aligned_alloc(al, sz) _aligned_malloc((sz),(al))
#ifndef _MSC_VER
#define _aligned_free free
#endif
#endif


#define STEPS 50000000

double func(double x)
{
    return x * x;
}

typedef double (*f_t) (double);

typedef struct partial_sum_t_
{
    alignas(64) double result;
} partial_sum_t;

typedef struct experiment_result
{
    double result;
    double time_ms;
} experiment_result;

typedef double (*I_t)(double, double, f_t);

experiment_result run_experiment(I_t I)
{
    double t0 = omp_get_wtime();
    double R = I(-1, 1, func);
    double t1 = omp_get_wtime();
    return { R, t1 - t0 };
}

void show_experiment_results_cli(I_t I, std::string name)
{
    double T1;
    std::cout << name << '\n';
    printf("%10s\t%10s\t%10s\n", "Result", "Time_ms", "Speed");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment(I);
        if(T == 1)
            T1 = R.time_ms;
        printf("%10g\t%10g\t%10g\n", R.result, R.time_ms, T1 / R.time_ms);
    };
    std::cout << '\n';
};

#include <filesystem>
void show_experiment_results_json(I_t I, std::string name, std::string file_path)
{
    using namespace std;

    ofstream out;

    if (!filesystem::exists(file_path)) {
        error_code ec;
        filesystem::create_directories(file_path, ec);
    }

    std::string full_file_path = file_path + "\\" + name + ".json";
    out.open(full_file_path);
    if (out.is_open())
    {
        struct_mapping::reg(&Schet::name, "name");
        struct_mapping::reg(&Schet::time_ms, "time_ms");
        struct_mapping::reg(&Schet::speed, "speed");

        Schet schet;

        schet.name = name;

        double T1;
        for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
            experiment_result R;
            set_num_threads(T);
            R = run_experiment(I);
            schet.time_ms.push_back(R.time_ms);
            if (T == 1)
                T1 = R.time_ms;
            double speed = T1 / R.time_ms;
            schet.speed.push_back(speed);
        }

        std::ostringstream json_data;
        struct_mapping::map_struct_to_json(schet, json_data, "  ");


        out << json_data.str() << std::endl;
    }
};




double integrate_crit(double a, double b, f_t f)
{
    double Result = 0, dx = (b - a) / STEPS;
    #pragma omp parallel shared(Result)
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num(), T = (unsigned) get_num_threads();
        for (unsigned i = t; i < STEPS; i += T)
        {
            R += f(i * dx + a);
        }
        #pragma omp critical
        Result += R;
    }
    return Result * dx;
}

double integrate_cpp_mtx(double a, double b, f_t f)
{
    using namespace std;

    unsigned T = get_num_threads();
    vector <thread> threads;
    mutex mtx;
    double Result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; ++t)
        threads.emplace_back([=, &Result, &mtx]()
            {
                double R = 0;
                for (unsigned i = t; i < STEPS; i += T)
                {
                    R += f(i * dx + a);
                }

                {
                    scoped_lock lock{ mtx };
                    Result += R;
                }
            });

    for (auto& thr : threads)
        thr.join();

    return Result * dx;
}

double integrate_false_sharing(double a, double b, f_t f)
{
    unsigned T;
    double Result = 0, dx = (b - a) / STEPS;
    double* Accum;
    #pragma omp parallel shared(Accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        #pragma omp single
        {
            T = (unsigned) get_num_threads();
            Accum = (double*)calloc(T, sizeof(double));
        }

        for (unsigned i = t; i < STEPS; i += T)
            Accum[t] += f(dx * i + a);
    }

    for (unsigned i = 0; i < T; i++)
        Result += Accum[i];

    free(Accum);

    return Result * dx;
}

double integrate_aligned(double a, double b, f_t f)
{
    unsigned T;
    double Result = 0, dx = (b - a) / STEPS;
    partial_sum_t* Accum;
    #pragma omp parallel shared(Accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        #pragma omp single
        {
            T = (unsigned) get_num_threads();
            Accum = (partial_sum_t*) aligned_alloc(alignof(partial_sum_t), T * sizeof(partial_sum_t_));
        }

        Accum[t].result = 0;
        for (unsigned i = t; i < STEPS; i += T)
            Accum[t].result += f(dx * i + a);
    }

    for (unsigned i = 0; i < T; i++)
        Result += Accum[i].result;

    _aligned_free(Accum);

    return Result * dx;
}

double integrate_reduction(double a, double b, f_t f)
{
    double Result = 0, dx = (b - a) / STEPS;

    #pragma omp parallel for reduction(+:Result)
    for (int i = 0; i < STEPS; i++)
    {
        Result += f(dx * i + a);
    }

    return Result * dx;
}

double integrate_cpp(double a, double b, f_t f)
{
    double Result = 0, dx = (b - a) / STEPS;
    using namespace std;
    unsigned T = get_num_threads();
    auto vec = vector(T, partial_sum_t{ 0.0 });
    vector <thread> threads;

    auto threads_proc = [=, &vec](auto t) {
        for (unsigned i = t; i < STEPS; i += T)
            vec[t].result += f(dx * i + a);
    };

    for (unsigned t = 1; t < T; t++)
        threads.emplace_back(threads_proc,t);

    threads_proc(0);

    for (auto &thread:threads)
        thread.join();

    for (auto &elem:vec)
        Result += elem.result;

    return Result * dx;
}

double integrate_omp_for(double a, double b, f_t f)
{
    double Result = 0, dx = (b - a) / STEPS;
    int i;

    #pragma omp parallel for shared (Result) 
    for (i = 0; i < STEPS; ++i)
    {
        double val = f(dx * i + a);
        #pragma omp atomic 
        Result += val;
    }

    return Result * dx;
}

double integrate_cpp_reduction(double a, double b, f_t f)
{
    using namespace std;
    unsigned T = get_num_threads();
    double dx = (b - a) / STEPS;
    vector <thread> threads;
    atomic <double> Result{ 0.0 };
    auto threads_proc = [dx, &Result, f, a, T](auto t)
    {
        
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        Result += R;
    };

    for (unsigned t = 1; t < T; t++)
        threads.emplace_back(threads_proc, t);

    threads_proc(0);

    for (auto& thread : threads)
        thread.join();

    return Result * dx;
}

//class Iterator
//{
//private:
//    f_t f;
//    double dx, a;
//    unsigned i = 0;
//
//public:
//    typedef double value_type, * pointer, & reference;
//    using iterator_category = std::random_access_iterator_tag;
//    //Iterator() = default; 
//
//    Iterator(f_t fun, double delta_x, double x0, unsigned index) :f(fun), dx(delta_x), a(x0), i(index) {}
//
//    double value() const {
//        return f(a + i * dx);
//    }
//
//    auto operator*() const {
//        return this->value(); 
//    }
//
//    Iterator& operator++()
//    {
//        ++i;
//        return *this;
//    }
//
//    Iterator operator++(int)
//    {
//        auto old = *this;
//        ++* this;
//        return old;
//    }
//
//    bool operator==(const Iterator& other) const
//    {
//        return i == other.i;
//    }
//};
//
//float integrate_cpp_reduce_1(double a, double b, f_t f)
//{
//    double dx = (b - a) / STEPS;
//    return std::reduce(Iterator(f, dx, a, 0), Iterator(f, dx, a, STEPS)) * dx;
//}
//
//#include "reduce_par.h" 
//float integrate_cpp_reduce_2(double a, double b, f_t f)
//{
//    double dx = (b - a) / STEPS;
//    return reduce_par_2([f, dx](double x, double y) {return x + y; }, f, (double)a, (double)b, (double)dx, 0.0) * dx;
//}

#include "reduction.h" 

double integrate_reduce(double a, double b, f_t f) {
    return reduce_range(a, b, STEPS, f, [](auto x, auto y) {return x + y;}, 0.0) * ((b-a)/STEPS);
}


class barrier
{
    bool lock_oddity = false;
    unsigned T;
    const unsigned T_max;
    std::condition_variable cv;
    std::mutex mtx;

public:
    void arrive_and_wait()
    {
        std::unique_lock lock(mtx);
        if (--T == 0)
        {
            lock_oddity = !lock_oddity;
            T = T_max;
            cv.notify_all();
        }
        else {
            auto my_lock = lock_oddity;
            while (my_lock == lock_oddity)
                cv.wait(lock);
        }
    }
};

    
int main()
{
    show_experiment_results_json(integrate_crit, "integrate_crit", ".\\graphics");
    show_experiment_results_json(integrate_cpp_mtx, "integrate_cpp_mtx", ".\\graphics");
    show_experiment_results_json(integrate_false_sharing, "integrate_false_sharing", ".\\graphics");
    show_experiment_results_json(integrate_aligned, "integrate_aligned", ".\\graphics");
    show_experiment_results_json(integrate_reduction, "integrate_reduction", ".\\graphics");
    show_experiment_results_json(integrate_cpp, "integrate_cpp", ".\\graphics");
    show_experiment_results_json(integrate_omp_for, "integrate_omp_for", ".\\graphics");
    show_experiment_results_json(integrate_cpp_reduction, "integrate_cpp_reduction", ".\\graphics");
    show_experiment_results_json(integrate_reduce, "integrate_reduce", ".\\graphics");

    //show_experiment_results_cli(integrate_crit, "integrate_crit");
    //show_experiment_results_cli(integrate_cpp_mtx, "integrate_cpp_mtx");
    //show_experiment_results_cli(integrate_false_sharing, "integrate_false_sharing");
    //show_experiment_results_cli(integrate_aligned, "integrate_aligned");
    //show_experiment_results_cli(integrate_reduction, "integrate_reduction");
    //show_experiment_results_cli(integrate_cpp, "integrate_cpp");
    //show_experiment_results_cli(integrate_omp_for, "integrate_omp_for");
    //show_experiment_results_cli(integrate_cpp_reduction, "integrate_cpp_reduction");
    //show_experiment_results_cli(integrate_reduce, "integrate_reduce");

    return 0;
}