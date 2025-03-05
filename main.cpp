#include <cmath>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <thread>
#include <vector>

void print_vector(double *x, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";
}

// computes y <- A @ x
void gemv_omp(const double *A, const double *x, double *y, int m, int n)
{
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            y[i] = 0.;
            for (int j = 0; j < n; j++) {
                y[i] += A[i * n + j] * x[j];
            }
        }
    }
}

// computes y <- y + alpha * x
void axpy_omp(double *y, double alpha, const double *x, int n) {
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub =
            (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            y[i] += alpha * x[i];
        }
    }
}

// computes (x, y)
double dot_omp(const double *x, const double *y, int n) {
    double r = 0.;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub =
            (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double br = 0.;
        for (int i = lb; i <= ub; i++) {
            br += x[i] * y[i];
        }
#pragma omp atomic
        r += br;
    }

    return r;
}

// computes ||x||_2
double norm_omp(const double *x, int n) {
    return std::sqrt(dot_omp(x, x, n));
}

// computes x <- A^-1 b by simple iterations method
int solve_simple_iters(const double *A, const double *b, double *x, double tau, double eps,
                       int n) {
    double b_norm = norm_omp(b, n);

    double *buf = (double *)malloc(n * sizeof(double));
    if (buf == nullptr) {
        return 1;
    }

    do {
        gemv_omp(A, x, buf, n, n);
        axpy_omp(buf, -1., b, n);
        axpy_omp(x, -tau, buf, n);
    } while (norm_omp(buf, n) >= eps * b_norm);

    free(buf);

    return 0;
}

// integrates a given function
double integrate(double (*f)(double), double a, double b, int n) {
    double r = 0.;
    double h = (b - a) / (double)n;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub =
            (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double br = 0.;
        for (int i = lb; i <= ub; i++) {
            br += f(a + h * ((double)i + 0.5));
        }

#pragma omp atomic
        r += br * h;
    }

    return r;
}

double fn(double x) {
    return std::sin(x);
}


// init data for simple iterations method test
void init_omp(double *A, double *x0, double *b, int n) {
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub =
            (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            x0[i] = 0.;
            b[i] = n + 1;
            for (int j = 0; j < n; j++) {
                A[i * n + j] = i == j ? 2. : 1.;
            }
        }
    }
}

template <typename T> int measure_time(T fn) {
    auto start_time = std::chrono::high_resolution_clock::now();

    fn();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    return time / std::chrono::milliseconds(1);
}

template <typename T> void benchmark_threads(T fn) {
    int t;
    for (int threads: {1,2,4,7,8,16,20,40}) {
        omp_set_num_threads(threads);

        int tries = 10;
        int sum = 0;

        for (int i = 0; i < tries; i++) {
            sum += measure_time(fn);
        }

        int time = sum / tries;

        std::cout << threads << " threads: ";
        std::cout.flush();

        if (threads == 1) {
            t = time;
        }

        std::cout << time << " ms, " << ((double)t / (double)time) << "x speedup" << std::endl;
    }
}

int main(int argc, char **argv) {

    for (int n : {20000, 40000}) {
        std::vector<double> A(n * n), x0(n), b(n), output(n);
        init_omp(A.data(), x0.data(), b.data(), n);
        std::cout << "matmul, " << n << "x" << n << ":"<< std::endl;

        benchmark_threads([&A, &b, &x0, n] {
            auto x = x0;
            gemv_omp(A.data(), b.data(), x.data(), n, n);
        });
    }

    for (int n : {20000, 40000}) {
        std::vector<double> A(n * n), x0(n), b(n), output(n);
        init_omp(A.data(), x0.data(), b.data(), n);

        std::cout << "solve, " << n << "x" << n << ":"<< std::endl;

        benchmark_threads([&A, &b, &x0, n] {
            auto x = x0;
            solve_simple_iters(A.data(), b.data(), x.data(), 1. / (double)n, 1e-5, n);
        });
    }

    for (int n : {40000000}) {
        std::cout << "integrate, " << n << " pts:" << std::endl;
        benchmark_threads([n] {
            integrate(fn, 0., 2 * 3.1415926, n);
        });
    }


    return 0;
}