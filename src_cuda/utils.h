#pragma once


#include <limits>
#include <cstdint>
#include <random>
#include <numbers>
#include <memory>
#include <iostream>


#ifdef __CUDACC__
#include "cuda.h"
#include "curand.h"
#include "curand_globals.h"
#include "curand_kernel.h"

#define CUDA

#define GPU __host__ __device__

#define CU(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#else
#define GPU
typedef int curandGenerator_t;
typedef int curandState;
#endif
constexpr static const double infinity = std::numeric_limits<double>::infinity();


template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

GPU double degrees_to_radians(double degrees) {
    return degrees * std::numbers::pi / 180.0;
}


namespace Random {

    static thread_local unsigned int rng_state = std::rand();

    static unsigned int rand_pcg() {
        unsigned int state = rng_state;
        rng_state = rng_state * 747796405u + 2891336453u;
        unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }


    static inline unsigned int XOrShift32() {
        unsigned int x = rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng_state = x;
        return x;
    }

    double generate_canonical() {
        //return std::rand() / (RAND_MAX + 1.0); // NOLINT(*-msc50-cpp)
        return rand_pcg() / (std::numeric_limits<u_int32_t>::max() + 1.0);
        //return XOrShift32() / (std::numeric_limits<uint32_t>::max() + 1.0);
    }

    double _double() { return generate_canonical(); }

    GPU double _double(curandState *rand) {
#ifdef __CUDA_ARCH__
#ifdef CUDA
        return -curand_uniform_double(rand) + 1;
#else
        return 0;
#endif
#else
        return generate_canonical();
#endif
    }

    GPU double _double(double min, double max, curandState *rand) {
        return min + (max - min) * _double(rand);
    }
}