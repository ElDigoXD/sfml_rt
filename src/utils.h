#pragma once


#include <limits>
#include <random>

constexpr static const double infinity = std::numeric_limits<double>::infinity();
namespace Random {

    static std::random_device rd;
    static std::default_random_engine gen(rd());

    inline double generate_canonical() {
        return std::generate_canonical<double, 10>(gen);
    }

    inline double _double() { return generate_canonical(); }

    inline double _double(double min, double max) { return min + (max - min) * _double(); }
}
