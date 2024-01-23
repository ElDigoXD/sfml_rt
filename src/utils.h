#pragma once


#include <limits>
#include <random>

constexpr static const double infinity = std::numeric_limits<double>::infinity();
namespace Random {

    std::random_device rd;
    std::default_random_engine gen(rd());

    double generate_canonical() {
        return std::generate_canonical<double, 10>(gen);
    }
}
