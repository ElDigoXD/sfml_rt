#pragma once


#include <limits>
#include <random>
#include "third-party/pcg_random/pcg_random.hpp"

constexpr static const double infinity = std::numeric_limits<double>::infinity();
namespace Random {
    class Random {
    public:
        pcg32_fast gen;
        std::uniform_real_distribution<double> a;
        static Random *instance;

        Random() {
            gen = pcg32_fast(1);
        }

    public:
        static Random *get_instance();


        double _double() { return a(gen); }
    };

    Random *Random::instance = nullptr;

    Random *Random::get_instance() {
        if (instance == nullptr)
            instance = new Random();
        return instance;
    }


    static double generate_canonical() {
        return std::rand() / (RAND_MAX + 1.0); // NOLINT(*-msc50-cpp)
        return Random::get_instance()->_double();
    }

    static double _double() { return generate_canonical(); }

    static double _double(double min, double max) { return min + (max - min) * _double(); }
}
