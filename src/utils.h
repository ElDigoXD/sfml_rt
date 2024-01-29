#pragma once


#include <limits>
#include <cstdint>
#include <random>
#include "third-party/pcg_random/pcg_random.hpp"

constexpr static const double infinity = std::numeric_limits<double>::infinity();

double degrees_to_radians(double degrees) {
    return degrees * std::numbers::pi / 180.0;
}


namespace Random {
    class Random {
    public:
        pcg32_fast gen;
        std::uniform_real_distribution<double> a;
        thread_local static Random *instance;

        Random() {
            gen = pcg32_fast(1);
        }

    public:
        static Random *get_instance();


        double _double() { return a(gen); }
    };

    thread_local Random *Random::instance = nullptr;

    Random *Random::get_instance() {
        if (instance == nullptr)
            instance = new Random();
        return instance;
    }

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

    static double generate_canonical() {
#ifdef _WIN32
        return std::rand() / (RAND_MAX + 1.0); // NOLINT(*-msc50-cpp)
#endif
#ifdef __linux__
#endif
        return rand_pcg() / (std::numeric_limits<uint32_t>::max() + 1.0);
        return XOrShift32() / (std::numeric_limits<uint32_t>::max() + 1.0);
    }

    static double _double() { return generate_canonical(); }

    static double _double(double min, double max) { return min + (max - min) * _double(); }
}

#ifdef IS_SFML
namespace ImGui {
    bool SliderDouble(const char *label, double *v, double v_min, double v_max, const char *format = "%.3f",
                      ImGuiSliderFlags flags = 0) {
        return SliderScalar(label, ImGuiDataType_Double, v, &v_min, &v_max, format, flags);
    }
}
#endif