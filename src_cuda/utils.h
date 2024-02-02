#pragma once


#include <limits>
#include <cstdint>
#include <random>
#include <numbers>
#include <memory>

constexpr static const double infinity = std::numeric_limits<double>::infinity();


template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

double degrees_to_radians(double degrees) {
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

    bool SliderDouble3(const char *label, double v[3], double v_min, double v_max, const char *format = "%.3f",
                       ImGuiSliderFlags flags = 0) {
        return SliderScalarN(label, ImGuiDataType_Double, v, 3, &v_min, &v_max, format, flags);
    }

    bool DragDouble(const char *label, double v[3], float speed, double v_min, double v_max, const char *format = "%.3f",
                       ImGuiSliderFlags flags = 0) {
        return DragScalar(label, ImGuiDataType_Double, v, speed, &v_min, &v_max, format, flags);
    }
    bool DragDouble3(const char *label, double v[3], float speed, double v_min, double v_max, const char *format = "%.3f",
                     ImGuiSliderFlags flags = 0) {
        return DragScalarN(label, ImGuiDataType_Double, v, 3, speed, &v_min, &v_max, format, flags);
    }
}
#endif