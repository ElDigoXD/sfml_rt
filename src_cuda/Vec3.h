#pragma once

#include <cmath>

#ifdef CUDA
#include <curand_kernel.h>
#endif

#include "utils.h"
#include "Interval.h"

class Vec3 {
public:
    union {
        struct {
            double x;
            double y;
            double z;
        };
        struct {
            double r;
            double g;
            double b;
        };
        double e[3]{0, 0, 0};
    };

    GPU Vec3() {} // NOLINT(*-pro-type-member-init)

    GPU Vec3(double x, double y, double z) : e{x, y, z} {} // NOLINT(*-pro-type-member-init)


    GPU Vec3 operator-() const { return {-x, -y, -z}; }

    GPU double operator[](int i) const { return e[i]; }

    GPU double &operator[](int i) { return e[i]; }

    GPU Vec3 &operator+=(const Vec3 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    GPU Vec3 &operator*=(double t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    GPU Vec3 &operator*=(Vec3 b) {
        x *= b.x;
        y *= b.y;
        z *= b.z;
        return *this;
    }

    GPU Vec3 &operator/=(double t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] GPU double length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] GPU double length_squared() const {
        return x * x + y * y + z * z;
    }

    [[nodiscard]]GPU  bool is_near_zero() const {
        return (fabs(x) < 1e-8) && (fabs(y) < 1e-8) && (fabs(z) < 1e-8);
    }

    [[nodiscard]]GPU  bool is_near_zero(double epsilon) const {
        return (fabs(x) < epsilon) && (fabs(y) < epsilon) && (fabs(z) < epsilon);
    }

    [[nodiscard]]GPU  Vec3 abs() const {
        return {std::abs(x), std::abs(y), std::abs(z)};
    }

    [[nodiscard]] GPU Vec3 clamp(double a, double b) const {
        Interval i(a, b);
        return {i.clamp(x), i.clamp(y), i.clamp(z)};
    }

    GPU static Vec3 random(curandState *rand) {
        return {Random::_double(rand), Random::_double(rand), Random::_double(rand)};
    }

    GPU static Vec3 random(double min, double max, curandState *rand) {
        return {Random::_double(min, max, rand), Random::_double(min, max, rand), Random::_double(min, max, rand)};
    }

    GPU Vec3 normalize();
};


// Utility
GPU Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

GPU Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

GPU Vec3 operator*(const Vec3 &a, const Vec3 &b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }

GPU bool operator==(const Vec3 &a, const Vec3 &b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

GPU bool operator!=(const Vec3 &a, const Vec3 &b) { return !(a == b); }

GPU Vec3 operator*(const Vec3 &a, double t) { return {a.x * t, a.y * t, a.z * t}; }

GPU Vec3 operator*(double t, const Vec3 &a) { return a * t; }

GPU Vec3 operator/(const Vec3 &a, double t) { return (1 / t) * a; }

GPU double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

GPU Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

GPU Vec3 unit_vector(Vec3 v) { return v / v.length(); }

GPU Vec3 Vec3::normalize() { return unit_vector(*this); }


GPU Vec3 random_in_unit_sphere(curandState *rand) {
    Vec3 p;
    do {
        p = Vec3::random(-1, 1, rand);
    } while (p.length_squared() >= 1);
    return p;

}

GPU Vec3 random_in_unit_disk(curandState *rand) {
    while (true) {
        auto p = Vec3{Random::_double(-1, 1, rand), Random::_double(-1, 1, rand), 0};
        if (p.length_squared() < 1) {
            return p;
        }
    }
}


GPU Vec3 random_unit_vector(curandState *rand) { return random_in_unit_sphere(rand).normalize(); }

/*
__host__  Vec3 random_on_hemisphere(const Vec3 &normal) {
    Vec3 vec = random_unit_vector();
    return dot(vec, normal) > 0.0 ? vec : -vec;
}
*/
GPU Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

// R′⊥=(η/η′)(R+(−R⋅n)n)
// R′∥=−√(1−|R′⊥|^(2))n
GPU Vec3 refract(const Vec3 &unit_v, const Vec3 &n, double refraction_ratio) {
    auto cos_theta = fmin(dot(-unit_v, n), 1.0);
    auto ray_out_perp = refraction_ratio * (unit_v + cos_theta * n);
    auto ray_out_par = -std::sqrt(std::abs(1 - ray_out_perp.length_squared())) * n;
    return ray_out_perp + ray_out_par;
}
//

using Color = Vec3;
using Point3 = Vec3;


#ifdef IS_SFML

#include <SFML/Graphics.hpp>

sf::Color to_sf_color(Color color) {
    return {static_cast<sf::Uint8>(color.r * 255),
            static_cast<sf::Uint8>(color.g * 255),
            static_cast<sf::Uint8>(color.b * 255),
            255};
}

sf::Color to_sf_gamma_color(Color color) {
    return {static_cast<sf::Uint8>(  std::sqrt(color.r) * 255),
            static_cast<sf::Uint8>(std::sqrt(color.g) * 255),
            static_cast<sf::Uint8>( std::sqrt(color.b) * 255),
            255};
}

#endif

GPU Color to_gamma_color(const Color &color) {
    return {std::sqrt(color.r),
            std::sqrt(color.g),
            std::sqrt(color.b)};
}

GPU void to_float_array(const Color &color, float *array) {
    array[0] = static_cast<float>(color.x);
    array[1] = static_cast<float>(color.y);
    array[2] = static_cast<float>(color.z);
}

GPU Color from_float_array(const float *array) {
    return {array[0], array[1], array[2]};
}

GPU Color from_hex_code(const uint hex_code) {
    return Color((hex_code >> 16) & 0xFF, (hex_code >> 8) & 0xFF, hex_code & 0xFF) / 255;
}

namespace Colors {
    GPU Color red() { return {1, 0, 0}; };

    GPU Color green() { return {0, 1, 0}; };

    GPU Color blue() { return {0, 0, 1}; };

    GPU Color white() { return {1, 1, 1}; };

    GPU Color black() { return {0, 0, 0}; };

    GPU Color blue_sky() { return {0.5, 0.7, 1}; };
}


