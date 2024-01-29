#pragma once

#include <cmath>
#include "utils.h"

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

    Vec3() {} // NOLINT(*-pro-type-member-init)

    Vec3(double x, double y, double z) : e{x, y, z} {} // NOLINT(*-pro-type-member-init)


    Vec3 operator-() const { return {-x, -y, -z}; }

    double operator[](int i) const { return e[i]; }

    double &operator[](int i) { return e[i]; }

    Vec3 &operator+=(const Vec3 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3 &operator*=(double t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    Vec3 &operator/=(double t) {
        return *this *= 1 / t;
    }

    [[nodiscard]] double length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] double length_squared() const {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] bool is_near_zero() const {
        return (fabs(x) < 1e-8) && (fabs(y) < 1e-8) && (fabs(z) < 1e-8);
    }

    static Vec3 random() { return {Random::_double(), Random::_double(), Random::_double()}; }

    static Vec3 random(double min, double max) {
        return {Random::_double(min, max), Random::_double(min, max), Random::_double(min, max)};
    }


    Vec3 normalize();
};


// Utility
Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

Vec3 operator*(const Vec3 &a, const Vec3 &b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }

Vec3 operator*(const Vec3 &a, double t) { return {a.x * t, a.y * t, a.z * t}; }

Vec3 operator*(double t, const Vec3 &a) { return a * t; }

Vec3 operator/(const Vec3 &a, double t) { return (1 / t) * a; }

double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 unit_vector(Vec3 v) { return v / v.length(); }

Vec3 Vec3::normalize() { return unit_vector(*this); }

Vec3 random_in_unit_sphere() {
    while (true) {
        auto p = Vec3::random(-1, 1);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

Vec3 random_in_unit_disk() {
    while (true) {
        auto p = Vec3{Random::_double(-1, 1), Random::_double(-1, 1), 0};
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

Vec3 random_unit_vector() { return random_in_unit_sphere().normalize(); }

Vec3 random_on_hemisphere(const Vec3 &normal) {
    Vec3 vec = random_unit_vector();
    return dot(vec, normal) > 0.0 ? vec : -vec;
}

Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

// R′⊥=(η/η′)(R+(−R⋅n)n)
// R′∥=−√(1−|R′⊥|^(2))n
Vec3 refract(const Vec3 &unit_v, const Vec3 &n, double refraction_ratio) {
    auto cos_theta = std::min(dot(-unit_v, n), 1.0);
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

Color to_gamma_color(const Color &color) {
    return {std::sqrt(color.r),
            std::sqrt(color.g),
            std::sqrt(color.b)};
}

void to_float_array(const Color &color, float *array) {
    array[0] = static_cast<float>(color.x);
    array[1] = static_cast<float>(color.y);
    array[2] = static_cast<float>(color.z);
}

Color from_float_array(const float *array) {
    return {array[0], array[1], array[2]};
}

namespace Colors {
    static Color red = Color(1, 0, 0);
    static Color green = Color(0, 1, 0);
    static Color blue = Color(0, 0, 1);
    static Color white = Color(1, 1, 1);
    static Color black = Color(0, 0, 0);
    static Color blue_sky = Color(0.5, 0.7, 1);
}


