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

    bool is_near_zero() const {
        auto s = 1e-8;
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }

    inline static Vec3 random() { return {Random::_double(), Random::_double(), Random::_double()}; }

    inline static Vec3 random(double min, double max) {
        return {Random::_double(min, max), Random::_double(min, max), Random::_double(min, max)};
    }


    inline Vec3 normalize();
};


// Utility
inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

inline Vec3 operator*(const Vec3 &a, const Vec3 &b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }

inline Vec3 operator*(const Vec3 &a, double t) { return {a.x * t, a.y * t, a.z * t}; }

inline Vec3 operator*(double t, const Vec3 &a) { return a * t; }

inline Vec3 operator/(const Vec3 &a, double t) { return (1 / t) * a; }

inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline Vec3 unit_vector(Vec3 v) { return v / v.length(); }

inline Vec3 Vec3::normalize() { return unit_vector(*this); }

inline Vec3 random_in_unit_sphere() {
    while (true) {
        auto p = Vec3::random(-1, 1);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

inline Vec3 random_unit_vector() { return random_in_unit_sphere().normalize(); }

inline Vec3 random_on_hemisphere(const Vec3 &normal) {
    Vec3 vec = random_unit_vector();
    return dot(vec, normal) > 0.0 ? vec : -vec;
}

inline const Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

//

using Color = Vec3;
using Point3 = Vec3;


//#ifdef IS_SFML

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
//#endif

namespace Colors {
    static Color red = Color(1, 0, 0);
    static Color green = Color(0, 1, 0);
    static Color blue = Color(0, 0, 1);
    static Color white = Color(1, 1, 1);
    static Color black = Color(0, 0, 0);
    static Color blue_sky = Color(0.5, 0.7, 1);
}


