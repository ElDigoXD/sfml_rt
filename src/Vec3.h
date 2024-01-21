#pragma once

#include <cmath>

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


    Vec3 unit();
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

inline Vec3 Vec3::unit() { return unit_vector(*this); }

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

#endif