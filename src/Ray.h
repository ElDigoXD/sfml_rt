#pragma once

#include "Vec3.h"

class Ray {

private:
    Point3 orig;
    Vec3 dir;

public:
    Ray() = default;

    Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

    [[nodiscard]] Point3 origin() const { return orig; }

    [[nodiscard]] Vec3 direction() const { return dir; }

    Point3 at(double t) const {
        return orig + t * dir;

    }
};