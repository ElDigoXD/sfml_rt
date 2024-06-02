#pragma once

#include "Vec3.h"

class Ray {

private:
    Point3 orig;
    Vec3 dir;

public:
    GPU Ray() {};

    GPU Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

    [[nodiscard]] GPU Point3 origin() const { return orig; }

    [[nodiscard]] GPU Vec3 direction() const { return dir; }

    [[nodiscard]] GPU Point3 at(double t) const {
        return orig + t * dir;

    }
};