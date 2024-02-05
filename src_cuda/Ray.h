#pragma once

#include "Vec3.h"

class Ray {

private:
    Point3 orig;
    Vec3 dir;

public:
    __host__ __device__ Ray() {};

    __host__ __device__ Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

    [[nodiscard]] __host__ __device__ Point3 origin() const { return orig; }

    [[nodiscard]] __host__ __device__ Vec3 direction() const { return dir; }

    [[nodiscard]] __host__ __device__ Point3 at(double t) const {
        return orig + t * dir;

    }
};