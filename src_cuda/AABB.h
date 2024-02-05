#pragma once

#include "Interval.h"
#include "Ray.h"
#include "Vec3.h"

class AABB {
public:
    Interval x;
    Interval y;
    Interval z;

    [[nodiscard]] __host__ __device__ Interval index(int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }

    __host__ __device__ AABB() {};

    __host__ __device__ AABB(const Interval &_x, const Interval &_y, const Interval &_z) : x(_x), y(_y), z(_z) {
    }

    __host__ __device__ AABB(const Point3 &a, const Point3 &b) {
        x = Interval(min(a.x, b.x), max(a.x, b.x));
        y = Interval(min(a.y, b.y), max(a.y, b.y));
        z = Interval(min(a.z, b.z), max(a.z, b.z));
    }

    __host__ __device__ AABB(const AABB &a, const AABB &b) {
        x = {a.x, b.x};
        y = {a.y, b.y};
        z = {a.z, b.z};
    }

    [[nodiscard]] __host__ __device__ bool hit_0(const Ray &r, Interval ray_t) const {
        for (int i = 0; i < 3; ++i) {
            auto t0 = min(
                    (index(i).min_ - r.origin()[i]) / r.direction()[i],
                    (index(i).max_ - r.origin()[i]) / r.direction()[i]);
            auto t1 = max(
                    (index(i).min_ - r.origin()[i]) / r.direction()[i],
                    (index(i).max_ - r.origin()[i]) / r.direction()[i]);
            ray_t.min_ = max(t0, ray_t.min_);
            ray_t.max_ = min(t1, ray_t.max_);
            if (ray_t.max_ <= ray_t.min_) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] __host__ __device__ bool hit(const Ray &r, Interval ray_t) const {
        for (int i = 0; i < 3; i++) {
            const auto invD = 1 / r.direction()[i];
            const auto orig = r.origin()[i];

            auto t0 = (index(i).min_ - orig) * invD;
            auto t1 = (index(i).max_ - orig) * invD;

            if (invD < 0) {
                auto aux = t0;
                t0 = t1;
                t1 = aux;
            }

            if (t0 > ray_t.min_) ray_t.min_ = t0;
            if (t1 < ray_t.max_) ray_t.max_ = t1;

            if (ray_t.max_ <= ray_t.min_)
                return false;
        }
        return true;
    }
};
