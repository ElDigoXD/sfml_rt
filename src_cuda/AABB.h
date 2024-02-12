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
        x = Interval(fmin(a.x, b.x), fmax(a.x, b.x));
        y = Interval(fmin(a.y, b.y), fmax(a.y, b.y));
        z = Interval(fmin(a.z, b.z), fmax(a.z, b.z));
    }

    __host__ __device__ AABB(const AABB &a, const AABB &b) {
        x = {a.x, b.x};
        y = {a.y, b.y};
        z = {a.z, b.z};
    }

    [[nodiscard]] __host__ __device__ bool hit_0(const Ray &r, Interval ray_t) const {
        for (int i = 0; i < 3; ++i) {
            auto t0 = fmin(
                    (index(i).min_ - r.origin()[i]) / r.direction()[i],
                    (index(i).max_ - r.origin()[i]) / r.direction()[i]);
            auto t1 = fmax(
                    (index(i).min_ - r.origin()[i]) / r.direction()[i],
                    (index(i).max_ - r.origin()[i]) / r.direction()[i]);
            ray_t.min_ = fmax(t0, ray_t.min_);
            ray_t.max_ = fmin(t1, ray_t.max_);
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

    __host__ __device__ AABB pad() {
        auto delta = 0.0001;
        Interval new_x = (x.max_ - x.min_ >= delta) ? x : x.expand(delta);
        Interval new_y = (y.max_ - y.min_ >= delta) ? y : y.expand(delta);
        Interval new_z = (z.max_ - z.min_ >= delta) ? z : z.expand(delta);

        return {new_x, new_y, new_z};
    }
};
