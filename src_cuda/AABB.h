#pragma once
#include "Interval.h"
#include "Ray.h"
#include "Vec3.h"

class AABB {
public:
    Interval x;
    Interval y;
    Interval z;
    [[nodiscard]] Interval index(int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }

    AABB() = default;

    AABB(const Interval &_x, const Interval &_y, const Interval &_z): x(_x), y(_y), z(_z) {
    }

    AABB(const Point3 &a, const Point3 &b) {
        x = Interval(std::min(a.x, b.x), std::max(a.x, b.x));
        y = Interval(std::min(a.y, b.y), std::max(a.y, b.y));
        z = Interval(std::min(a.z, b.z), std::max(a.z, b.z));
    }

    AABB(const AABB& a, const AABB& b) {
        x = {a.x, b.x};
        y = {a.y, b.y};
        z = {a.z, b.z};
    }

    bool hit_0(const Ray &r, Interval ray_t) const {
        for (int i = 0; i < 3; ++i) {
            auto t0 = std::min(
                (index(i).min - r.origin()[i]) / r.direction()[i],
                (index(i).max - r.origin()[i]) / r.direction()[i]);
            auto t1 = std::max(
                (index(i).min - r.origin()[i]) / r.direction()[i],
                (index(i).max - r.origin()[i]) / r.direction()[i]);
            ray_t.min = std::max(t0, ray_t.min);
            ray_t.max = std::min(t1, ray_t.max);
            if (ray_t.max <= ray_t.min) {
                return false;
            }
        }
        return true;
    }

    bool hit(const Ray &r, Interval ray_t) const {
        for (int i = 0; i < 3; i++) {
            const auto invD = 1 / r.direction()[i];
            const auto orig = r.origin()[i];

            auto t0 = (index(i).min - orig) * invD;
            auto t1 = (index(i).max - orig) * invD;

            if (invD < 0)
                std::swap(t0, t1);

            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }
};
