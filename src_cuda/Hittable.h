#pragma once

#include "AABB.h"
#include "Vec3.h"
#include "Ray.h"
#include "memory"
#include "vector"
#include "Interval.h"
#include "Material.h"

class Hittable {
public:
    __host__ __device__ virtual bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const = 0;

    [[nodiscard]] __host__ __device__ virtual AABB bounding_box() const = 0;
};
