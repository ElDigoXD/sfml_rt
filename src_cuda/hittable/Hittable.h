#pragma once

#include "../AABB.h"
#include "../Vec3.h"
#include "../Ray.h"
#include "memory"
#include "vector"
#include "../Interval.h"
#include "../material/Material.h"

class Hittable {
public:
    AABB bbox;

    GPU virtual bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const = 0;

    [[nodiscard]] GPU virtual AABB bounding_box() const final {
        return bbox;
    };

    GPU virtual bool hit(const Ray &ray) const = 0;
};
