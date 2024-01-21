#pragma once

#include "Hittable.h"

class Sphere : public Hittable {
private:
    Point3 center;
    double radius;
public:
    Sphere(Point3 _center, double _radious) : center(_center), radius(_radious) {}

    bool hit(const Ray &ray, Interval interval, HitRecord &record) const override {
        Vec3 oc = ray.origin() - center;
        auto a = ray.direction().length_squared();
        auto half_b = dot(oc, ray.direction());
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;

        auto sqrt_discriminant = sqrt(discriminant);

        // - root is the first
        // if first root is not in range
        auto root = (-half_b - sqrt_discriminant) / a;
        if (!interval.surrounds(root)) {
            root = (-half_b - sqrt_discriminant) / a;
            // if second root is not in range
            if (!interval.surrounds(root)) {
                return false;
            }
        }

        record.t = root;
        record.p = ray.at(root);
        auto outward_normal = (record.p - center) / radius;
        record.set_face_normal(ray, outward_normal);

        return true;
    }
};