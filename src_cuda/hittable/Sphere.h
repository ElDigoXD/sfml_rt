#pragma once


#include "../material/Material.h"
#include "Hittable.h"
#include "../Vec3.h"

class Sphere : public Hittable {
private:
    double radius;
public:
    Material *material;
    Point3 center;

    GPU Sphere() {};

    GPU Sphere(Point3 _center, double _radius, Material *_material) : center(_center), radius(_radius),
                                                                                      material(_material) {
        auto radius_vec = Vec3(radius, radius, radius);
        bbox = AABB(center - radius_vec, center + radius_vec);
    }

    GPU bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
        //if (!bbox.hit(ray, interval)) return false;
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
            root = (-half_b + sqrt_discriminant) / a;
            // if second root is not in range
            if (!interval.surrounds(root)) {
                return false;
            }
        }

        record.t = root;
        record.p = ray.at(root);
        auto outward_normal = (record.p - center) / radius;
        record.set_face_normal(ray, outward_normal);
        record.material = material;

        return true;
    }

    GPU bool hit(const Ray &ray) const override {
        Vec3 oc = ray.origin() - center;
        auto a = ray.direction().length_squared();
        auto c = oc.length_squared() - radius * radius;
        if (c <= 0) return true;
        auto b = dot(oc, ray.direction());
        if (b > 0) return false;
        auto discriminant = b * b - a * c;
        return discriminant > 0;
    }
};