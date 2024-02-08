#pragma once


#include "Material.h"
#include "Hittable.h"
#include "Vec3.h"

class Sphere: public Hittable {
private:
    double radius;
    AABB bbox;
public:
    Material *material;
    Point3 center;

    __host__ __device__ Sphere() {};

    __host__ __device__ Sphere(Point3 _center, double _radius, Material *_material) : center(_center), radius(_radius),
                                                                  material(_material) {
        auto radius_vec = Vec3(radius, radius, radius);
        bbox = AABB(center - radius_vec, center + radius_vec);
    }

    __host__ __device__ bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
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

    // [[nodiscard]]__host__ __device__ AABB bounding_box() const override {
    //     return bbox;
    // }
};