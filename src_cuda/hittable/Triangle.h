#pragma once

#include "Hittable.h"
#include "../Vec3.h"
#include "../Ray.h"


class Triangle : public Hittable {
public:
    Material *material;
    const Vec3 v[3];

    __host__ __device__ Triangle(const Vec3 _v[3], Material *_material) :
            v{_v[0], _v[1], _v[2]}, material(_material) {
        bbox = AABB(AABB(v[0], v[1]), AABB(v[0], v[2]));
    }

    __host__ __device__ Triangle(const Vec3 v0, const Vec3 v1, const Vec3 v2, Material *_material) :
            v{v0, v1, v2}, material(_material) {
        bbox = AABB(AABB(v[0], v[1]), AABB(v[0], v[2]));
    }

    __host__ __device__ bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {

        auto v0v1 = v[1] - v[0];
        auto v0v2 = v[2] - v[0];

        auto n = cross(v0v1, v0v2);
        auto area2 = n.length();

        // Check if the ray and plane are parallel.
        auto n_dot_ray_dir = dot(n, ray.direction());
        if (abs(n_dot_ray_dir) < 1e-8)
            return false;

        auto d = -dot(n, v[0]);
        auto t = -(dot(n, ray.origin()) + d) / n_dot_ray_dir;
        if (!interval.surrounds(t))
            return false;

        auto p = ray.at(t);

        auto edge0 = v[1] - v[0];
        auto vp0 = p - v[0];
        auto c = cross(edge0, vp0);
        if (dot(n, c) < 0)
            return false;

        auto edge1 = v[2] - v[1];
        auto vp1 = p - v[1];
        c = cross(edge1, vp1);
        if (dot(n, c) < 0)
            return false;

        auto edge2 = v[0] - v[2];
        auto vp2 = p - v[2];
        c = cross(edge2, vp2);
        if (dot(n, c) < 0)
            return false;


        record.t = t;
        record.p = p;
        record.material = material;
        record.set_face_normal(ray, n.normalize());
        return true;
    }

    __host__ __device__ bool hit(const Ray &ray) const override {
        auto v0v1 = v[1] - v[0];
        auto v0v2 = v[2] - v[0];

        auto n = cross(v0v1, v0v2);

        // Check if the ray and plane are parallel.
        auto n_dot_ray_dir = dot(n, ray.direction());
        if (abs(n_dot_ray_dir) < 1e-8)
            return false;

        auto d = -dot(n, v[0]);
        auto t = -(dot(n, ray.origin()) + d) / n_dot_ray_dir;
        if (!Interval(0.0000001, infinity).surrounds(t))
            return false;

        auto p = ray.at(t);

        auto edge0 = v[1] - v[0];
        auto vp0 = p - v[0];
        auto c = cross(edge0, vp0);
        if (dot(n, c) < 0)
            return false;

        auto edge1 = v[2] - v[1];
        auto vp1 = p - v[1];
        c = cross(edge1, vp1);
        if (dot(n, c) < 0)
            return false;

        auto edge2 = v[0] - v[2];
        auto vp2 = p - v[2];
        c = cross(edge2, vp2);
        if (dot(n, c) < 0)
            return false;
        return true;
    }
};