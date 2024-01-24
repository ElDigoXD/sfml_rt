#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "memory"
#include "vector"
#include "Interval.h"

class Material;

class HitRecord {
public:
    Point3 p;
    Vec3 normal;
    std::shared_ptr<Material> material;
    double t{};
    bool front_face{};

    void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    virtual ~Hittable() = default;

    virtual bool hit(const Ray &ray, Interval interval, HitRecord &record) const = 0;
};

class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    HittableList() {}

    HittableList(std::shared_ptr<Hittable> object) { add(object); }

    void add(std::shared_ptr<Hittable> object) {
        objects.push_back(object);
    }

    bool hit(const Ray &ray, Interval interval, HitRecord &record) const override {
        HitRecord temp_record;
        bool hit = false;
        auto closest = interval.max;

        for (const auto &object: objects) {
            if (object->hit(ray, Interval(interval.min, closest), temp_record)) {
                hit = true;
                closest = temp_record.t;
                record = temp_record;
            }
        }
        return hit;
    }
};