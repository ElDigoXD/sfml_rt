#pragma once

#include "Hittable.h"
#include "Material.h"

class HittableList : public Hittable {
public:
    int list_size;
    Hittable **list;

    //__host__ __device__ HittableList() {};

    __host__ __device__ explicit HittableList(Hittable **l, int n) : list(l), list_size(n) {
        if (list_size == 0) return;
        bbox = list[0]->bounding_box();
        for (int i = 0; i < list_size; i++) {
            bbox = AABB(bbox, list[i]->bounding_box());
        }
    }

    __host__ __device__ bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
        HitRecord temp_record;
        bool hit = false;
        auto in = Interval(interval);

        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(ray, in, temp_record)) {
                hit = true;
                in.max_ = temp_record.t;
                record = temp_record;
            }
        }
        return hit;
    }

    [[nodiscard]] __host__ __device__ bool hit2(const Ray &ray) const {
        const Interval interval(0.0001, infinity);
        HitRecord record;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(ray, interval, record)) {
                return true;
            }
        }
        return false;
    }
};