#pragma once

#include "Hittable.h"
#include "Material.h"

class HittableList : public Hittable {
public:
    int list_size{};
    Hittable **list = nullptr;

    __host__ __device__ HittableList() {};

    __host__ __device__ explicit HittableList(Hittable **l, int n) : list(l), list_size(n) {
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
};