#pragma once

#include "Hittable.h"
#include "../material/Material.h"

class HittableList : public Hittable {
public:
    int list_size;
    Hittable **list;

    //GPU HittableList() {};

    GPU explicit HittableList(Hittable **l, int n) : list(l), list_size(n) {
        if (list_size == 0) return;
        bbox = list[0]->bounding_box();
        for (int i = 0; i < list_size; i++) {
            bbox = AABB(bbox, list[i]->bounding_box());
        }
    }

    GPU bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
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

    [[nodiscard]] GPU bool hit(const Ray &ray) const override {
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(ray)) {
                return true;
            }
        }
        return false;
    }
};