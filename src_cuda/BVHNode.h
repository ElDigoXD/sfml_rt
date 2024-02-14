#pragma once

#include "thrust/host_vector.h"
#include "thrust/universal_vector.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"

class BVHNode : public Hittable {
public:
    Hittable *left;
    Hittable *right;

    __host__ __device__ BVHNode(Hittable **list, int list_size, curandState *rand) :
            BVHNode(list, 0,
                    list_size, rand) {}

    __host__ __device__ BVHNode(Hittable **list, int start, int end, curandState *rand) {
        auto objects = list;

        auto axis = Random::_double(rand);
        auto comparator = (axis < 0.33) ? box_x_compare
                                        : (axis < 0.66) ? box_y_compare
                                                        : box_z_compare;

        auto object_span = end - start;
        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start + 1])) {
                left = objects[start];
                right = objects[start + 1];
            } else {
                left = objects[start + 1];
                right = objects[start];
            }
        } else {
            std::sort(objects + start, objects + end, comparator);

            auto mid = start + object_span / 2;

            left = new BVHNode(objects, start, mid, rand);
            right = new BVHNode(objects, mid, end, rand);
        }
        bbox = AABB(left->bounding_box(), right->bounding_box());
    }

    __host__ __device__ bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
        if (!bbox.hit(ray, interval))
            return false;

        bool hit_left = left->hit(ray, interval, record);
        bool hit_right = right->hit(ray, Interval(interval.min_, hit_left ? record.t : interval.max_), record);

        return hit_left || hit_right;
    }

    static bool box_compare(const Hittable *a, const Hittable *b, int axis_index) {
        return a->bounding_box().index(axis_index).min_ < b->bounding_box().index(axis_index).min_;
    }

    static bool box_x_compare(const Hittable *a, const Hittable *b) { return box_compare(a, b, 0); }

    static bool box_y_compare(const Hittable *a, const Hittable *b) { return box_compare(a, b, 1); }

    static bool box_z_compare(const Hittable *a, const Hittable *b) { return box_compare(a, b, 2); }


    __host__ __device__ void dev_sort(Hittable **a, Hittable **b,
                                      bool (*comparator)(Hittable *a, Hittable *b)) {
        for (int i = 0; a + i < b; i++) {
            for (int j = 0; a + j < b; j++) {
                if (!((*comparator)(*(a + i), *(a + j)))) {
                     Hittable *tmp = *(a + i);
                    *(a + i) = *(a + j);
                    *(a + j) = tmp;
                }
            }
        }
    }
};