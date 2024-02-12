#pragma once


class BVHNode : public Hittable {
public:
    Hittable *left;
    Hittable *right;
    AABB bbox;

    BVHNode(const HittableList) {
        // Todo
    }

    __host__ __device__ bool hit(const Ray &ray, const Interval &interval, HitRecord &record) const override {
        if (!bbox.hit(ray, interval))
            return false;

        bool hit_left = left->hit(ray, interval, record);
        bool hit_right = right->hit(ray, Interval(interval.min_, hit_left ? record.t : interval.max_), record);

        return hit_left || hit_right;
    }


    [[nodiscard]] AABB bounding_box() const {
        return bbox;
    }
};