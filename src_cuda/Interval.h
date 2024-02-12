#pragma once

#include "utils.h"

class Interval {
public:
    double min_, max_;

    __host__ __device__ Interval() : min_(+infinity), max_(-infinity) {}

    __host__ __device__ Interval(double _min, double _max) : min_(_min), max_(_max) {}

    __host__ __device__ Interval(const Interval& a, const Interval& b): min_(fmin(a.min_, b.min_)), max_(fmax(a.max_, b.max_)){}

    [[nodiscard]] __host__ __device__ bool contains(double x) const {
        return min_ <= x && x <= max_;
    }

    [[nodiscard]] __host__ __device__ bool surrounds(double x) const {
        return min_ < x && x < max_;
    }

    [[nodiscard]] __host__ __device__ double clamp(double x) const {
        if (x < min_) return min_;
        if (x > max_) return max_;
        return x;
    }
    __host__ __device__ Interval expand(double delta) const{
        auto padding = delta/2;
        return {min_-padding, max_+padding};
    }


    static const Interval empty, universe;

};

static const Interval empty = Interval(+infinity, -infinity);
static const Interval _all = Interval(+infinity, -infinity);