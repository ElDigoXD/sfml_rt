#pragma once

#include "utils.h"

class Interval {
public:
    double min_, max_;

    GPU Interval() : min_(+infinity), max_(-infinity) {}

    GPU Interval(double _min, double _max) : min_(_min), max_(_max) {}

    GPU Interval(const Interval& a, const Interval& b): min_(fmin(a.min_, b.min_)), max_(fmax(a.max_, b.max_)){}

    [[nodiscard]] GPU bool contains(double x) const {
        return min_ <= x && x <= max_;
    }

    [[nodiscard]] GPU bool surrounds(double x) const {
        return min_ < x && x < max_;
    }

    [[nodiscard]] GPU double clamp(double x) const {
        if (x < min_) return min_;
        if (x > max_) return max_;
        return x;
    }
    GPU Interval expand(double delta) const{
        auto padding = delta/2;
        return {min_-padding, max_+padding};
    }


    static const Interval empty, universe;

};

static const Interval empty = Interval(+infinity, -infinity);
static const Interval _all = Interval(+infinity, -infinity);