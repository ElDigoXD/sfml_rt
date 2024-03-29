#pragma once

#include "utils.h"

class Interval {
public:
    double min, max;

    Interval() : min(+infinity), max(-infinity) {}

    Interval(double _min, double _max) : min(_min), max(_max) {}

    Interval(const Interval& a, const Interval& b):min(std::min(a.min, b.min)), max(std::max(a.max, b.max)){}

    [[nodiscard]] bool contains(double x) const {
        return min <= x && x <= max;
    }

    [[nodiscard]] bool surrounds(double x) const {
        return min < x && x < max;
    }

    [[nodiscard]] double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const Interval empty, universe;

};

static const Interval empty = Interval(+infinity, -infinity);
static const Interval all = Interval(+infinity, -infinity);