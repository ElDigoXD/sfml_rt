#pragma once

#include "Hittable.h"
#include "Ray.h"


class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(Ray &ray_in, const HitRecord& record, Color &attenuation, Ray &scattered_ray) const = 0;
};


class Lambertian : public Material {
public:
    Color albedo;
    explicit Lambertian(const Color &_albedo) : albedo(_albedo) {}

    bool scatter(Ray &ray_in, const HitRecord& record, Color &attenuation, Ray &scattered_ray) const override {
        auto scatter_direction = record.normal + random_unit_vector();
        if (scatter_direction.is_near_zero())
            scatter_direction = record.normal;

        scattered_ray = Ray(record.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
public:
    Color albedo;
    double fuzz;
    Metal(const Color &_albedo, double _fuzz) : albedo(_albedo), fuzz(_fuzz < 1 ? _fuzz : 1) {}

    bool scatter(Ray &ray_in, const HitRecord& record, Color &attenuation, Ray &scattered_ray) const override {
        auto reflect_direction = reflect(ray_in.direction().normalize(), record.normal);

        scattered_ray = Ray(record.p, reflect_direction + fuzz * random_unit_vector());
        attenuation = albedo;
        return (dot(scattered_ray.direction(), record.normal) > 0);
    }
};