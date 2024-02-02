#pragma once

#include "Hittable.h"
#include "Ray.h"


class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const = 0;
};


class Lambertian : public Material {
public:
    Color albedo;

    explicit Lambertian(const Color &_albedo) : albedo(_albedo) {}

    bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
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

    bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        auto reflect_direction = reflect(ray_in.direction().normalize(), record.normal);

        scattered_ray = Ray(record.p, reflect_direction + fuzz * random_unit_vector());
        attenuation = albedo;
        return (dot(scattered_ray.direction(), record.normal) > 0);
    }
};

class Dielectric : public Material {
public:
    double refraction_index;

    explicit Dielectric(double _refraction_index) : refraction_index(_refraction_index) {}

    bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        attenuation = Colors::white;
        double refraction_ratio = record.front_face ? (1.0 / refraction_index) : refraction_index;

        Vec3 unit_direction = ray_in.direction().normalize();
        double cos_theta = std::min(dot(-unit_direction, record.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        Vec3 direction;
        if (refraction_ratio * sin_theta > 1 || reflectance(cos_theta, refraction_ratio) > Random::_double()) {
            direction = reflect(ray_in.direction().normalize(), record.normal);
        } else {
            direction = refract(ray_in.direction().normalize(), record.normal, refraction_ratio);

        }

        scattered_ray = Ray(record.p, direction);
        return true;
    }

private:
    static double reflectance(double cosine, double _refraction_index) {
        // Schlick's approximation
        auto r0 = (1 - _refraction_index) / (1 + _refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class Normals : public Material {
public:
    Normals() = default;

    bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        auto scatter_direction = record.normal + random_unit_vector();
        if (scatter_direction.is_near_zero())
            scatter_direction = record.normal;

        attenuation = 0.5 * (record.normal + Colors::white);
        scattered_ray = {record.p, scatter_direction};
        return true;
    };
};
