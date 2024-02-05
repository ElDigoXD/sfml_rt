#pragma once


class Material;

class HitRecord {
public:
    Point3 p;
    Vec3 normal;
    Material *material{};
    double t{};
    bool front_face{};

    __host__ __device__ void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class Material {
public:
    virtual ~Material() = default;

    __host__ virtual bool
    scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const = 0;

    __device__ virtual bool
    scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray, curandState *rand) const = 0;
};


class Lambertian : public Material {
public:
    Color albedo;

    __host__ __device__ explicit Lambertian(const Color &_albedo) : albedo(_albedo) {}

    __host__ bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        auto scatter_direction = record.normal + random_unit_vector();
        if (scatter_direction.is_near_zero())
            scatter_direction = record.normal;

        scattered_ray = Ray(record.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    __device__ bool scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray,
                            curandState *rand) const override {
        auto scatter_direction = record.normal + random_unit_vector(rand);
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

    __host__ __device__ Metal(const Color &_albedo, double _fuzz) : albedo(_albedo), fuzz(_fuzz < 1 ? _fuzz : 1) {}

    __host__ bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        auto reflect_direction = reflect(ray_in.direction().normalize(), record.normal);

        scattered_ray = Ray(record.p, reflect_direction + fuzz * random_unit_vector());
        attenuation = albedo;
        return (dot(scattered_ray.direction(), record.normal) > 0);
    }

    __device__ bool scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray,
                            curandState *rand) const override {
        auto reflect_direction = reflect(ray_in.direction().normalize(), record.normal);

        scattered_ray = Ray(record.p, reflect_direction + fuzz * random_unit_vector(rand));
        attenuation = albedo;
        return (dot(scattered_ray.direction(), record.normal) > 0);
    }
};

class Dielectric : public Material {
public:
    double refraction_index;

    __host__ __device__ explicit Dielectric(double _refraction_index) : refraction_index(_refraction_index) {}

    __host__ bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        attenuation = Colors::white();
        double refraction_ratio = record.front_face ? (1.0 / refraction_index) : refraction_index;

        Vec3 unit_direction = ray_in.direction().normalize();
        double cos_theta = min(dot(-unit_direction, record.normal), 1.0);
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

    __device__ bool scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray,
                            curandState *rand) const override {

        attenuation = Colors::white();
        double refraction_ratio = record.front_face ? (1.0 / refraction_index) : refraction_index;

        Vec3 unit_direction = ray_in.direction().normalize();
        double cos_theta = min(dot(-unit_direction, record.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        Vec3 direction;
        if (refraction_ratio * sin_theta > 1 || reflectance(cos_theta, refraction_ratio) > Random::_double(rand)) {
            direction = reflect(ray_in.direction().normalize(), record.normal);
        } else {
            direction = refract(ray_in.direction().normalize(), record.normal, refraction_ratio);

        }

        scattered_ray = Ray(record.p, direction);
        return true;
    }

private:
    __host__ __device__ static double reflectance(double cosine, double _refraction_index) {
        // Schlick's approximation

        auto r0 = (1 - _refraction_index) / (1 + _refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class Normals : public Material {
public:
    __host__ __device__ Normals() {};

    __host__ bool scatter(Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray) const override {
        auto scatter_direction = record.normal + random_unit_vector();
        if (scatter_direction.is_near_zero())
            scatter_direction = record.normal;

        attenuation = 0.5 * (record.normal + Colors::white());
        scattered_ray = {record.p, scatter_direction};
        return true;
    };

    __device__ bool scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &scattered_ray,
                            curandState *rand) const override {
        auto scatter_direction = record.normal + random_unit_vector(rand);
        if (scatter_direction.is_near_zero())
            scatter_direction = record.normal;

        attenuation = 0.5 * (record.normal + Colors::white());
        scattered_ray = {record.p, scatter_direction};
        return true;
    };
};
