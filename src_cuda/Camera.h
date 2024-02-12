#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "HittableList.h"
#include <iostream>
#include <random>
#include <cstring>

class Camera {
public:
    int image_width;
    int image_height;
    int samples_per_pixel;

    int max_depth;
    double vfov{90};

    Point3 look_from = Point3(0, 0, 0);
    Point3 look_at = Point3(0, 0, -1);

    Vec3 u, v, w;

    Point3 camera_center;

    double defocus_angle = 0;
    double focus_dist;

private:

    double viewport_width{};
    double viewport_height{};

    Vec3 pixel_delta_x;
    Vec3 pixel_delta_y;
    Vec3 subpixel_delta_x;
    Vec3 subpixel_delta_y;


    Vec3 viewport_x;
    Vec3 viewport_y;
    Vec3 defocus_disk_x;
    Vec3 defocus_disk_y;
    Vec3 viewport_upper_left;

    Point3 pixel_00_location;


public:
    void *operator new(size_t size) {
        return malloc(size);
    }


    __host__ void *operator new(size_t len, bool gpu) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    __host__ __device__ Camera() : Camera(600, 400) {}

    __host__ __device__ Camera(int _image_width, int _image_height) : Camera(_image_width, _image_height, 10, 10) {}

    __host__ __device__ Camera(int _image_width, int _image_height, int _samples_per_pixel, int _max_depth)
            : image_width(_image_width),
              image_height(_image_height),
              samples_per_pixel(_samples_per_pixel),
              max_depth(_max_depth) {
        update();
    }

    __host__ __device__ void update() { update(image_width, image_height); }

    __host__ __device__ void update(int width, int height) {
        image_width = width;
        image_height = height;

        camera_center = look_from;

        focus_dist = (look_from - look_at).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);

        viewport_height = 2 * h * focus_dist;
        viewport_width = viewport_height * (width * 1.0 / height);

        w = (look_from - look_at).normalize();
        u = cross(Vec3(0, 1, 0), w).normalize();
        v = cross(w, u);


        viewport_x = viewport_width * u;
        viewport_y = viewport_height * -v;

        pixel_delta_x = viewport_x / width;
        pixel_delta_y = viewport_y / height;

        viewport_upper_left = camera_center - (focus_dist * w) - viewport_x / 2 - viewport_y / 2;
        pixel_00_location = viewport_upper_left + 0.5 * (pixel_delta_x + pixel_delta_y);

        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_x = u * defocus_radius;
        defocus_disk_y = v * defocus_radius;
    }

    __host__ __device__ Ray get_random_ray_at(int i, int j, curandState *rand) const {
        auto pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
        auto pixel_sample = pixel_center + pixel_sample_square(rand);

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(rand);
        //auto ray_origin = camera_center;

        return Ray(ray_origin, pixel_sample - ray_origin);
    }

/*
    Ray get_random_ray_at_subpixel(int i, int j, int sub_i, int sub_j) {
        auto subpixel_center =
                viewport_upper_left + i * pixel_delta_x + j * pixel_delta_y + sub_i * subpixel_delta_x +
                sub_j * subpixel_delta_y;
        auto pixel_sample = subpixel_center + subpixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample();

        return Ray(ray_origin, pixel_sample - ray_origin);
    }
*/
    __host__ __device__ Ray get_ray_at(int i, int j) {
        auto pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
        auto ray_origin = camera_center;

        return Ray(ray_origin, pixel_center - ray_origin);
    }

    [[nodiscard]] __host__ __device__ Vec3 pixel_sample_square(curandState *rand) const {
        return (-0.5 + Random::_double(rand)) * pixel_delta_x +
               (-0.5 + Random::_double(rand)) * pixel_delta_y;
    }

    [[nodiscard]] Vec3 subpixel_sample_square() const {
        return (-0.5 + Random::_double()) * subpixel_delta_x +
               (-0.5 + Random::_double()) * subpixel_delta_y;
    }

    [[nodiscard]] __host__ __device__ Point3 defocus_disk_sample(curandState *rand) const {
        auto p = random_in_unit_disk(rand);
        return camera_center + p.x * defocus_disk_x + p.y * defocus_disk_y;
    }

    void render(unsigned char pixels[], HittableList **world) {
        for (int j = 0; j < image_height; ++j) {
            render_pixel_line(&pixels[j * image_width * 4], world, j);
        }
    }

    void render_pixel_line(unsigned char pixels[], HittableList **world, int line) const {

        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = Color(0, 0, 0);

            for (int sample = 0; sample < samples_per_pixel; ++sample) {

                Ray ray = get_random_ray_at(i, line, nullptr);
                pixel_color += ray_color(ray, max_depth, world, nullptr);
            }
            pixel_color /= samples_per_pixel;

            auto rgba_color = to_gamma_color(pixel_color);
            pixels[i * 4 + 0] = static_cast<unsigned char>(rgba_color.r * 255);
            pixels[i * 4 + 1] = static_cast<unsigned char>(rgba_color.g * 255);
            pixels[i * 4 + 2] = static_cast<unsigned char>(rgba_color.b * 255);
            pixels[i * 4 + 3] = 255;
        }
    }

    void render_color_line(Color pixels[], HittableList **world, int line) {
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = Color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                Ray ray = get_random_ray_at(i, line, nullptr);
                pixel_color += ray_color(ray, max_depth, world, nullptr);
            }
            pixel_color /= samples_per_pixel;
            pixels[i] = pixel_color;
        }
    }

    __host__ __device__ static Color lerp(Color start, Color end, double a) {
        return (1 - a) * start + a * end;
    }

    constexpr static const double infinity = std::numeric_limits<double>::infinity();

    Color ray_color_recursive(Ray &ray, int depth, const Hittable &world) {
        HitRecord record;

        if (depth <= 0)
            return (Colors::black());

        if (world.hit(ray, Interval(0.001, infinity), record)) {
            Ray scattered_ray;
            Color attenuation;
            if (record.material->scatter(ray, record, attenuation, scattered_ray, nullptr)) {
                return attenuation * ray_color_recursive(scattered_ray, depth - 1, world);
            };
            return Colors::black();
        }

        Vec3 unit_direction = ray.direction().normalize();
        auto a = 0.5 * (unit_direction.y + 1.0);
        return lerp(Colors::white(), Colors::blue_sky(), a);
    }

    Point3 light{0, 0, 0};
    Color light_color{0, 0, 0};
    union {
        struct {
            double diffuse_intensity;
            double specular_intensity;
            double sky_intensity;
        };
        double intensity[3]{0, 0.3, 1};
    };
    int shinyness = 1000;

    __host__ __device__ Color ray_color(const Ray &ray, int depth, HittableList **world, curandState *rand) const {
        HitRecord record;
        Color attenuation;
        Color sky_color = Colors::blue_sky();
        Color illumination_color{0, 0, 0};
        Ray cur_ray = ray;
        int cur_depth = depth;
        Ray scattered_ray;
        bool has_point_light = light != Color{0, 0, 0};

        while ((*world)->hit(cur_ray, Interval(0.001, infinity), record)) {
            // Ray does not escape, so it's represented as black
            if (cur_depth-- <= 0) break;

            auto light_ray = Ray(record.p, light - record.p);

            if (!record.material->scatter(cur_ray, record, attenuation, scattered_ray, rand)) return {0, 0, 0};
            sky_color = sky_color * attenuation;


            if (has_point_light && !(*world)->hit2(light_ray) && attenuation != Color{1, 1, 1} /* Not dielectric */) {
                if (dot(scattered_ray.direction(), record.normal) <= 0) break;

                auto diffuse = attenuation * light_color * dot(light_ray.direction().normalize(), record.normal);

                auto h = ((camera_center - record.p).normalize() + (light - record.p).normalize()).normalize();
                auto specular = light_color * pow(dot(h, record.normal), shinyness / 4);

                illumination_color += (diffuse * diffuse_intensity + specular * specular_intensity);

            }
            cur_ray = scattered_ray;
        }

        return (sky_color * sky_intensity + illumination_color).clamp(0, 1);
    }

    void set_focus_dist(double dist) {
        look_at = -dist * w + look_from;
        update();
    }
};