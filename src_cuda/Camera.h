#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include "Material.h"
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

    Point3 look_from = Point3(0, 0, -1);
    Point3 look_at = Point3(0, 0, 0);

    Vec3 u, v, w;

    Point3 camera_center;

    double defocus_angle = 10;
    double focus_dist = 3.4;

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
    Camera() : Camera(600, 400) {}

    Camera(int _image_width, int _image_height) : Camera(_image_width, _image_height, 10, 100) {}

    Camera(int _image_width, int _image_height, int _samples_per_pixel, int _max_depth)
            : image_width(_image_width),
              image_height(_image_height),
              samples_per_pixel(_samples_per_pixel),
              max_depth(_max_depth) {
        update(image_width, image_height);
    }

    void update() { update(image_width, image_height); }

    void update(int width, int height) {
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

    Ray get_random_ray_at(int i, int j) {
        auto pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
        auto pixel_sample = pixel_center + pixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample();

        return Ray(ray_origin, pixel_sample - ray_origin);
    }

    Ray get_random_ray_at_subpixel(int i, int j, int sub_i, int sub_j) {
        auto subpixel_center =
                viewport_upper_left + i * pixel_delta_x + j * pixel_delta_y + sub_i * subpixel_delta_x +
                sub_j * subpixel_delta_y;
        auto pixel_sample = subpixel_center + subpixel_sample_square();

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample();

        return Ray(ray_origin, pixel_sample - ray_origin);
    }


    Ray get_ray_at(int i, int j) {
        auto pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
        auto ray_origin = camera_center;

        return Ray(ray_origin, pixel_center - ray_origin);
    }

    [[nodiscard]] Vec3 pixel_sample_square() const {
        return (-0.5 + Random::generate_canonical()) * pixel_delta_x +
               (-0.5 + Random::generate_canonical()) * pixel_delta_y;
    }

    [[nodiscard]] Vec3 subpixel_sample_square() const {
        return (-0.5 + Random::generate_canonical()) * subpixel_delta_x +
               (-0.5 + Random::generate_canonical()) * subpixel_delta_y;
    }

    [[nodiscard]] Point3 defocus_disk_sample() const {
        auto p = random_in_unit_disk();
        return camera_center + p.x * defocus_disk_x + p.y * defocus_disk_y;
    }

    void render(unsigned char pixels[], const HittableList &world) {
        for (int j = 0; j < image_height; ++j) {
            render_pixel_line(&pixels[j * image_width * 4], world, j);
        }
    }

    void render_pixel_line(unsigned char pixels[], const HittableList &world, int line) {
        int spp_sq = std::floor(std::sqrt(samples_per_pixel));
        subpixel_delta_x = pixel_delta_x / spp_sq;
        subpixel_delta_y = pixel_delta_y / spp_sq;

        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = Color(0, 0, 0);

            for (int sample_i = 0; sample_i < spp_sq; ++sample_i) {
                for (int sample_j = 0; sample_j < spp_sq; ++sample_j) {
                    Ray ray = get_random_ray_at_subpixel(i, line, sample_i, sample_j);
                    pixel_color += ray_color(ray, max_depth, world);
                }
            }
            pixel_color /= std::pow(spp_sq, 2);

            auto rgba_color = to_gamma_color(pixel_color);
            pixels[i * 4 + 0] = static_cast<unsigned char>(rgba_color.r * 255);
            pixels[i * 4 + 1] = static_cast<unsigned char>(rgba_color.g * 255);
            pixels[i * 4 + 2] = static_cast<unsigned char>(rgba_color.b * 255);
            pixels[i * 4 + 3] = 255;
        }
    }

    void render_color_line(Color pixels[], const HittableList &world, int line) {
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = Color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                Ray ray = get_random_ray_at(i, line);
                pixel_color += ray_color(ray, max_depth, world);
            }
            pixel_color /= samples_per_pixel;
            pixels[i] = pixel_color;
        }
    }

    static Color lerp(Color start, Color end, double a) {
        return (1 - a) * start + a * end;
    }

    constexpr static const double infinity = std::numeric_limits<double>::infinity();

    Color ray_color_recursive(Ray &ray, int depth, const Hittable &world) {
        HitRecord record;

        if (depth <= 0)
            return (Colors::black);

        if (world.hit(ray, Interval(0.001, infinity), record)) {
            Ray scattered_ray;
            Color attenuation;
            if (record.material->scatter(ray, record, attenuation, scattered_ray)) {
                return attenuation * ray_color_recursive(scattered_ray, depth - 1, world);
            };
            return Colors::black;
        }

        Vec3 unit_direction = ray.direction().normalize();
        auto a = 0.5 * (unit_direction.y + 1.0);
        return lerp(Colors::white, Colors::blue_sky, a);
    }

    static Color ray_color(Ray &ray, int depth, const Hittable &world) {
        HitRecord record;
        Color attenuation;
        Color ray_color{1, 1, 1};

        if (depth <= 0)
            return (Colors::black);

        while (world.hit(ray, Interval(0.001, infinity), record)) {
            if (depth-- <= 0) {
                return (Colors::black);
            } else if (record.material->scatter(ray, record, attenuation, ray)) {
                ray_color = ray_color * attenuation;
            } else {
                return Colors::black;
            }
        }

        Vec3 unit_direction = ray.direction().normalize();
        auto a = 0.5 * (unit_direction.y + 1.0);
        return ray_color * lerp(Colors::white, Colors::blue_sky, a);
    }

    void set_focus_dist(double dist) {
        look_at = -dist * w + look_from;
        update();
    }
};