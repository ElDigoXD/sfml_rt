#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include <iostream>
#include <random>
#include <cstring>
#include "SFML/Graphics/Image.hpp"

class Camera {
public:
    double aspect_ratio{};
    int image_width{};
    int image_height{};

    int samples_per_pixel = 10;
    int max_depth = 10;
    float reflectance = 0.5;
private:

    double focal_length{};
    double viewport_width{};
    double viewport_height{};
    Point3 camera_center;

    Vec3 viewport_x;
    Vec3 viewport_y;

    Vec3 pixel_delta_x;
    Vec3 pixel_delta_y;

    Vec3 viewport_upper_left;

    Point3 pixel_00_location;

public:
    Camera() {
        image_width = 600;
        image_height = 400;
        update(image_width, image_height);
    }

    void update(int width, int height) {
        image_width = width;
        image_height = height;

        aspect_ratio = width * 1.0 / height;

        focal_length = 1;
        viewport_height = 2;
        viewport_width = viewport_height * (width * 1.0 / height);
        camera_center = Point3(0, 0, 0);

        viewport_x = Vec3(viewport_width, 0, 0);
        viewport_y = Vec3(0, -viewport_height, 0);

        pixel_delta_x = viewport_x / width;
        pixel_delta_y = viewport_y / height;

        viewport_upper_left = camera_center - Vec3(0, 0, focal_length) - viewport_x / 2 - viewport_y / 2;
        pixel_00_location = viewport_upper_left + 0.5 * (pixel_delta_x + pixel_delta_y);
    }

    Ray get_random_ray_at(int i, int j) {
        auto pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
        auto pixel_sample = pixel_center + pixel_sample_square();

        return Ray(camera_center, pixel_sample - camera_center);

    }

    [[nodiscard]] Vec3 pixel_sample_square() const {
        return (-0.5 + Random::generate_canonical()) * pixel_delta_x +
               (-0.5 + Random::generate_canonical()) * pixel_delta_y;
    }


#include "SFML/Graphics/Image.hpp"

    void render(sf::Image &image, const HittableList &world) {
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                Color pixel_color = Colors::black;
                for (int sample = 0; sample < samples_per_pixel; ++sample) {
                    Ray ray = get_random_ray_at(i, j);
                    pixel_color += ray_color(ray, max_depth, world);
                }
                pixel_color /= samples_per_pixel;


                image.setPixel(i, j, to_sf_color(pixel_color));
            }
        }
    }

    void render2(unsigned int pixels[], const HittableList &world) {
        for (int j = 0; j < image_height; ++j) {
            render_pixel_line(&pixels[j * image_width], world, j);
        }
    }

    inline void render_pixel_line(unsigned int pixels[], const HittableList &world, int line) {
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color = Color(0,0,0);
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                Ray ray = get_random_ray_at(i, line);
                pixel_color += ray_color(ray, max_depth, world);
            }
            pixel_color /= samples_per_pixel;

            auto sf_color = to_sf_gamma_color(pixel_color);
            std::memcpy(&pixels[i], &sf_color, 4);
        }
    }


    static Color lerp(Color start, Color end, double a) {
        return (1 - a) * start + a * end;
    }

    constexpr static const double infinity = std::numeric_limits<double>::infinity();

    Color ray_color(const Ray &ray, int depth, const Hittable &world) {
        HitRecord record;

        if (depth <= 0)
            return (Colors::black);

        if (world.hit(ray, Interval(0.001, infinity), record)) {
            Vec3 diffusion_direction = random_on_hemisphere(record.normal);
            Vec3 lambertian_direction = record.normal + random_unit_vector();

            return reflectance * ray_color(Ray(record.p, lambertian_direction), depth - 1, world);

            // Show normal vectors, maybe a toggle
            // return 0.5 * (record.normal + Color(1, 1, 1));
        }

        Vec3 unit_direction = ray.direction().normalize();
        auto a = 0.5 * (unit_direction.y + 1.0);
        return lerp(Colors::white, Colors::blue_sky, a);
    }
};