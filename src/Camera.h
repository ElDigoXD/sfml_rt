#pragma once

#include "Vec3.h"
#include "Ray.h"
#include <iostream>

class Camera {
private:
    double aspect_ratio{};
    int image_width{};
    int image_height{};

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

    static bool hit_sphere(const Point3 &center, double radious, const Ray ray) {
        Vec3 oc = ray.origin() - center;
        auto a = dot(ray.direction(), ray.direction());
        auto b = 2.0 * dot(oc, ray.direction());
        auto c = dot(oc, oc) - radious * radious;
        return (b * b - 4 * a * c) >= 0;
    }

#ifdef IS_SFML

#include "SFML/Graphics/Image.hpp"

    void render(sf::Image &image) {
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                Point3 pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
                auto ray_direction = pixel_center - camera_center;
                Ray r = Ray(camera_center, ray_direction);

                Color pixel_color = ray_color(r);
                image.setPixel(i, j, to_sf_color(pixel_color));
            }
        }
    }

#endif
#ifdef PROFILE
    volatile int a = 0;

    void render() {
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                Point3 pixel_center = pixel_00_location + (i * pixel_delta_x) + (j * pixel_delta_y);
                auto ray_direction = pixel_center - camera_center;
                Ray r = Ray(camera_center, ray_direction);

                Color pixel_color = ray_color(r);
                a = pixel_color.r;
            }
        }
    }
#endif

    static Color lerp(Color start, Color end, double a) {
        return (1 - a) * start + a * end;
    }

    static Color ray_color(Ray &ray) {
        if (hit_sphere(Point3(0, 0, -1), 0.5, ray)) {
            return {1, 0, 0};
        }

        Vec3 unit_direction = ray.direction().unit();
        auto a = 0.5 * (unit_direction.y + 1.0);
        return lerp(Color(1, 1, 1), Color(0.5, 0.7, 1), a);
    }
};