#pragma once

#include "Hittable.h"
#include "Camera.h"

namespace Scene {
    HittableList book_1_end(Camera &camera) {
        HittableList world{};
        camera.vfov = 20;
        camera.look_from = {13, 2, 3};
        camera.look_at = camera.look_from - unit_vector(camera.look_from) * 10;
        camera.defocus_angle = 0.6;
        camera.update();

        auto ground_material = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
        world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double();
                Point3 center(a + 0.9 * Random::_double(), 0.2, b + 0.9 * Random::_double());

                if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                    std::shared_ptr<Material> sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = Color::random() * Color::random();
                        sphere_material = std::make_shared<Lambertian>(albedo);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = Color::random(0.5, 1);
                        auto fuzz = Random::_double(0, 0.5);
                        sphere_material = std::make_shared<Metal>(albedo, fuzz);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    } else {
                        // glass
                        sphere_material = std::make_shared<Dielectric>(1.5);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    }
                }
            }
        }

        auto material1 = std::make_shared<Dielectric>(1.5);
        world.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

        auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
        world.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

        auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
        world.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

        return world;
    }
}