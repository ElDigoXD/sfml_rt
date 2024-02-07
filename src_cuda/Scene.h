#pragma once

#include <cuda.h>
#include "Sphere.h"
#include "utils.h"

#define create_scene(name, length)                                                                                                                               \
                                                                                                                                                                 \
__global__ void ___##name##_kernel(Hittable **d_list, HittableList **d_world, Camera &d_camera, curandState *d_global_state, int d_list_length);                \
__host__ void name(Hittable ***d_list, HittableList ***d_world, Camera &d_camera, curandState *d_global_state) {                                               \
    CU(cudaMalloc(&*d_list, sizeof(Hittable *) * length));                                                                                                       \
    CU(cudaMalloc(&*d_world, sizeof(HittableList *)));                                                                                                           \
                                                                                                                                                                 \
    ___##name##_kernel<<<1,1>>>(*d_list, *d_world, d_camera, d_global_state, length);                                                                           \
    CU(cudaGetLastError());                                                                                                                                      \
    CU(cudaDeviceSynchronize());                                                                                                                                 \
                                                                                                                                                                 \
}                                                                                                                                                                \
__global__ void ___##name##_kernel(Hittable **d_list, HittableList **d_world, Camera &d_camera,  curandState *d_global_state, int d_list_length) {              \
    if (threadIdx.x != 0 || blockIdx.x != 0) return;                                                                                                             \
    auto l_rand = d_global_state;

#define end_create_scene                                                                                                                                         \
    *d_world = new HittableList(d_list, d_list_length);                                                                                                          \
}

#define load_scene(scene) scene(&d_list, &d_world, *d_camera, d_global_state2)
#ifdef __CUDACC__
#include <device_launch_parameters.h>

namespace Scene {

    create_scene(ch10_metal, 4)
        d_list[0] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 1));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.8, 0.8), 0.3));

        d_camera.vfov = 20;
        d_camera.update();

    end_create_scene

    create_scene(ch11_dielectrics, 5)
        d_list[0] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[1] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));

    end_create_scene

    create_scene(ch12_camera_19, 2)
        auto R = cos(std::numbers::pi / 4);
        d_list[0] = new Sphere(Vec3(R, 0, -1), R, new Lambertian(Colors::blue()));
        d_list[1] = new Sphere(Vec3(-R, 0, -1), R, new Lambertian(Colors::red()));

        d_camera.vfov = 90;
        d_camera.update();
    end_create_scene

    create_scene(ch12_camera_20, 5)
        d_list[0] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[1] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[2] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));

        d_camera.vfov = 90;
        d_camera.look_from = Point3(-2, 2, 1);
        d_camera.look_at = Point3(0, 0, -1);
        d_camera.update();
    end_create_scene

    create_scene(ch12_camera_21, 5)
        d_list[0] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[1] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[2] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));

        d_camera.vfov = 20;
        d_camera.look_from = Point3(-2, 2, 1);
        d_camera.look_at = Point3(0, 0, -1);
        d_camera.update();
    end_create_scene


    create_scene(ch13_defocus_blur, 5)
        d_list[0] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[1] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[2] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));

        d_camera.vfov = 20;
        d_camera.look_from = Point3(-2, 2, 1);
        d_camera.look_at = Point3(0, 0, -1);
        d_camera.defocus_angle = 10;
        d_camera.update();
    end_create_scene

    // Depends on seed
    create_scene(ch14_what_next, (485))
        auto idx = 0;
        d_list[idx++] = new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double(l_rand);
                Point3 center(a + 0.9 * Random::_double(l_rand), 0.2, b + 0.9 * Random::_double(l_rand));

                if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                    Material *sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = Vec3::random(l_rand) * Vec3::random(l_rand);
                        sphere_material = new Lambertian(albedo);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = Color(0.5, 0.5, 0.5) + Vec3::random(l_rand) * 0.5;
                        auto fuzz = 0.5 * Random::_double(l_rand);
                        sphere_material = new Metal(albedo, fuzz);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    } else {
                        // glass
                        sphere_material = new Dielectric(1.5);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    }
                }

            }
        }
        auto material1 = new Dielectric(1.5);
        d_list[idx++] = new Sphere(Point3(0, 1, 0), 1.0, material1);

        auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        d_list[idx++] = new Sphere(Point3(-4, 1, 0), 1.0, material2);

        auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
        d_list[idx++] = new Sphere(Point3(4, 1, 0), 1.0, material3);

        assert(485 == idx);
        d_camera.vfov = 20;
        d_camera.look_from = {13, 2, 3};
        d_camera.look_at = d_camera.look_from - unit_vector(d_camera.look_from) * 10;
        d_camera.defocus_angle = 0.6;
        d_camera.update();
    end_create_scene
}
#endif

namespace CPUScene {
    HittableList *book_1_end(Camera &d_camera) {
        auto idx = 0;
        auto d_list = new Hittable *[485];

        d_list[idx++] = new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double();
                Point3 center(a + 0.9 * Random::_double(), 0.2, b + 0.9 * Random::_double());

                if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                    Material *sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = Vec3::random() * Vec3::random();
                        sphere_material = new Lambertian(albedo);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = Color(0.5, 0.5, 0.5) + Vec3::random() * 0.5;
                        auto fuzz = 0.5 * Random::_double();
                        sphere_material = new Metal(albedo, fuzz);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    } else {
                        // glass
                        sphere_material = new Dielectric(1.5);
                        d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                    }
                }

            }
        }
        auto material1 = new Dielectric(1.5);
        d_list[idx++] = new Sphere(Point3(0, 1, 0), 1.0, material1);

        auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        d_list[idx++] = new Sphere(Point3(-4, 1, 0), 1.0, material2);

        auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
        d_list[idx++] = new Sphere(Point3(4, 1, 0), 1.0, material3);

        assert(485 == idx);
        d_camera.vfov = 20;
        d_camera.look_from = {13, 2, 3};
        d_camera.look_at = d_camera.look_from - unit_vector(d_camera.look_from) * 10;
        d_camera.defocus_angle = 0.6;
        d_camera.update();

        return new HittableList(d_list, 485);
    }
}