#pragma once

#include <cuda.h>
#include "hittable/Sphere.h"
#include "utils.h"
#include "hittable/Triangle.h"
#include "obj.h"
#include "hittable/BVHNode.h"
#include "HoloCamera.h"
#include "Camera.h"

#ifdef CUDA

#define create_scene(name, length)                                                                                                                               \
                                                                                                                                                                 \
__global__ void ___##name##_kernel(Hittable **d_list, HittableList **d_world, Camera &d_camera, curandState *d_global_state, int d_list_length);                \
__host__ void name(Hittable ***d_list, HittableList ***d_world, Camera &d_camera, curandState *d_global_state) {                                               \
    CU(cudaMalloc(&*d_list, sizeof(Hittable *) * (length)));                                                                                                       \
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

#define create_holo_scene(name, length)                                                                                                                               \
                                                                                                                                                                 \
__global__ void ___##name##_kernel(Hittable **d_list, HittableList **d_world, HoloCamera &d_camera, curandState *d_global_state, int d_list_length);                \
__host__ void name(Hittable ***d_list, HittableList ***d_world, HoloCamera &d_camera, curandState *d_global_state) {                                               \
    CU(cudaMalloc(&*d_list, sizeof(Hittable *) * (length)));                                                                                                       \
    CU(cudaMalloc(&*d_world, sizeof(HittableList *)));                                                                                                           \
                                                                                                                                                                 \
    ___##name##_kernel<<<1,1>>>(*d_list, *d_world, d_camera, d_global_state, length);                                                                           \
    CU(cudaGetLastError());                                                                                                                                      \
    CU(cudaDeviceSynchronize());                                                                                                                                 \
                                                                                                                                                                 \
}                                                                                                                                                                \
__global__ void ___##name##_kernel(Hittable **d_list, HittableList **d_world, HoloCamera &d_camera,  curandState *d_global_state, int d_list_length) {              \
    if (threadIdx.x != 0 || blockIdx.x != 0) return;                                                                                                             \
    auto l_rand = d_global_state;

#define end_create_scene                                                                                                                                         \
    *d_world = new HittableList(d_list, d_list_length);                                                                                                          \
}

#define load_scene(scene) scene(&d_list, &d_world, *d_camera, d_global_state2)



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
    create_scene(ch14_what_next, (22 * 22 + 1 + 3))
        auto idx = 0;
        d_list[idx++] = new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double(l_rand);
                Point3 center(a + 0.9 * Random::_double(l_rand), 0.2, b + 0.9 * Random::_double(l_rand));

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
        auto material1 = new Dielectric(1.5);
        d_list[idx++] = new Sphere(Point3(0, 1, 0), 1.0, material1);

        auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        d_list[idx++] = new Sphere(Point3(-4, 1, 0), 1.0, material2);

        auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
        d_list[idx++] = new Sphere(Point3(4, 1, 0), 1.0, material3);

        assert(22 * 22 + 1 + 3 == idx);
        d_camera.vfov = 20;
        d_camera.look_from = {13, 2, 3};
        d_camera.look_at = d_camera.look_from - unit_vector(d_camera.look_from) * 10;
        d_camera.defocus_angle = 0.6;
        d_camera.update();
    end_create_scene

    create_scene(point_light, (4))
        d_list[0] = new Sphere(Vec3(0, 1, -1), 0.5, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(0, -99.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vec3(1, 1, -1), 0.5, new Dielectric(1.5));
        d_list[3] = new Sphere(Vec3(-1, 1, -1), 0.5, new Metal(Colors::red(), 0));

        d_camera.vfov = 10;
        d_camera.look_from = {0, 5, 10};
        d_camera.look_at = {0, 1, 0};
        d_camera.update();

    end_create_scene

    create_holo_scene(hologram, (3))
        auto mm = 1e-3;

        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 0) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        d_camera.light = Point3(5, 5, 10);
        d_camera.light_color = {1, 0, 0};
        d_camera.diffuse_intensity = 1;
        d_camera.specular_intensity = 0;

    end_create_scene

    HittableList *hologram_cpu(HoloCamera &d_camera) {
        auto mm = 1e-3;
        auto d_list = new Hittable *[3];
        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 0) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        d_camera.light = Point3(5, 5, 10);
        d_camera.light_color = {1, 0, 0};
        d_camera.diffuse_intensity = 1;
        d_camera.specular_intensity = 0;

        // d_camera.update();

        return new HittableList(d_list, 3);
    }
}

namespace ObjScene {
    __global__ void
    ___shuttle_kernel(Hittable **d_list, HittableList **d_world, Camera &d_camera, const Vec3 *vertices,
                      curandState *d_global_state, int d_list_length);

    __host__ void shuttle(Hittable ***d_list, HittableList ***d_world, Camera &d_camera, curandState *d_global_state) {
        Vec3 *out_vertices;
        auto len = Obj::get_vertices(&out_vertices, true);

        CU(cudaMalloc(&*d_list, sizeof(Hittable *) * len));
        CU(cudaMalloc(&*d_world, sizeof(HittableList *)));
        CU(cudaDeviceSynchronize());

        ___shuttle_kernel<<<1, 1>>>(*d_list, *d_world, d_camera, out_vertices, d_global_state, len);
        CU(cudaGetLastError());
        CU(cudaDeviceSynchronize());
    }

    __global__ void
    ___shuttle_kernel(Hittable **d_list, HittableList **d_world, Camera &d_camera, const Vec3 *vertices,
                      curandState *d_global_state, int d_list_length) {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;
        auto l_rand = d_global_state;

        auto *mat = new Lambertian({0.2, 0.5, 0.8});
        for (int i = 0; i < d_list_length / 3; i++) {
            d_list[i] = new Triangle(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2], mat);
        }

        d_camera.look_from = {-6.31, 4.55, 3.32};
        d_camera.look_at = {-1.5, -1.1, -0.8};
        d_camera.light = {0, 10, 10};
        d_camera.light_color = {1, 1, 1};
        d_camera.diffuse_intensity = 1;
        d_camera.sky_intensity = 0;
        d_camera.update();

        *d_world = new HittableList(d_list, d_list_length / 3);
    }
}
#endif

namespace CPUScene {
    HittableList *book_1_end(Camera &d_camera) {
        auto idx = 0;
        auto d_list = new Hittable *[22 * 22 + 1 + 3];

        d_list[idx++] = new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double();
                Point3 center(a + 0.9 * Random::_double(), 0.2, b + 0.9 * Random::_double());

                Material *sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = Vec3::random(nullptr) * Vec3::random(nullptr);
                    sphere_material = new Lambertian(albedo);
                    d_list[idx++] = new Sphere(center, 0.2, sphere_material);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = Color(0.5, 0.5, 0.5) + Vec3::random(nullptr) * 0.5;
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
        auto material1 = new Dielectric(1.5);
        d_list[idx++] = new Sphere(Point3(0, 1, 0), 1.0, material1);

        auto material2 = new Lambertian(Color(0.4, 0.2, 0.1));
        d_list[idx++] = new Sphere(Point3(-4, 1, 0), 1.0, material2);

        auto material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0);
        d_list[idx++] = new Sphere(Point3(4, 1, 0), 1.0, material3);

        assert(22 * 22 + 1 + 3 == idx);
        d_camera.vfov = 20;
        d_camera.look_from = {13, 2, 3};
        d_camera.look_at = d_camera.look_from - unit_vector(d_camera.look_from) * 10;
        d_camera.defocus_angle = 0.6;
        d_camera.update();
        d_camera.light = {0, 3, 0};
        d_camera.light_color = {1, 1, 1};
        d_camera.diffuse_intensity = 1;
        d_camera.sky_intensity = 1;
        return new HittableList(d_list, 22 * 22 + 1 + 3);
    }

    HittableList *point_light(Camera &d_camera) {
        auto d_list = new Hittable *[5];

        d_list[0] = new Sphere(Vec3(0, 1, -1), 0.5, new Lambertian(Colors::red()));
        d_list[1] = new Sphere(Vec3(0, -99.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vec3(1, 1, -1), 0.5, new Dielectric(1.5));
        d_list[3] = new Sphere(Vec3(1, 1, -3), 0.5, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[4] = new Sphere(Vec3(-1, 1, -1), 0.5, new Metal(Colors::red(), 0));

        d_camera.vfov = 10;
        d_camera.look_from = {0, 5, 10};
        d_camera.look_at = {0, 1, 0};
        d_camera.update();

        d_camera.light = {0, 3, 0};
        d_camera.light_color = {1, 1, 1};

        d_camera.diffuse_intensity = 1;
        d_camera.sky_intensity = 0;


        return new HittableList(d_list, 5);
    }

    HittableList *triangle(Camera &d_camera) {
        auto d_list = new Hittable *[6];


        d_list[0] = new Triangle({-1, 0, -1}, {1, 0, -1}, {0, 1, 0}, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Triangle({1, 0, -1}, {1, 0, 1}, {0, 1, 0}, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Triangle({1, 0, 1}, {-1, 0, 1}, {0, 1, 0}, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[3] = new Triangle({-1, 0, 1}, {-1, 0, -1}, {0, 1, 0}, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[4] = new Triangle({-1, 0, -1}, {1, 0, 1}, {1, 0, -1}, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[5] = new Triangle({1, 0, 1}, {-1, 0, -1}, {-1, 0, 1}, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        d_camera.vfov = 90;
        d_camera.look_from = {1.130, 3, 1.5};
        d_camera.look_at = {0, 0, 0};
        d_camera.update();

        d_camera.light = {0, 3, 3};
        d_camera.light_color = {1, 1, 1};

        return new HittableList(d_list, 6);
    }

    HittableList *shuttle(Camera &d_camera) {
        Vec3 *vertices;
        auto d_list_length = Obj::get_vertices(&vertices, false);
        auto d_list = new Hittable *[d_list_length];

        // auto *mat = new Lambertian({0.2, 0.5, 0.8});
        auto *mat = new Metal({0.2, 0.5, 0.8}, 1);
        for (int i = 0; i < d_list_length / 3; i++) {
            d_list[i] = new Triangle(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2], mat);
        }

        d_camera.look_from = {-6.31, 4.55, 3.32};
        d_camera.look_at = {-1.5, -1.1, -0.8};
        d_camera.light = {0, 10, 10};
        d_camera.light_color = {1, 1, 1};
        d_camera.diffuse_intensity = 1;
        d_camera.sky_intensity = 0;
        d_camera.update();

        auto root = new BVHNode(d_list, d_list_length / 3, nullptr);

        return new HittableList(new Hittable *[]{(Hittable *) root}, 1);
    }


    HittableList *hologram(HoloCamera &d_camera) {
        auto mm = 1e-3;
        auto d_list = new Hittable *[3];
        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 4) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        d_camera.light = Point3(5, 5, 10);
        d_camera.light_color = {1, 0, 0};
        d_camera.diffuse_intensity = 1;
        d_camera.specular_intensity = 0;

        // d_camera.update();

        return new HittableList(d_list, 3);
    }

    HittableList *hologram_cpu(HoloCamera &d_camera) {
        auto mm = 1e-3;
        auto d_list = new Hittable *[3];
        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 4) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        d_camera.light = Point3(5, 5, 10);
        d_camera.light_color = {1, 0, 0};
        d_camera.diffuse_intensity = 1;
        d_camera.specular_intensity = 0;

        // d_camera.update();

        return new HittableList(d_list, 1);
    }
}

namespace TFGScene {
    auto blue = from_hex_code(0x26547c);
    auto magenta = from_hex_code(0xef476f);
    auto salmon = from_hex_code(0xf78c6b);
    auto yellow = from_hex_code(0xffd166);
    auto gold = from_hex_code(0xFFB916);
    auto teal = from_hex_code(0x06d6a0);
    auto white = from_hex_code(0xfffcf9);

    HittableList *example(Camera &camera) {
        auto list = new Hittable *[3];


        list[0] = new Sphere(Vec3(0, 1.4, -1), 0.2, new Lambertian(from_hex_code(0x89EE95))); // l_green
        list[1] = new Sphere(Vec3(0, 1, -1), 0.3, new Lambertian({from_hex_code(0x257CFF)})); // l_blue
        list[2] = new Sphere(Vec3(0, 0.5, -1), 0.4, new Lambertian(from_hex_code(0xFFBD1E))); // yellow

        camera.vfov = 10;
        camera.look_from = {0, 5, 10};
        camera.look_at = {0, 1.3, 0};
        camera.update();
        camera.sky_color = Colors::blue_sky();
        camera.light = Point3(3, 11, 3);
        camera.light_color = {1, 1, 1};
        camera.diffuse_intensity = 1;
        camera.specular_intensity = 1;
        return new HittableList(list, 3);
    }

    HittableList *lambertian(Camera &camera) {
        auto list = new Hittable *[4];


        list[0] = new Sphere(Vec3(0, -99.5, -1), 100, new Lambertian({.99, .99, .99})); // white
        list[1] = new Sphere(Vec3(-1, 1, -1), 0.5, new Lambertian(from_hex_code(0x89EE95))); // l_green
        list[2] = new Sphere(Vec3(0, 1, -1), 0.5, new Lambertian({from_hex_code(0x257CFF)})); // l_blue
        list[3] = new Sphere(Vec3(1, 1, -1), 0.5, new Lambertian(from_hex_code(0xFFBD1E))); // yellow

        camera.vfov = 10;
        camera.look_from = {0, 5, 10};
        camera.look_at = {0, 1.3, 0};
        camera.update();
        camera.sky_color = Colors::white();
        return new HittableList(list, 4);
    }

    HittableList *metal(Camera &camera) {
        auto list = new Hittable *[4];


        list[0] = new Sphere(Vec3(0, -99.5, -1), 100, new Metal(from_hex_code(0x3f3f3f), 0)); // gray
        list[1] = new Sphere(Vec3(-1, 1, -1), 0.5, new Metal(from_hex_code(0x89EE95), 0.1)); // l_green
        list[2] = new Sphere(Vec3(0, 1, -1), 0.5, new Metal({from_hex_code(0x257CFF)}, 0)); // l_blue
        list[3] = new Sphere(Vec3(1, 1, -1), 0.5, new Metal(from_hex_code(0xFFBD1E), 0.5)); // yellow

        camera.vfov = 10;
        camera.look_from = {0, 5, 10};
        camera.look_at = {0, 1.3, 0};
        camera.update();
        camera.sky_color = Colors::white();
        return new HittableList(list, 4);
    }

    HittableList *dielectric(Camera &camera) {
        auto list = new Hittable *[7];

        list[0] = new Sphere(Vec3(0, -99.5, -1), 100, new Lambertian(from_hex_code(0x9D9D9D))); // l_gray
        list[1] = new Sphere(Vec3(-1, 1, -1), 0.5, new Dielectric(1.3));
        list[2] = new Sphere(Vec3(0, 1, -1), 0.5, new Dielectric(1.5));
        list[3] = new Sphere(Vec3(1, 1, -1), 0.5, new Dielectric(2.4));
        list[4] = new Sphere(Vec3(-1, 1, -3), 0.5, new Lambertian(from_hex_code(0x89EE95))); // l_green
        list[5] = new Sphere(Vec3(0, 1, -3), 0.5, new Lambertian(from_hex_code(0x257CFF))); // l_blue
        list[6] = new Sphere(Vec3(1, 1, -3), 0.5, new Lambertian(from_hex_code(0xFFBD1E))); // yellow
        camera.vfov = 1.2;
        camera.look_from = {0, 16.7, 100};
        camera.look_at = {0, 1.3, 0};
        camera.update();
        camera.sky_color = from_hex_code(0xC2DAFF);
        return new HittableList(list, 7);
    }

    HittableList *materials(Camera &camera) {
        auto list = new Hittable *[7];

        list[0] = new Sphere(Vec3(0, -99.5, -1), 100, new Metal(from_hex_code(0x3f3f3f), 0)); // gray
        list[1] = new Sphere(Vec3(-1, 1, -1), 0.5, new Dielectric(1.3));
        list[2] = new Sphere(Vec3(0, 1, -1), 0.5, new Dielectric(1.5));
        list[3] = new Sphere(Vec3(1, 1, -1), 0.5, new Dielectric(2.4));
        list[4] = new Sphere(Vec3(-1, 1, -3), 0.5, new Lambertian(from_hex_code(0x89EE95))); // l_green
        list[5] = new Sphere(Vec3(0, 1, -3), 0.5, new Metal(from_hex_code(0x257CFF), 0.3)); // l_blue
        list[6] = new Sphere(Vec3(1, 1, -3), 0.5, new Lambertian(from_hex_code(0xFFBD1E))); // yellow
        camera.vfov = 1.2;
        camera.look_from = {0, 16.7, 100};
        camera.look_at = {0, 1, 0};
        camera.update();
        camera.sky_color = from_hex_code(0xC2DAFF);
        return new HittableList(list, 7);
    }

    HittableList *light(Camera &camera) {
        auto list = new Hittable *[4];


        list[0] = new Sphere(Vec3(0, -99.5, -1), 100, new Lambertian({.99, .99, .99})); // white
        list[1] = new Sphere(Vec3(-1, 1, -1), 0.5, new Lambertian(from_hex_code(0x89EE95))); // l_green
        list[2] = new Sphere(Vec3(0, 1, -1), 0.5, new Lambertian({from_hex_code(0x257CFF)})); // l_blue
        list[3] = new Sphere(Vec3(1, 1, -1), 0.5, new Lambertian(from_hex_code(0xFFBD1E))); // yellow

        camera.vfov = 10;
        camera.look_from = {0, 5, 10};
        camera.look_at = {0, 1.3, 0};
        camera.update();

        camera.sky_color = Colors::blue_sky();
        camera.light = Point3(0, 3, -1);
        camera.light_color = {1, 1, 1};
        camera.diffuse_intensity = 1;
        camera.specular_intensity = 1;


        return new HittableList(list, 4);
    }

    HittableList *hologram_cgi_scene(Camera &camera) {
        auto mm = 1e-3;
        auto d_list = new Hittable *[3];
        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 0) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        camera.light = Point3(5, 5, 10);
        camera.light_color = {1, 0, 0};
        camera.sky_intensity = 0;
        camera.diffuse_intensity = 1;
        camera.specular_intensity = 0;
        camera.look_from = Point3(0, 0, 0.200);
        camera.look_at = Point3(0, 0, 0);
        camera.vfov = 2;
        camera.update();

        return new HittableList(d_list, 3);


    }

    HittableList *hologram_cgh_scene(HoloCamera &camera) {
        auto mm = 1e-3;
        auto d_list = new Hittable *[3];
        d_list[0] = new Sphere(Vec3(-0.8, 0.25, -12) * mm, 3 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[1] = new Sphere(Vec3(1, -0.25, 0) * mm, 2.0 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));
        d_list[2] = new Sphere(Vec3(2, 2, 5) * mm, .5 * mm, new Lambertian(Vec3(0.7, 0.3, 0.3)));

        camera.light = Point3(50, 40, 150)*mm;
        camera.light_color = {1, 0, 0};
        camera.diffuse_intensity = 1;
        camera.specular_intensity = 1;

        return new HittableList(d_list, 3);
    }


}
