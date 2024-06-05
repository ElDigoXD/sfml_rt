#pragma once

#include "Ray.h"
#include "hittable/HittableList.h"
#include "hittable/Hittable.h"
#include <complex>
#include <cmath>

#ifndef SCREEN_HEIGHT_IN_PX
#define SCREEN_HEIGHT_IN_PX 16
#endif
#ifndef THREADS
#define THREADS 1
#endif

#ifdef CUDA
#include <thrust/complex.h>
typedef thrust::complex<double> Complex;
#else
typedef std::complex<double> Complex;
#endif

class HoloCamera {

public:
    int samples_per_pixel;

    int max_depth;
    const double mm = 1e-3;

    Point3 look_from = Point3(0, 0, 200) * mm;
    Point3 look_at = Point3(0, 0, 0);

    Vec3 u, v, w;

    Point3 camera_center;


    // Holo stuff
    const double wavelength = 0.6328e-6;
    // Focal distance
    const double slm_z = 200 * mm;

    // Slm screen
    int slm_width_in_px;
    int slm_height_in_px;

    // Point cloud screen
    const int screen_height_in_px;
    const int screen_width_in_px = std::floor(screen_height_in_px * 1.77);

    //int slm_width_px = 256;
    //int slm_height_px = 256;

    const double slm_pixel_size = 8e-6;

    // Normal stuff

    Vec3 screen_pixel_delta_x;
    Vec3 screen_pixel_delta_y;

    Vec3 slm_pixel_delta_x;
    Vec3 slm_pixel_delta_y;

    Point3 screen_pixel_00_location;
    Point3 slm_pixel_00_location;

    void *operator new(size_t size) {
        return malloc(size);
    }

#ifdef CUDA
    __host__ void *operator new(size_t len, bool gpu) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }
#endif

    GPU HoloCamera(int _image_width, int _image_height, int _samples_per_pixel, int _max_depth, int screen_height)
            : slm_width_in_px(_image_width),
              slm_height_in_px(_image_height),
              samples_per_pixel(_samples_per_pixel),
              max_depth(_max_depth),
              screen_height_in_px(screen_height) {
        update();
    }

    GPU void update() { update(slm_width_in_px, slm_height_in_px); }

    GPU void update(int width, int height) {
        camera_center = Point3(0, 0, slm_z);

        // SLM screen
        slm_width_in_px = width;
        slm_height_in_px = height;

        // SLM "real" size: 8.64 x 15.36 mm
        double h_slm_size = slm_pixel_size * (slm_width_in_px);
        double v_slm_size = slm_pixel_size * (slm_height_in_px);

        // Point cloud screen
        double h_screen_pixel_size = slm_pixel_size * slm_width_in_px / screen_width_in_px;
        double v_screen_pixel_size = slm_pixel_size * slm_height_in_px / screen_height_in_px;
        double h_screen_size = h_screen_pixel_size * (screen_width_in_px);
        double v_screen_size = v_screen_pixel_size * (screen_height_in_px);

        printf("screen size: %f %f\n", h_screen_size, v_screen_size);
        printf("slm size: %f %f\n", h_slm_size, v_slm_size);

        w = (look_from - look_at).normalize();
        u = cross(Vec3(0, 1, 0), w).normalize();
        v = cross(w, u);

        Vec3 screen_x = h_screen_size * u;
        Vec3 screen_y = v_screen_size * -v;

        screen_pixel_delta_x = screen_x / screen_width_in_px;
        screen_pixel_delta_y = screen_y / screen_height_in_px;

        Vec3 screen_upper_left = camera_center - (slm_z * w) - screen_x / 2 - screen_y / 2;
        screen_pixel_00_location = screen_upper_left + 0.5 * (screen_pixel_delta_x + screen_pixel_delta_y);

        Vec3 slm_x = h_slm_size * u;
        Vec3 slm_y = v_slm_size * -v;

        slm_pixel_delta_x = slm_x / slm_width_in_px;
        slm_pixel_delta_y = slm_y / slm_height_in_px;

        Vec3 slm_upper_left = camera_center - slm_x / 2 - slm_y / 2;
        slm_pixel_00_location = slm_upper_left + 0.5 * (slm_pixel_delta_x + slm_pixel_delta_y);
    }

    [[nodiscard]] GPU Ray get_ray_at_screen(int i, int j) const {
        auto pixel_center = screen_pixel_00_location + (i * screen_pixel_delta_x) + (j * screen_pixel_delta_y);
        auto ray_origin = camera_center;

        return Ray(ray_origin, pixel_center - ray_origin);
    }

    [[nodiscard]] GPU Vec3 screen_pixel_sample_square(curandState *rand) const {
        return (-0.5 + Random::_double(rand)) * screen_pixel_delta_x +
               (-0.5 + Random::_double(rand)) * screen_pixel_delta_y;
    }

    [[nodiscard]] GPU Vec3 slm_pixel_sample_square(curandState *rand) const {
        return (-0.5 + Random::_double(rand)) * slm_pixel_delta_x +
               (-0.5 + Random::_double(rand)) * slm_pixel_delta_y;
    }

    GPU Ray get_random_ray_at_screen(int i, int j, curandState *rand) const {
        auto pixel_center = screen_pixel_00_location + (i * screen_pixel_delta_x) + (j * screen_pixel_delta_y);
        auto ray_origin = camera_center;

        auto pixel_sample = pixel_center + screen_pixel_sample_square(rand);
        return Ray(ray_origin, pixel_sample - ray_origin);
    }

    [[nodiscard]] std::vector<Point3> generate_point_cloud(const HittableList &world) const {
        auto point_cloud = std::vector<Point3>();
        HitRecord record;
        Ray ray;
        for (int j = 0; j < screen_height_in_px; ++j) {
            for (int i = 0; i < screen_width_in_px; ++i) {
#ifdef ENABLE_RANDOM_SCREEN_RAYS
                ray = get_random_ray_at_screen(i, j, nullptr);
#else
                ray = get_ray_at_screen(i, j);
#endif
                if (!(world).hit(ray, Interval(0.0000001, infinity), record)) {
                    continue;
                }
                point_cloud.push_back(record.p);
            }
        }

        std::printf("point cloud size: %zu\n", point_cloud.size());

        return point_cloud;
    }


    GPU static Color lerp(Color start, Color end, double a) {
        return (1 - a) * start + a * end;
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

    void render_CGH_line(
            Complex pixels[], const HittableList &world, const std::vector<Point3> &point_cloud,
            int j) const {

        for (int i = 0; i < slm_width_in_px; ++i) {
            auto slm_pixel_center = slm_pixel_00_location + (i * slm_pixel_delta_x) + (j * slm_pixel_delta_y);

            for (const auto &point: point_cloud) {
                auto ray = Ray(slm_pixel_center, point - slm_pixel_center);
                const Complex cgh = ray_wave_cgh(ray, point, max_depth, world, nullptr);
                pixels[i] += cgh;
            }
            pixels[i] /= (slm_width_in_px * slm_height_in_px * 1.0);

        }
        if (j % 100 == 0) {
            std::printf("line %d\n", j);
        }
    }

    //https://www.alcf.anl.gov/sites/default/files/2020-01/OpenMP_Jose.pdf
    void render_CGH(Complex pixels[], const HittableList &world,
                    const std::vector<Point3> &point_cloud) const {

        //auto point_cloud = generate_point_cloud(world);

#pragma omp parallel for default(none) firstprivate(point_cloud, world) shared(pixels) num_threads(THREADS)

        for (int j = 0; j < slm_height_in_px; ++j) {
            for (int i = 0; i < slm_width_in_px; ++i) {
                auto slm_pixel_center = slm_pixel_00_location + (i * slm_pixel_delta_x) + (j * slm_pixel_delta_y);

                for (const auto &point: point_cloud) {
                    auto ray = Ray(slm_pixel_center, point - slm_pixel_center);
                    const Complex cgh = ray_wave_cgh(ray, point, max_depth, world, nullptr);
                    pixels[i + j * slm_width_in_px] += cgh;
                }
                pixels[i + j * slm_width_in_px] /= (slm_width_in_px * slm_height_in_px * 1.0);
            }
            if (j % 100 == 0) {
                std::printf("line %d\n", j);
            }
        }
    }

    GPU Complex ray_wave_cgh(const Ray &ray, const Point3 expected_point, int depth, const HittableList &world,
                             curandState *rand) const {
        HitRecord record;
        Color attenuation;
        Color total_point_light_attenuation = Colors::white();
        //Color sky_color = Colors::blue_sky();
        Color diffuse;
        Color specular;
        Color illumination_color{0, 0, 0};
        Ray cur_ray = ray;
        int cur_depth = depth;
        Ray scattered_ray;
        bool has_point_light = light != Color{0, 0, 0};
        auto po_distance = 0.0;


        while (world.hit(cur_ray, Interval(0.0000001, infinity), record)) {
            // Ray does not escape, so it's represented as black
            if (cur_depth == max_depth && !(record.p - expected_point).is_near_zero()) {
                return 0;
            }
            if (cur_depth-- <= 0) break;

            auto light_ray = Ray(record.p, light - record.p);

            if (!record.material->scatter(cur_ray, record, attenuation, scattered_ray, rand)) break;
            //sky_color *= attenuation;
            total_point_light_attenuation *= attenuation;

            if (has_point_light
                && !(world).hit(light_ray)
                && attenuation != Color{1, 1, 1} /* Not dielectric */) {

                if (dot(scattered_ray.direction(), record.normal) <= 0) break;

                auto intersection_to_light_distance = (light - record.p).length();
                po_distance = (ray.origin() - record.p).length() + intersection_to_light_distance;

                if (record.material->is_diffuse()) {
                    diffuse += total_point_light_attenuation * light_color *
                               dot(light_ray.direction().normalize(), record.normal);
                }

                auto h = ((camera_center - record.p).normalize() + (light - record.p).normalize()).normalize();
                specular += light_color * pow(dot(h, record.normal), shinyness / 4);
            }
            cur_ray = scattered_ray;
        }
        auto sub_image = (diffuse * diffuse_intensity + specular * specular_intensity).clamp(0, 1);
        auto sub_phase = ((2 * M_PI / wavelength) * po_distance);
#ifdef CUDA
        Complex sub_phase_c = thrust::exp(thrust::complex(0.0, 1.0 * sub_phase));
#else
        Complex sub_phase_c = std::exp(Complex(0, sub_phase));
#endif
        auto sub_cgh_ray = (sub_image.r * sub_phase_c);
        return sub_cgh_ray;
    }

    void print_properties() const {
        std::printf("wavelength: %f\n", wavelength);
        std::printf("slm_pixel_size: %f\n", slm_pixel_size);
        std::printf("slm_width_in_px: %d\n", slm_width_in_px);
        std::printf("slm_height_in_px: %d\n", slm_height_in_px);
        std::printf("slm_z: %.3f\n", slm_z);
        std::printf("screen_width_in_px: %d\n", screen_width_in_px);
        std::printf("screen_height_in_px: %d\n", screen_height_in_px);
        std::printf("camera_center: %.3f %.3f %.3f\n", camera_center.x, camera_center.y, camera_center.z);
        std::printf("look_from: %.3f %.3f %.3f\n", look_from.x, look_from.y, look_from.z);
        std::printf("look_at: %.3f %.3f %.3f\n", look_at.x, look_at.y, look_at.z);
        std::printf("light_color: %.3f %.3f %.3f\n", light_color.r, light_color.g, light_color.b);
        std::printf("diffuse_intensity: %.3f\n", diffuse_intensity);
        std::printf("specular_intensity: %.3f\n", specular_intensity);
        std::printf("sky_intensity: %.3f\n", sky_intensity);
        std::printf("samples_per_pixel: %d\n", samples_per_pixel);
        std::printf("max_depth: %d\n", max_depth);
    }
};