#include "Vec3.h"
#include "Camera.h"
#include "chrono"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"
#include "third-party/BS_thread_pool.h"

#include "third-party/BS_thread_pool.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include "third-party/tiny_obj_loader.h"

#include "Scene.h"

#include "cmath"

bool ENABLE_AMPLITUDE = true;

int main(int argc, char *argv[]) {
    int image_width = 1920;
    int image_height = 1080;
    int samples_per_pixel = 1;
    int num_threads = 1;
    if (argc != 1) {
        if (argc > 1) {
            samples_per_pixel = std::atoi(argv[1]);
        }
        if (argc > 2) {
            num_threads = std::atoi(argv[2]);
        }
        if (argc > 3) {
            image_width = std::atoi(argv[3]) * 16 / 9;
            image_height = std::atoi(argv[3]);
        }
    }


    //image_width = 256;
    //image_height = 256;


    unsigned char pixels[image_width * image_height];
    auto camera = HoloCamera(image_width, image_height, samples_per_pixel, 10);

    HittableList *world;
    world = CPUScene::hologram(camera);


    BS::thread_pool pool{static_cast<unsigned int>(num_threads)};

    std::cout << "Rendering a " << image_width << "x" << image_height << " hologram with " << camera.samples_per_pixel
              << " samples per pixel with " << pool.get_thread_count() << " threads.\n";

    camera.print_properties();

    auto start = time(nullptr);

    auto *pixels_complex = new std::complex<double>[image_width * image_height];
    auto point_cloud = camera.generate_point_cloud(&world);
    camera.render_CGH(pixels_complex, &world, point_cloud);

    // pool.detach_loop(0, image_height, [&](int j) {
    //     camera.render_CGH_line(&pixels_complex[j * image_width], &world, point_cloud, j);
    // }, 50);
    // pool.wait();

    // std::arg -> If no errors occur, this is the phase angle of z in the interval [−π; π].
    // [-pi, pi] -> [0, 2pi] -> [0, 1] -> [0, 255]
    for (int i = 0; i < image_width * image_height; i++) {
        pixels[i] = static_cast<unsigned char>((std::arg(pixels_complex[i]) + M_PI) / (2 * M_PI) * 255);
    }

    auto end = time(nullptr);
    auto duration = end - start;

    std::string filename;

    if (image_width == 1920 && image_height == 1080 && camera.samples_per_pixel == 1 && camera.max_depth == 10) {
        filename = string_format("ph_%dcpu_%.1lds.png", pool.get_thread_count(), duration);
    } else {
        filename = string_format("ph_%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                                 camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(),
                                 duration);
    }
    stbi_write_png(filename.c_str(), image_width, image_height, 1, pixels, 0);
    std::cerr << "Rendered in " << duration << "s" << std::endl;

    if (ENABLE_AMPLITUDE) {

        auto min = std::abs(
                *std::min_element(pixels_complex, pixels_complex + image_width * image_height, [](auto a, auto b) {
                    return std::abs(a) < std::abs(b);
                }));

        auto max = std::abs(
                *std::max_element(pixels_complex, pixels_complex + image_width * image_height, [](auto a, auto b) {
                    return std::abs(a) < std::abs(b);
                }));

        for (int i = 0; i < image_width * image_height; i++) {
            pixels[i] = static_cast<unsigned char>((std::abs(pixels_complex[i]) - min) / (max - min) * 255);
        }

        stbi_write_png(string_format("amp_%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                                     camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(),
                                     duration).c_str(),
                       image_width, image_height, 1, pixels, 0);

    }

    return 0;
}