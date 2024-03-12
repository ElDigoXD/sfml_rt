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

#include "curand.h"
#include "cmath"

int main(int argc, char *argv[]) {
    int image_width = 256;
    int image_height = 256;
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

    unsigned char pixels[image_width * image_height * 4];
    auto camera = HoloCamera(image_width, image_height, samples_per_pixel, 10);

    curandCreateGeneratorHost(&Random::l_rand, curandRngType::CURAND_RNG_PSEUDO_DEFAULT);

    HittableList *world;
    world = CPUScene::hologram(camera);


    BS::thread_pool pool{static_cast<unsigned int>(num_threads)};

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << camera.samples_per_pixel
              << " samples per pixel " << "with " << pool.get_thread_count() << " threads.\n";

    auto start = time(nullptr);

    auto *pixels_complex = new std::complex<double>[image_width * image_height];
    camera.render_CGH(pixels_complex, &world);

    for (int i = 0; i < image_width * image_height; i++) {
        pixels[i * 4 + 0] = std::arg(pixels_complex[i]) / (2 * M_PI) * 255;
        pixels[i * 4 + 1] = std::arg(pixels_complex[i]) / (2 * M_PI) * 255;
        pixels[i * 4 + 2] = std::arg(pixels_complex[i]) / (2 * M_PI) * 255;
        pixels[i * 4 + 3] = 255;
    }

    /*
    pool.detach_loop(0, image_height, [camera, &pixels, &world](int j) {
        camera.render_pixel_line(&pixels[j * camera.image_width * 4], &world, (int) j);
    }, 50);
    pool.wait();
    */
    auto end = time(nullptr);
    auto duration = end - start;

    stbi_write_png(string_format("ph_%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                                 camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(), duration).c_str(),
                   image_width, image_height, 4, pixels, 0);

    std::cerr << "Rendered in " << duration << "s" << std::endl;

    auto min = std::abs(
            *std::min_element(pixels_complex, pixels_complex + image_width * image_height, [](auto a, auto b) {
                return std::abs(a) < std::abs(b);
            }));

    auto max = std::abs(
            *std::max_element(pixels_complex, pixels_complex + image_width * image_height, [](auto a, auto b) {
                return std::abs(a) < std::abs(b);
            }));

    for (int i = 0; i < image_width * image_height; i++) {
        pixels[i * 4 + 0] = static_cast<unsigned char>((std::abs(pixels_complex[i]) - min) / (max - min) * 255);
        pixels[i * 4 + 1] = static_cast<unsigned char>((std::abs(pixels_complex[i]) - min) / (max - min) * 255);
        pixels[i * 4 + 2] = static_cast<unsigned char>((std::abs(pixels_complex[i]) - min) / (max - min) * 255);
        pixels[i * 4 + 3] = 255;
    }

    stbi_write_png(string_format("amp_%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                                 camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(), duration).c_str(),
                   image_width, image_height, 4, pixels, 0);


}