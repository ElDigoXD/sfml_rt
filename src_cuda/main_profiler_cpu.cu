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

int main(int argc, char *argv[]) {
    int image_width = 1920;
    int image_height = 1080;
    int samples_per_pixel = 1;
    int num_threads = 4;
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
    Camera camera = Camera(image_width, image_height, samples_per_pixel, 10);

    curandCreateGeneratorHost(&Random::l_rand, curandRngType::CURAND_RNG_PSEUDO_DEFAULT);

    HittableList *world;
    world = CPUScene::shuttle(camera);


    BS::thread_pool pool{static_cast<unsigned int>(num_threads)};

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << camera.samples_per_pixel
              << " samples per pixel " << "with " << pool.get_thread_count() << " threads.\n";

    auto start = time(nullptr);

    pool.detach_loop(0, image_height, [camera, &pixels, &world](int j) {
        camera.render_pixel_line(&pixels[j * camera.image_width * 4], &world, (int) j);
    }, 50);
    pool.wait();

    auto end = time(nullptr);
    auto duration = end - start;

    stbi_write_png(string_format("%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                                 camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(), duration).c_str(),
                   image_width, image_height, 4, pixels, 0);

    std::cerr << "Rendered in " << duration << "s" << std::endl;

}