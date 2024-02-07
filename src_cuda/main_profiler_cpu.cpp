#include "Vec3.h"
#include "Camera.h"
#include "chrono"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"
#include "third-party/BS_thread_pool.h"
#include "Scene.h"

#include "curand.h"

static const int image_width = 1920;
static const int image_height = 1080;
static unsigned char pixels[image_width * image_height * 4];
static HittableList *world;
static Camera camera = Camera(image_width, image_height, 1, 10);

static void render_pixel_line(int j) {
    camera.render_pixel_line(&pixels[j * camera.image_width * 4], *world, (int) j);
}

int main() {
    curandCreateGeneratorHost(&Random::l_rand, curandRngType::CURAND_RNG_PSEUDO_DEFAULT);

    world = CPUScene::book_1_end(camera);

    Random::rand = true;

    BS::thread_pool pool{4};

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << camera.samples_per_pixel
              << " samples per pixel " << "with " << pool.get_thread_count() << " threads.\n";

    auto start = time(nullptr);

    pool.detach_loop(0, image_height, render_pixel_line, 50);
    pool.wait();

    auto end = time(nullptr);
    auto duration = end - start;

    stbi_write_png(string_format("%dx%d_%d_%d_%dcpu_%.1ld.png", image_height, image_width,
                          camera.samples_per_pixel, camera.max_depth, pool.get_thread_count(), duration).c_str(),
            image_width, image_height, 4, pixels, 0);

    std::cerr << "Rendered in " << duration << "s" << std::endl;

}