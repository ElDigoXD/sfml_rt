#include "Vec3.h"
#include "Camera.h"
#include "chrono"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"
#include "third-party/BS_thread_pool.h"
#include "Scene.h"

static const int image_width = 1920;
static const int image_height = 1080;
static unsigned char pixels[image_width * image_height * 4];
static HittableList world;
static Camera camera = Camera(image_width, image_height, static_cast<int>(std::pow(2, 2)), 10);
static int count = 0;

static void render_pixel_line(int j) {
    camera.render_pixel_line(&pixels[j * camera.image_width * 4], world, (int) j);
    std::clog << ++count << ' ' << std::flush;
}

int main() {
    srand(1);
    world = Scene::book_1_end(camera);
    srand(time(nullptr));

    BS::thread_pool pool{4};

    //for (auto threads: {6}) {
    //    pool.reset(threads);

    auto start{std::chrono::steady_clock::now()};

    pool.detach_loop(0, image_height, render_pixel_line, 50);
    pool.wait();

    auto end{std::chrono::steady_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("%d: %.1fms\n", pool.get_thread_count(), duration.count());

    //}

    auto out = string_format("gpu_%dx%d_%d_%d_%d_%.0fms.png", image_height, image_width, camera.samples_per_pixel,
                        camera.max_depth,
                        pool.get_thread_count(), duration.count());

    stbi_write_png(out.c_str(), image_width, image_height, 4, pixels, 0);
}