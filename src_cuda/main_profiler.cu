#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include "third-party/tiny_obj_loader.h"

#include "third-party/BS_thread_pool.h"
#include "cuda.h"
#include <chrono>
#include <cfloat>
#include <curand_kernel.h>

#include "Vec3.h"
#include "Camera.h"
#include "Sphere.h"
#include "Scene.h"


__global__ void
render(Vec3 *fb, int max_x, int max_y, Camera *d_camera, HittableList **world, curandState *global_state) {
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    uint pixel_index = j * max_x + i;
    curandState local_state = global_state[pixel_index];
    Vec3 col(0, 0, 0);
    for (int s = 0; s < (d_camera)->samples_per_pixel; s++) {
        Ray r = (d_camera)->get_random_ray_at(i, j, &local_state);
        col += (d_camera)->ray_color(r, (d_camera)->max_depth, world, &local_state);
    }
    fb[pixel_index] = col / (d_camera)->samples_per_pixel;
}

__global__ void render_init(int max_x, int max_y, curandState *global_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(pixel_index, 0, 0, &global_state[pixel_index]);
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(0, 0, 0, rand_state);
    }
}

int main(int argc, char *argv[]) {
    int image_width = 1920;
    int image_height = 1080;
    int samples_per_pixel = 10;
    int tx = 8, ty = 8;
    if (argc != 1) {
        if (argc > 1) {
            samples_per_pixel = std::atoi(argv[1]);
        }
        if (argc > 2) {
            tx = std::atoi(argv[2]);
            ty = std::atoi(argv[2]);
        }
        if (argc > 3) {
            image_width = std::atoi(argv[3]) * 16 / 9;
            image_height = std::atoi(argv[3]);
        }
    }

    unsigned char pixels[image_width * image_height * 3];

    auto start = time(nullptr);

    auto num_pixels = image_width * image_height;
    auto frame_buffer_size = num_pixels * sizeof(Vec3);

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);


    Vec3 *fb;
    curandState *d_global_state;
    curandState *d_global_state2;
    CU(cudaMalloc((void **) &d_global_state2, 1 * sizeof(curandState)));
    rand_init<<<1, 1>>>(d_global_state2);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    Hittable **d_list;
    HittableList **d_world;
    Camera *d_camera;

    d_camera = new(true) Camera(image_width, image_height, samples_per_pixel, 100);

    load_scene(ObjScene::shuttle);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << d_camera->samples_per_pixel
              << " samples per pixel " << "in " << tx << "x" << ty << " blocks.\n";

    CU(cudaMallocManaged((void **) &fb, frame_buffer_size));
    CU(cudaMalloc((void **) &d_global_state, num_pixels * sizeof(curandState)));

    render_init<<<blocks, threads>>>(image_width, image_height, d_global_state);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, image_width, image_height,
                                d_camera,
                                d_world,
                                d_global_state);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());


    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            //auto rgba_color = to_gamma_color(fb[i + (image_height - j) * image_width]);
            auto rgba_color = to_gamma_color(fb[i + j * image_width]);
            int pixel_index = j * image_width * 3 + i * 3;
            pixels[pixel_index + 0] = static_cast<unsigned char>(rgba_color.r * 255);
            pixels[pixel_index + 1] = static_cast<unsigned char>(rgba_color.g * 255);
            pixels[pixel_index + 2] = static_cast<unsigned char>(rgba_color.b * 255);
        }
    }

    auto end = time(nullptr);
    auto duration = end - start;


    stbi_write_png(string_format("%dx%d_%d_%d_gpu_%.0ld.png", image_height, image_width,
                                 d_camera->samples_per_pixel, d_camera->max_depth, duration).c_str(),
                   image_width, image_height, 3, pixels, 0);

    std::cerr << "Rendered in " << duration << "s" << std::endl;

    // clean up
    // CU(cudaDeviceSynchronize());
}
