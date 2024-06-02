#define SCREEN_HEIGHT_IN_PX 50
#define ENABLE_RANDOM_SCREEN_RAYS

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include "third-party/tiny_obj_loader.h"

#include "third-party/BS_thread_pool.h"

#include "utils.h"
#include "cuda.h"
#include <chrono>
#include <cfloat>
#include <curand_kernel.h>
#include <thrust/complex.h>

#include "Vec3.h"
#include "Camera.h"
#include "hittable/Sphere.h"
#include "Scene.h"
#include "third-party/argparse.hpp"

__global__ void
render(thrust::complex<double> *fb, int max_x, int max_y, HoloCamera *d_camera, HittableList **d_world,
       Point3 *point_cloud,
       int point_cloud_size, curandState *global_state) {

    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    uint pixel_index = j * max_x + i;
    curandState local_state = global_state[pixel_index];

    const auto *world = *d_world;
    auto slm_pixel_center =
            d_camera->slm_pixel_00_location + (i * d_camera->slm_pixel_delta_x) + (j * d_camera->slm_pixel_delta_y);

    for (int pi = 0; pi < point_cloud_size; pi++) {
        auto ray = Ray(slm_pixel_center, point_cloud[pi] - slm_pixel_center);
        const thrust::complex<double> cgh = d_camera->ray_wave_cgh(ray, point_cloud[pi], d_camera->max_depth, *world,
                                                                   &local_state);
        fb[pixel_index] += cgh;
    }
    fb[pixel_index] /= (d_camera->slm_width_in_px * d_camera->slm_height_in_px * 1.0);
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

struct MyArgs : public argparse::Args {
    int &image_height = kwarg("ih,image-height", "Image height in pixels").set_default(1080);
    int &screen_height = kwarg("sh,screen-height", "Screen height in pixels (for the point cloud)").set_default(SCREEN_HEIGHT_IN_PX);
    int &samples_per_pixel = kwarg("spp,samples", "Number of samples per pixel (unimplemented)").set_default(1);
    int &max_depth = kwarg("d,depth", "Maximum ray depth").set_default(10);
    int &tx = kwarg("tx", "Number of blocks in the x dimension").set_default(32);
    int &ty = kwarg("ty", "Number of blocks in the y dimension").set_default(16);
    bool &verbose = flag("v,verbose", "A flag to toggle verbose");
};

int main(int argc, char *argv[]) {
    auto args = argparse::parse<MyArgs>(argc, argv);

    const int image_width = args.image_height * 16 / 9;
    const int image_height = args.image_height;
    const int screen_height = args.screen_height;

    const int samples_per_pixel = args.samples_per_pixel;
    const int max_depth = args.max_depth;

    const int tx = args.tx;
    const int ty = args.ty;

    if (args.verbose) {
        args.print();
    }

    unsigned char pixels[image_width * image_height];

    auto start = time(nullptr);

    auto num_pixels = image_width * image_height;
    auto frame_buffer_size = num_pixels * sizeof(thrust::complex<double>);

    dim3 block(tx, ty);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);


    thrust::complex<double> *fb;
    curandState *d_global_state;
    curandState *d_global_state2;
    CU(cudaMalloc((void **) &d_global_state2, 1 * sizeof(curandState)));
    rand_init<<<1, 1>>>(d_global_state2);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    Hittable **d_list;
    HittableList **d_world;
    HoloCamera *d_camera;

    d_camera = new(true) HoloCamera(image_width, image_height, samples_per_pixel, max_depth, screen_height);

    Scene::hologram(&d_list, &d_world, *d_camera, d_global_state2);
    const HittableList *h_world = Scene::hologram_cpu(*d_camera);

    auto point_cloud = d_camera->generate_point_cloud(*h_world);

    Vec3 *d_point_cloud;

    CU(cudaMalloc((void **) &d_point_cloud, point_cloud.size() * sizeof(Vec3)));
    CU(cudaMemcpy(d_point_cloud, point_cloud.data(), point_cloud.size() * sizeof(Vec3), cudaMemcpyHostToDevice));

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << d_camera->samples_per_pixel
              << " samples per pixel " << "in " << tx << "x" << ty << " blocks.\n";

    CU(cudaMallocManaged((void **) &fb, frame_buffer_size));
    CU(cudaMalloc((void **) &d_global_state, num_pixels * sizeof(curandState)));

    render_init<<<grid, block>>>(image_width, image_height, d_global_state);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    render<<<grid, block>>>(fb, image_width, image_height,
                            d_camera,
                            d_world,
                            d_point_cloud,
                            point_cloud.size(),
                            d_global_state);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

// std::arg -> If no errors occur, this is the phase angle of z in the interval [−π; π].
// [-pi, pi] -> [0, 2pi] -> [0, 1] -> [0, 255]
    for (int i = 0; i < image_width * image_height; i++) {
        pixels[i] = static_cast<unsigned char>((thrust::arg(fb[i]) + M_PI) / (2 * M_PI) * 255);
    }

    auto end = time(nullptr);
    auto duration = end - start;

    std::string filename;

    if (image_width == 1920 && image_height == 1080 && d_camera->samples_per_pixel == 1 && d_camera->max_depth == 10) {
        filename = string_format("ph_%dgpu_%.1lds.png", THREADS, duration);
    } else {
        filename = string_format("ph_%dx%d_%d_%d_%dgpu_%.1ld.png", image_height, image_width,
                                 d_camera->samples_per_pixel, d_camera->max_depth, THREADS,
                                 duration);
    }
    stbi_write_png(filename.c_str(), image_width, image_height, 1, pixels, 0);
    std::cerr << "Rendered in " << duration << "s" << std::endl;
    printf("Image saved as: %s\n", filename.c_str());

    auto command = string_format("python3 ../propagation/main.py %s", filename.c_str());
    std::system(command.c_str());
}
