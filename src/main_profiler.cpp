#include "Vec3.h"
#include "Camera.h"
#include "chrono"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"

int main() {
    auto image_width = 1920;
    auto image_height = 1080;
    auto camera_10 = Camera(image_width, image_height, 2, 100);
    auto camera_100 = Camera(image_width, image_height, 100, 100);
    auto camera_1000 = Camera(image_width, image_height, 1000, 100);

    HittableList world;
    auto material_ground = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<Lambertian>(Color(0.7, 0.3, 0.3));
    auto material_left = std::make_shared<Metal>(Color(0.8, 0.8, 0.8), 0.3);
    auto material_right = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1);

    world.add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(std::make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, material_right));


    auto pixels = new unsigned char[image_width * image_height * 4];

    auto start{std::chrono::steady_clock::now()};

    camera_100.render(pixels, world);

    auto end{std::chrono::steady_clock::now()};
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("%.1fms\n", duration.count());

    stbi_write_png("out.png", image_width, image_height, 4, pixels, 0);
}