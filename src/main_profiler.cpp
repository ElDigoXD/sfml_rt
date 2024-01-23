#include "Vec3.h"
#include "Camera.h"


int main() {
    auto const max_window_width = 1920;
    auto const max_window_height = 1080;
    auto current_width = 1920u;
    auto current_height = 1080u;
    auto image_width = 1920u - 200;
    auto image_height = 1080u;

    auto camera = Camera();
    camera.update(current_width, current_height);


    HittableList world;
    world.add(std::make_shared<Sphere>(Sphere({0, 0, -1}, 0.5)));
    world.add(std::make_shared<Sphere>(Point3(0, -100.5, -1), 100));
    auto pixels = new unsigned int[current_width * current_height];

    sf::Image image;
    sf::Clock clock;
    camera.render2(pixels, world);
    image.create(current_width, current_height, (unsigned char *) (pixels));
    image.saveToFile("out.png");
    printf("%d\n", clock.restart().asMilliseconds());
    camera.render(image, world);
    image.saveToFile("out.png");
    printf("%d\n", clock.restart().asMilliseconds());
}