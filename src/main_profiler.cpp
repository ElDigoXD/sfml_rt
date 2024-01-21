#define PROFILE
#include "Vec3.h"
#include "Camera.h"




int main() {
    auto const max_window_width = 1920;
    auto const max_window_height = 1080;
    auto current_width = 800u;
    auto current_height = 400u;
    auto image_width = 600u;
    auto image_height = 400u;

    auto camera = Camera();

    camera.render();
}