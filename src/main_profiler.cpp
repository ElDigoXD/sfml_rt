#define PROFILE
#include "Vec3.h"
#include "Camera.h"




int main() {
    auto const max_window_width = 1920;
    auto const max_window_height = 1080;
    auto current_width = 1920u;
    auto current_height = 1080u;
    auto image_width = 1920u-200;
    auto image_height = 1080u;

    auto camera = Camera();

    camera.render();
}