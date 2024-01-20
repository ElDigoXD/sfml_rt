#define SFML

#include <SFML/Graphics.hpp>

#include "Vec3.h"


void a(sf::Image &image, unsigned int window_width, unsigned int window_height) {
    for (int j = 0; j < window_height; ++j) {
        for (int i = 0; i < window_width; ++i) {
            auto r = double(i) / (window_width - 1);
            auto g = double(j) / (window_height - 1);
            auto b = 0;

            image.setPixel(i, j, to_sf_color(Color(r, g, b)));
        }
    }
}

int main() {


    auto const window_width = 1920;
    auto const window_height = 1080;
    auto window = sf::RenderWindow{{window_width, window_height}, "CMake SFML Project"};
    window.setFramerateLimit(144);
    sf::Texture texture;
    texture.create(window_width, window_height);

    sf::Image image;

    image.create(window_width, window_height);

    auto *pixels = new sf::Uint8[window_width * window_height * 4];

    a(image, window_width, window_height);
    texture.update(pixels);
    texture.update(image);


    sf::Sprite sprite;
    sprite.setTexture(texture, true);

    while (window.isOpen()) {
        auto event = sf::Event{};

        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::Resized) {
                window.setView(sf::View(sf::FloatRect(0, 0, static_cast<float>(event.size.width),
                                                      static_cast<float>(event.size.height))));
                if (event.size.width <= window_width && event.size.height <= window_height) {
                    a(image, event.size.width, event.size.height);
                } else {
                    a(image, window_width, window_height);

                }
                texture.update(image);
            }
        }

        window.clear();
        window.draw(sprite);

        window.display();
    }
}

