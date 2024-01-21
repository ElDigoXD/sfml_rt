#include "imgui.h"
#include "imgui-SFML.h"

#include <SFML/Graphics.hpp>

#include "Vec3.h"
#include "Camera.h"


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
    auto const max_window_width = 1920;
    auto const max_window_height = 1080;
    auto current_width = 800u;
    auto current_height = 400u;
    auto image_width = 600u;
    auto image_height = 400u;

    auto camera = Camera();

    auto window = sf::RenderWindow{{current_width, current_height}, "CMake SFML Project"};
    window.setFramerateLimit(165);
    auto _ = ImGui::SFML::Init(window);
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    sf::Texture texture;
    texture.create(max_window_width, max_window_height);

    sf::Image image;

    image.create(max_window_width, max_window_height);

    auto *pixels = new sf::Uint8[max_window_width * max_window_height * 4];

    a(image, image_width, image_height);
    texture.update(pixels);

    int render_time;
    sf::Clock render_clock;
    camera.render(image);
    render_time = render_clock.getElapsedTime().asMilliseconds();
    texture.update(image);


    sf::Sprite sprite;
    sprite.setTexture(texture, true);

    sf::Clock delta_clock;
    while (window.isOpen()) {
        auto event = sf::Event{};

        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::Resized) {
                current_width = event.size.width;
                current_height = event.size.height;
                image_width = current_width-200;
                image_height = current_height;
                window.setView(sf::View(sf::FloatRect(0, 0,
                                                      static_cast<float>(current_width),
                                                      static_cast<float>(current_height))));
                if (current_width <= max_window_width && current_height <= max_window_height) {
                    a(image, image_width, image_height);
                    render_clock.restart();
                    camera.update(static_cast<int>(image_width), static_cast<int>(image_height));
                    camera.render(image);
                    render_time = render_clock.getElapsedTime().asMilliseconds();
                } else {
                    a(image, max_window_width, max_window_height);
                }
                texture.update(image);
                //image.saveToFile("out.png");
            }
        }

        const sf::Time &dt = delta_clock.restart();
        ImGui::SFML::Update(window, dt);
        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoResize;
        window_flags |= ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoTitleBar;

        ImGui::SetNextWindowPos({static_cast<float>(current_width-200),0});
        ImGui::SetNextWindowSize({200, max_window_height});
        ImGui::SetNextWindowBgAlpha(1);
        ImGui::Begin("Hello, world!", nullptr, window_flags);
        if (ImGui::Button("Export image")){
            image.saveToFile("image.png");
            // Todo: Trim image
        }
        ImGui::Text("%dx%d (%.2f)", image_width, image_height,  1/dt.asSeconds());
        ImGui::Text("Render: %dms", render_time);
        ImGui::End();

        window.clear();
        window.draw(sprite);
        ImGui::SFML::Render(window);
        window.display();
    }
}