#include "imgui.h"
#include "imgui-SFML.h"

#include <SFML/Graphics.hpp>
#include <thread>

#include "Vec3.h"
#include "Camera.h"

class GUI {
public:
    int const max_window_width = 3440;
    int const max_window_height = 1440;
    unsigned int current_width = 800u;
    unsigned int current_height = 400u;
    unsigned int image_width = 600u;
    unsigned int image_height = 400u;

    sf::RenderWindow window = sf::RenderWindow{{current_width, current_height}, "CMake SFML Project"};
    sf::Sprite sprite;

    Camera camera = Camera();
    HittableList world;
    sf::Texture texture;
    unsigned int *pixels = new unsigned int[max_window_width * max_window_height];

    enum TState : int {
        IDLE = -1,
        RENDERING = 0,
        DONE = 1,
    };

    unsigned int t_n = std::thread::hardware_concurrency();
    volatile TState t_state = IDLE;
    volatile TState *ts_state = new TState[t_n];
    std::jthread t;
    std::jthread *ts = new std::jthread[t_n];

    int render_time = 0;
    int render_update_ms = 100;

    sf::Clock update_texture_clock;
    sf::Clock delta_clock;
    sf::Clock render_clock;


    GUI() {

    }

    void run() {
        // Gui stuff
        window.setFramerateLimit(165);
        auto _ = ImGui::SFML::Init(window);
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        texture.create(max_window_width, max_window_height);
        sprite.setTexture(texture, true);

        // World declaration
        world.add(std::make_shared<Sphere>(Sphere({0, 0, -1}, 0.5)));
        world.add(std::make_shared<Sphere>(Point3(0, -100.5, -1), 100));

        t_n = 1;

        start_render();

        while (window.isOpen()) {
            auto event = sf::Event{};

            while (window.pollEvent(event)) {
                ImGui::SFML::ProcessEvent(window, event);
                if (event.type == sf::Event::Closed) {
                    window.close();
                } else if (event.type == sf::Event::Resized) {
                    current_width = std::min(event.size.width, (unsigned int) max_window_width);
                    current_height = std::min(event.size.height, (unsigned int) max_window_height);

                    window.setView(sf::View(sf::FloatRect(0, 0,
                                                          static_cast<float>(current_width),
                                                          static_cast<float>(current_height))));
                    render_clock.restart();
                    image_width = current_width - 200;
                    image_height = current_height;
                    camera.update(static_cast<int>(image_width), static_cast<int>(image_height));

                    start_render();
                }
            }

            // Render thread process
            if (t_state != IDLE) {
                render_time = render_clock.getElapsedTime().asMilliseconds();
                t_state = DONE;

                for (int i = 0; i < t_n; ++i) {
                    if (ts_state[i] == RENDERING) {
                        t_state = RENDERING;
                        break;
                    }
                }

                if (t_state == DONE || update_texture_clock.getElapsedTime().asMilliseconds() > render_update_ms) {
                    if (t_state == DONE) {
                        for (int i = 0; i < t_n; ++i) {
                            if (ts[i].joinable()) {
                                ts[i].join();
                            }
                        }
                        t_state = IDLE;
                        printf("Rendered: %dx%d\tin %dms\n", image_height, image_width, render_time);
                    }
                    texture.update((unsigned char *) pixels, image_width, image_height, 0, 0);
                    update_texture_clock.restart();
                }
            }

            // Imgui
            imgui();

            window.clear();
            window.draw(sprite);
            ImGui::SFML::Render(window);
            window.display();
        }
    }


    void start_render() {
        for (int i = 0; i < t_n; ++i) {
            ts[i].request_stop();
            if (ts[i].joinable()) ts[i].join();
        }

        render_clock.restart();
        memset(pixels, 0, max_window_width * max_window_height * 4);
        texture.update((unsigned char *) pixels, image_width, image_height, 0, 0);
        auto length = image_height * 1.0 / t_n;
        for (int i = 0; i < t_n; ++i) {
            ts_state[i] = RENDERING;
            ts[i] = std::jthread(&GUI::concurrent_renderer_thread, this, std::floor(i * length), length + 1, i);
        }
        t_state = RENDERING;
    };

    void concurrent_renderer_thread(const std::stop_token &stop, unsigned int offset, unsigned int length, int i) {
        auto top = std::min(offset + length, image_height);
        for (unsigned int j = offset; j < top && !stop.stop_requested(); ++j) {
            camera.render_pixel_line(&pixels[j * camera.image_width], world, (int) j);
        }
        if (!stop.stop_requested()) {
            ts_state[i] = DONE;
        }
    }


    void imgui() {
        const sf::Time &dt = delta_clock.restart();
        ImGui::SFML::Update(window, dt);
        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoResize;
        window_flags |= ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoTitleBar;

        ImGui::SetNextWindowPos({static_cast<float>(current_width - 200), 0});
        ImGui::SetNextWindowSize({200, static_cast<float>(current_height)});
        ImGui::SetNextWindowBgAlpha(1);
        ImGui::Begin("Hello, world!", nullptr, window_flags);
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::Button("Re-render", {-1, 0})) {
            start_render();
        }
        ImGui::Text("Render update:");
        ImGui::SliderInt("##a", &render_update_ms, 1, 1000, "%dms");
        ImGui::Text("Render threads:");
        ImGui::SliderInt("##b", (int *) &t_n, 1, (int) std::thread::hardware_concurrency());
        ImGui::Text("Samples per pixel:");
        ImGui::SliderInt("##c", (int *) &camera.samples_per_pixel, 1, 100);


        ImGui::Text("%dx%d (%.2f) %d", image_width, image_height, 1 / dt.asSeconds(), t_state);
        ImGui::Text("Render: %dms", render_time);
        if (ImGui::Button("Save render", {-1, 0})) {
            sf::Image image;
            image.create(image_width, image_height, (unsigned char *) pixels);
            image.saveToFile("out.png");
        }
        ImGui::End();

    }

};


int main() {
    GUI gui;
    gui.run();
}

