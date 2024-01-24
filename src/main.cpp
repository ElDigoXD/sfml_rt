#include "imgui.h"
#include "imgui-SFML.h"
#include "third-party/BS_thread_pool.h"
#include <SFML/Graphics.hpp>
#include <thread>

#include "utils.h"
#include "Vec3.h"
#include "Camera.h"
#include "Material.h"

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
    unsigned char *pixels = new unsigned char[max_window_width * max_window_height * 4];

    enum TState : int {
        IDLE = -1,
        RENDERING = 0,
    };

    unsigned int t_n;
    TState t_state = IDLE;
    BS::thread_pool pool{1};

    int render_time = 0;
    int render_update_ms;

    sf::Clock update_texture_clock;
    sf::Clock delta_clock;
    sf::Clock render_clock;


    GUI() {
        // Imgui variables
        render_update_ms = 1;
        t_n = 4;
        pool.reset(t_n);
        camera.samples_per_pixel = 100;
        camera.max_depth = 100;

        auto material_ground = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
        auto material_center = std::make_shared<Lambertian>(Color(0.7, 0.3, 0.3));
        auto material_left = std::make_shared<Metal>(Color(0.8, 0.8, 0.8), 0);
        auto material_right = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1);

        world.add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, material_ground));
        world.add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, material_center));
        world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, material_left));
        world.add(std::make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, material_right));
    }

    void run() {
        // Gui stuff
        window.setFramerateLimit(165);
        auto _ = ImGui::SFML::Init(window);
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        texture.create(max_window_width, max_window_height);
        sprite.setTexture(texture, true);

        // World declaration
        // old
        // world.add(std::make_shared<Sphere>(Point3(0, 0, -1), 0.5,
        //                                    std::make_shared<Lambertian>(Colors::white/2)));
        // world.add(std::make_shared<Sphere>(Point3(0, -100.5, -1), 100,
        //                                   std::make_shared<Lambertian>(Colors::white/2)));
        // new



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

            if (t_state != IDLE) {
                if (pool.get_tasks_total() > 0) {
                    if (update_texture_clock.getElapsedTime().asMilliseconds() > render_update_ms) {
                        update_texture_clock.restart();
                        texture.update(pixels, image_width, image_height, 0, 0);
                    }
                } else {
                    t_state = IDLE;
                    texture.update(pixels, image_width, image_height, 0, 0);
                    render_time = render_clock.getElapsedTime().asMilliseconds();
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

    void stop_render() {
        pool.purge();
    }

    void start_render() {
        stop_render();

        render_clock.restart();
        memset(pixels, 0, max_window_width * max_window_height * 4);
        texture.update(pixels, image_width, image_height, 0, 0);

        pool.detach_loop(0U, image_height, [this](int j) {
            camera.render_pixel_line(&pixels[j * camera.image_width * 4], world, (int) j);
        }, 50);
        t_state = RENDERING;
    };

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
        if (t_state == RENDERING) {
            if (ImGui::Button("Stop render", {-1, 0})) {
                stop_render();
            }
        } else {
            if (ImGui::Button("Start render", {-1, 0})) {
                start_render();
            }
        }
        ImGui::Text("Render update:");
        ImGui::SliderInt("##a", &render_update_ms, 1, 1000, "%dms");
        ImGui::BeginDisabled(t_state == RENDERING);
        ImGui::Text("Render threads:");
        if (ImGui::SliderInt("##b", (int *) &t_n, 1, (int) std::thread::hardware_concurrency()))
            pool.reset(t_n);
        ImGui::EndDisabled();
        ImGui::Text("Samples per pixel:");
        ImGui::SliderInt("##c", (int *) &camera.samples_per_pixel, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::Text("Ray depth:");
        ImGui::SliderInt("##d", (int *) &camera.max_depth, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic);


        ImGui::Text("%dx%d (%.2f) %d", image_width, image_height, 1 / dt.asSeconds(), t_state);
        ImGui::Text("Render: %dms",
                    t_state == RENDERING ? render_clock.getElapsedTime().asMilliseconds() : render_time);
        if (ImGui::Button("Save render", {-1, 0})) {
            sf::Image image;
            image.create(image_width, image_height, pixels);
            image.saveToFile("out.png");
        }
        ImGui::End();
    }

};

int main() {
    GUI gui;
    gui.run();
}

