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
        DONE = 1,
    };

    unsigned int t_n;
    TState t_state = IDLE;
    BS::thread_pool pool{1};

    int render_time = 0;
    int render_update_ms;

    sf::Clock update_texture_clock;
    sf::Clock delta_clock;
    sf::Clock render_clock;

    std::shared_ptr<Lambertian> material_ground;
    std::shared_ptr<Lambertian> material_center;
    std::shared_ptr<Metal> material_left;
    std::shared_ptr<Metal> material_right;

    Color *colors_agg = new Color[max_window_width * max_window_height];
    Color *colors = new Color[max_window_width * max_window_height];

    int pass_number{1};
    bool continuous_render;

    int continuous_render_sample_limit;

    GUI() {
        // Imgui variables
        render_update_ms = 1;
        t_n = 4;
        pool.reset(t_n);
        camera.samples_per_pixel = 10;
        camera.max_depth = 100;

        continuous_render = true;
        continuous_render_sample_limit = 100;

        material_ground = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
        material_center = std::make_shared<Lambertian>(Color(0.7, 0.3, 0.3));
        material_left = std::make_shared<Metal>(Color(0.8, 0.8, 0.8), 0.3);
        material_right = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 1);

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
                    return;
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
                if (continuous_render) {
                    if (pool.get_tasks_total() == 0 && t_state != DONE) {
                        for (int i = 0; i < image_width * image_height; ++i) {
                            colors_agg[i] = (colors_agg[i] * (pass_number - 1) + colors[i]) / pass_number;
                            auto rgba_color = to_gamma_color(colors_agg[i]);
                            pixels[i * 4 + 0] = static_cast<unsigned char>(rgba_color.r * 255);
                            pixels[i * 4 + 1] = static_cast<unsigned char>(rgba_color.g * 255);
                            pixels[i * 4 + 2] = static_cast<unsigned char>(rgba_color.b * 255);
                            pixels[i * 4 + 3] = 255;
                        }
                        texture.update(pixels, image_width, image_height, 0, 0);
                        pass_number++;
                        if (pass_number * camera.samples_per_pixel < continuous_render_sample_limit) {
                            if (pass_number == 10) {
                                pass_number /= 2;
                                camera.samples_per_pixel *= 2;
                            }
                            pool.detach_loop(0U, image_height, [this](int j) {
                                camera.render_color_line(&colors[j * camera.image_width], world, (int) j);
                            }, 50);
                        } else {
                            t_state = IDLE;
                            render_time = render_clock.getElapsedTime().asMilliseconds();
                        }

                    }
                } else if (pool.get_tasks_total() > 0) {
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
        t_state = DONE;
        render_time = render_clock.getElapsedTime().asMilliseconds();
    }

    void start_render() {
        stop_render();
        pass_number = 1;
        render_clock.restart();
        memset(pixels, 0, max_window_width * max_window_height * 4);

        texture.update(pixels, image_width, image_height, 0, 0);

        if (continuous_render) {
            camera.samples_per_pixel = 1;
            pool.detach_loop(0U, image_height, [this](int j) {
                camera.render_color_line(&colors[j * camera.image_width], world, (int) j);
            }, 50);
        } else {
            pool.detach_loop(0U, image_height, [this](int j) {
                camera.render_pixel_line(&pixels[j * camera.image_width * 4], world, (int) j);
            }, 50);
        }
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

        ImGui::GetStyle().WindowBorderSize = 0;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        ImGui::SetNextWindowPos({(float) (current_width - 200), 0});
        ImGui::SetNextWindowSize({200, (float) (current_height)});
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


        if (t_state == RENDERING) {
            if (continuous_render) {
                ImGui::ProgressBar((float) pass_number * camera.samples_per_pixel /
                                   continuous_render_sample_limit); // NOLINT(*-narrowing-conversions)
            } else {
                ImGui::ProgressBar((50.0f - (float) pool.get_tasks_total()) / 50);
            }
        } else {
            if (ImGui::Button("Save render", {-1, 0})) {
                sf::Image image;
                image.create(image_width, image_height, pixels);
                image.saveToFile("out.png");
            }
        }
        ImGui::Checkbox("Continuous render", &continuous_render);

        ImGui::Text("Render update:");
        ImGui::SliderInt("##a", &render_update_ms, 1, 1000, "%dms");

        ImGui::BeginDisabled(t_state == RENDERING);
        ImGui::Text("Render threads:");
        if (ImGui::SliderInt("##b", (int *) &t_n, 1, (int) std::thread::hardware_concurrency()))
            pool.reset(t_n);
        ImGui::EndDisabled();

        if (continuous_render) {
            ImGui::Text("Stop at n samples:");
            ImGui::SliderInt("##c", (int *) &continuous_render_sample_limit, 10, 10000, "%d",
                             ImGuiSliderFlags_Logarithmic);
        } else {
            ImGui::Text("Samples per pixel:");
            ImGui::SliderInt("##c", (int *) &camera.samples_per_pixel, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic);
        }

        ImGui::Text("Ray depth:");
        ImGui::SliderInt("##d", (int *) &camera.max_depth, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic);

        ImGui::Separator();

        bool re_render_on_material_change = true;
        ImGui::Checkbox("Render on change", &re_render_on_material_change);
        if (ImGui::CollapsingHeader("Materials")) {
            if (a(material_center) && re_render_on_material_change) {
                start_render();
            };
        }
        ImGui::Separator();
        ImGui::Text("%dx%d %d samples, %4.0ffps", image_width, image_height, pass_number * camera.samples_per_pixel,
                    1 / dt.asSeconds());
        ImGui::Text("Render: %dms",
                    t_state == RENDERING ? render_clock.getElapsedTime().asMilliseconds() : render_time);

        ImGui::PopStyleColor();
        ImGui::End();
    }

    template<class T = Material>
    bool a(T material) {
        auto updated = false;
        if (ImGui::TreeNode("Center")) {
            if constexpr (std::is_same<T, Metal>::value) {
                ImGui::SliderScalar("a", ImGuiDataType_Double, &material.fuzz);
            }

            ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
            float mat[3];
            to_float_array(material_center->albedo, mat);
            if (ImGui::ColorPicker3("A", mat)) {
                material_center->albedo = from_float_array(mat);
                updated = true;
            }
            ImGui::TreePop();
        }
        return updated;

    }


};

int main() {
    GUI gui;
    gui.run();
    exit(0);
}

