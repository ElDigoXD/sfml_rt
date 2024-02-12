#include "imgui.h"
#include "imgui-SFML.h"

#define BS_THREAD_POOL_ENABLE_PAUSE

#include "third-party/BS_thread_pool.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include "third-party/tiny_obj_loader.h"
#include <SFML/Graphics.hpp>
#include <thread>
#include <type_traits>

#include "utils.h"
#include "Vec3.h"
#include "Camera.h"
#include "Material.h"
#include "Scene.h"
#include "Triangle.h"
#include "obj.h"
#include "BVHNode.h"

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

    Camera camera;
    HittableList *world;
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
    std::jthread t;

    int render_time = 0;
    int render_update_ms;

    sf::Clock update_texture_clock;
    sf::Clock delta_clock;
    sf::Clock render_clock;

    Sphere *look_at;

    Sphere *selected_hittable = nullptr;

    Color *colors_agg = new Color[max_window_width * max_window_height];
    Color *colors = new Color[max_window_width * max_window_height];

    int total_samples{0};
    int next_samples_per_pixel{1};
    bool continuous_render;

    int target_samples;
    bool enable_camera_movement;

    sf::Vector2i before_click_mouse_position;
    sf::Vector2i before_move_mouse_position;
    bool mouse_pressed = false;
    std::vector<Hittable *> *triangles = new std::vector<Hittable *>;

    GUI() {
        // Imgui variables
        camera = Camera(image_width, image_height);

        render_update_ms = 1;
        t_n = 4;
        pool.reset(t_n);
        camera.samples_per_pixel = 10;
        camera.max_depth = 100;
        continuous_render = true;
        enable_camera_movement = true;
        target_samples = 2;

        //world = CPUScene::point_light(camera);

        //triangles = Obj::get_triangles();
//
        //world = new HittableList(&triangles->at(0), triangles->size());
        //camera.look_from = {-6.31, 4.55, 3.32};
        //camera.look_at = {-1.5, -1.1, -0.8};
        //camera.light = {0, 10, 10};
        //camera.light_color = {1, 1, 1};

        world = CPUScene::shuttle(camera);
    }

    void run() {
        // Gui stuff
        window.setFramerateLimit(60);
        auto _ = ImGui::SFML::Init(window);
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        texture.create(max_window_width, max_window_height);
        sprite.setTexture(texture, true);

        start_render();
        sf::Time dt;
        while (window.isOpen()) {
            dt = delta_clock.restart();
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
                } else if (event.type == sf::Event::MouseButtonPressed) {
                    if ((event.mouseButton.button == sf::Mouse::Left
                         || event.mouseButton.button == sf::Mouse::Right)
                        && enable_camera_movement
                        && sprite.getGlobalBounds().contains((float) event.mouseButton.x,
                                                             (float) event.mouseButton.y)
                        && sf::Mouse::getPosition(window).x < image_width) {
                        if (is_materials_tab_open) {
                            /*
                            auto ray = camera.get_ray_at(sf::Mouse::getPosition(window).x,
                                                         sf::Mouse::getPosition(window).y);
                            HitRecord record;
                            for (const auto &hittable: world.objects) {
                                if (hittable->hit(ray, Interval{0, INFINITY}, record)) {
                                    selected_hittable = std::dynamic_pointer_cast<Sphere>(hittable);
                                }
                            }
                            */

                        } else {
                            mouse_pressed = true;
                            window.setMouseCursorVisible(false);
                            before_click_mouse_position = sf::Mouse::getPosition(window);
                            before_move_mouse_position = before_click_mouse_position;
                        }
                    }
                } else if (event.type == sf::Event::MouseButtonReleased) {
                    if (mouse_pressed
                        && (event.mouseButton.button == sf::Mouse::Left
                            || event.mouseButton.button == sf::Mouse::Right)) {
                        mouse_pressed = false;
                        window.setMouseCursorVisible(true);
                        sf::Mouse::setPosition(before_click_mouse_position, window);
                    }
                } else if (event.type == sf::Event::MouseWheelMoved
                           && window.hasFocus()
                           && enable_camera_movement
                           &&
                           sprite.getGlobalBounds().contains((float) event.mouseButton.x,
                                                             (float) event.mouseButton.y)
                           && sf::Mouse::getPosition(window).x < image_width) {
                    camera.vfov -= event.mouseWheel.delta;

                    update_camera_and_start_render();
                }
            }

            handle_mouse();

            handle_keyboard(dt);


            if (t_state != IDLE) {
                if (continuous_render) {
                    if (pool.get_tasks_total() == 0 && t_state != DONE) {
                        colors_to_pixel();
                        texture.update(pixels, image_width, image_height, 0, 0);
                        total_samples += camera.samples_per_pixel;
                        camera.samples_per_pixel = next_samples_per_pixel;
                        if (total_samples <= target_samples) {
                            pool.detach_loop(0U, image_height, [this](int j) {
                                camera.render_color_line(&colors[j * camera.image_width], &world, (int) j);
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
            imgui(dt);

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
        total_samples = 1;
        render_clock.restart();
        memset(pixels, 0, max_window_width * max_window_height * 4);

        if (continuous_render) {
            camera.samples_per_pixel = 1;
            pool.detach_loop(0U, image_height, [this](int j) {
                camera.render_color_line(&colors[j * camera.image_width], &world, (int) j);
            }, 50);
        } else {
            pool.detach_loop(0U, image_height, [this](int j) {
                camera.render_pixel_line(&pixels[j * camera.image_width * 4], &world, (int) j);
            }, 50);
        }
        t_state = RENDERING;
    };

    void colors_to_pixel() {
        for (int i = 0; i < image_width * image_height; ++i) {
            colors_agg[i] = (colors_agg[i] * (total_samples - 1) + colors[i] * camera.samples_per_pixel) /
                            (total_samples - 1 + camera.samples_per_pixel);
            auto rgba_color = to_gamma_color(colors_agg[i]);
            pixels[i * 4 + 0] = static_cast<unsigned char>(rgba_color.r * 255);
            pixels[i * 4 + 1] = static_cast<unsigned char>(rgba_color.g * 255);
            pixels[i * 4 + 2] = static_cast<unsigned char>(rgba_color.b * 255);
            pixels[i * 4 + 3] = 255;
        }
    }

    void handle_mouse() {
        auto rodrigues_rotation = [](Vec3 v, Vec3 k, double deg) -> Vec3 {
            return v * cos(deg) + cross(k, v) * sin(deg) + k * dot(k, v) * (1 - cos(deg));
        };
        if (mouse_pressed) {
            auto new_position = sf::Mouse::getPosition(window);
            sf::Mouse::setPosition(before_click_mouse_position, window);
            auto delta = before_move_mouse_position - new_position;
            if (delta != sf::Vector2i(0, 0)) {

                auto new_look_at = rodrigues_rotation(
                        camera.camera_center - camera.look_at, camera.u, (delta.y) / 5000.0);
                new_look_at = rodrigues_rotation(
                        new_look_at, Vec3(0, 1, 0), (delta.x) / 5000.0);

                camera.look_at = (camera.camera_center - new_look_at);
                update_camera_and_start_render();
            }
        }
    }

    void handle_keyboard(sf::Time dt) {
        auto speed = 0.003 * dt.asMilliseconds();
        if (!window.hasFocus() || !enable_camera_movement) return;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
            auto displacement = speed * cross(Vec3(0, 1, 0), camera.u);
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();
        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            auto displacement = -speed * cross(Vec3(0, 1, 0), camera.u);
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
            auto displacement = -speed * camera.u;
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();
        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
            auto displacement = speed * camera.u;
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            auto displacement = speed * Vec3(0, 1, 0);
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();

        } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {
            auto displacement = -speed * Vec3(0, 1, 0);
            camera.look_at += displacement;
            camera.look_from += displacement;

            update_camera_and_start_render();
        }
    }

    void update_camera_and_start_render() {
        camera.update();

        look_at->center = camera.look_at;
        if (continuous_render) {
            colors_to_pixel();
        }
        texture.update(pixels, image_width, image_height, 0, 0);
        start_render();
    }

    bool enable_look_at = false;
    bool is_materials_tab_open;

    void imgui(sf::Time dt) {
        ImGui::SFML::Update(window, dt);
        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoResize;
        window_flags |= ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoTitleBar;

        ImGui::GetStyle().WindowBorderSize = 0;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        ImGui::PushItemWidth(-1.0f);
        ImGui::SetNextWindowPos({(float) (current_width - 200), 0});
        ImGui::SetNextWindowSize({200, (float) (current_height)});
        ImGui::SetNextWindowBgAlpha(1);
        ImGui::Begin("Hello, world!", nullptr, window_flags);
        ImGui::BeginTabBar("tab bar");

        if (ImGui::BeginTabItem("Render")) {
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
                    ImGui::ProgressBar((float) (total_samples - 1) /
                                       target_samples); // NOLINT(*-narrowing-conversions)
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

            ImGui::BeginDisabled(t_state == RENDERING);
            if (ImGui::Checkbox("Continuous render", &continuous_render)) {
                if (continuous_render) {
                    target_samples = camera.samples_per_pixel;
                } else {
                    camera.samples_per_pixel = target_samples;
                }
            }
            ImGui::EndDisabled();

            if (continuous_render) {
                ImGui::Text("Samples per pixel:");
                if (ImGui::SliderInt("##a", (int *) &next_samples_per_pixel, 1, 10000, "%d",
                                     ImGuiSliderFlags_Logarithmic)) {
                    int a = std::floor(std::sqrt(next_samples_per_pixel));
                    int b = (int) std::pow(a, 2);
                    int c = (int) std::pow(a + 1, 2);
                    if (next_samples_per_pixel - b < c - next_samples_per_pixel)
                        next_samples_per_pixel = b;
                    else
                        next_samples_per_pixel = c;
                }
            } else {
                ImGui::Text("Render update:");
                ImGui::SliderInt("##a", &render_update_ms, 1, 1000, "%dms");
            }

            ImGui::BeginDisabled(!continuous_render && t_state != IDLE);
            ImGui::Text("Render threads:");
            if (ImGui::SliderInt("##b", (int *) &t_n, 1, (int) std::thread::hardware_concurrency())) {
                pool.reset(t_n);
            }
            ImGui::EndDisabled();

            if (continuous_render) {
                ImGui::Text("Stop at samples:");
                if (ImGui::SliderInt("##c", (int *) &target_samples, 1, 10000, "%d",
                                     ImGuiSliderFlags_Logarithmic)) {
                    int a = std::floor(std::sqrt(target_samples));
                    int b = (int) std::pow(a, 2);
                    int c = (int) std::pow(a + 1, 2);
                    if (target_samples - b < c - target_samples)
                        target_samples = b;
                    else
                        target_samples = c;
                }
            } else {
                ImGui::Text("Samples per pixel:");
                if (ImGui::SliderInt("##c", (int *) &camera.samples_per_pixel, 1, 10000, "%d",
                                     ImGuiSliderFlags_Logarithmic)) {
                    int a = std::floor(std::sqrt(camera.samples_per_pixel));
                    int b = (int) std::pow(a, 2);
                    int c = (int) std::pow(a + 1, 2);
                    if (camera.samples_per_pixel - b < c - camera.samples_per_pixel)
                        camera.samples_per_pixel = b;
                    else
                        camera.samples_per_pixel = c;
                }
            }

            ImGui::Text("Ray depth:");
            ImGui::SliderInt("##d", (int *) &camera.max_depth, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic);

            ImGui::EndTabItem();
            ImGui::PopItemWidth();
        }
        if (ImGui::BeginTabItem("Camera")) {
            ImGui::PushItemWidth(-1.0f);
            ImGui::Checkbox("Enable camera movement", &enable_camera_movement);
            ImGui::Text("Fov:");
            if (ImGui::SliderDouble("##e", &camera.vfov, 1, 200)) {
                camera.update();
                start_render();
            }

            ImGui::Text("Defocus:");
            if (ImGui::SliderDouble("##i", &camera.defocus_angle, 0, 5)) {
                camera.update();
                start_render();
            }

            ImGui::Text("Look from:");
            if (ImGui::DragDouble3("##from", camera.look_from.e, 0.1, -100, 100)) {
                camera.update();
                start_render();
            };
            ImGui::Text("Look at:");
            if (ImGui::DragDouble3("##at", camera.look_at.e, 0.1, -100, 100)) {
                camera.update();
                start_render();
            };

            /*
            if (ImGui::Checkbox("Draw look at", &enable_look_at)) {
                if (enable_look_at)
                    world.objects.push_back(look_at);
                else
                    world.objects.pop_back();
                start_render();
            }
            */

            ImGui::Text("Focus distance:");
            if (ImGui::DragDouble("##focus", &camera.focus_dist, 0.1, 0.1, 100, "%.3f",
                                  ImGuiSliderFlags_AlwaysClamp)) {
                camera.set_focus_dist(camera.focus_dist);
                update_camera_and_start_render();
            };

            ImGui::EndTabItem();
            ImGui::PopItemWidth();
        }


        ImGui::Separator();

        if (ImGui::BeginTabItem("gpu")) {
            ImGui::PushItemWidth(-1.0f);


            ImGui::EndTabItem();
            ImGui::PopItemWidth();
        }

        is_materials_tab_open = ImGui::BeginTabItem("Materials");
        if (is_materials_tab_open) {
            ImGui::PushItemWidth(-1.0f);

            ImGui::Text("Light position:");
            if (ImGui::DragDouble3("##lp", camera.light.e, 0.1, -100, 100)) {
                start_render();
            };

            ImGui::Text("Light color:");
            float mat[3];
            to_float_array(camera.light_color, mat);
            if (ImGui::ColorEdit3("##lc", mat)) {
                camera.light_color = from_float_array(mat);
                start_render();
            };

            ImGui::Text("Intensity (diff, spec, sky):");
            if (ImGui::DragDouble3("##int", camera.intensity, 0.01, 0, 2)) {
                start_render();
            };

            ImGui::Text("Shinyness:");
            if (ImGui::SliderInt("##shi", &camera.shinyness, 1, 10000, "%d", ImGuiSliderFlags_Logarithmic)) {
                start_render();
            };

            if (selected_hittable) {
                /*
                if (imgui_mat((selected_hittable->material.get()))) {
                    start_render();
                }
                 */
            }

            ImGui::EndTabItem();
            ImGui::PopItemWidth();
        }
        ImGui::EndTabBar();

        ImGui::Separator();
        ImGui::Text("%dx%d %d samples, %4.0ffps", image_width, image_height,
                    continuous_render ? total_samples - 1 : camera.samples_per_pixel,
                    1 / dt.asSeconds());
        ImGui::Text("Render: %dms",
                    t_state == RENDERING ? render_clock.getElapsedTime().asMilliseconds() : render_time);

        ImGui::PopStyleColor();
        ImGui::End();
    }

    template<class T>
    bool imgui_mat(T material) {
        auto updated = false;
        if (auto *dielectric = dynamic_cast<Dielectric *>(material)) {
            if (ImGui::SliderDouble("Refraction index", &dielectric->refraction_index, 0.001, 5, "%.3f",
                                    ImGuiSliderFlags_AlwaysClamp)) {
                updated = true;
            }
        } else if (auto *metal = dynamic_cast<Metal *>(material)) {
            if (ImGui::SliderDouble("Fuzz", &metal->fuzz, 0, 1)) {
                updated = true;
            }
            float mat[3];
            to_float_array(metal->albedo, mat);
            if (ImGui::ColorPicker3("###pls", mat)) {
                metal->albedo = from_float_array(mat);
                updated = true;
            }
        } else if (auto *lambertian = dynamic_cast<Lambertian * > (material)) {
            float mat[3];
            to_float_array(lambertian->albedo, mat);
            if (ImGui::ColorPicker3("###pls2", mat)) {
                lambertian->albedo = from_float_array(mat);
                updated = true;
            }
        }
        int curr = 0;
        if (ImGui::Combo("Ch", &curr, "Change material\0Lambertian\0Metal\0Dielectric\0Normals\0")) {
            switch (curr) {
                case 0:
                    break;
                case 1:
                    delete selected_hittable->material;
                    selected_hittable->material = new Lambertian(Colors::blue());
                    break;
                case 2:
                    delete selected_hittable->material;
                    selected_hittable->material = new Metal(Colors::blue(), 0);
                    break;
                case 3:
                    delete selected_hittable->material;
                    selected_hittable->material = new Dielectric(1.5);
                    break;
                case 4:
                    delete selected_hittable->material;
                    selected_hittable->material = new Normals();
                    break;
                default:
                    break;
            }
            updated = true;
        }

        return updated;
    }
};


int main() {
    GUI gui;
    gui.run();
    exit(0);
}

