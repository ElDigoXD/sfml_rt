#include "imgui.h"
#include "imgui-SFML.h"

#define BS_THREAD_POOL_ENABLE_PAUSE

#include "third-party/BS_thread_pool.h"
#include <SFML/Graphics.hpp>
#include <thread>
#include <type_traits>

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
    std::jthread t;

    int render_time = 0;
    int render_update_ms;

    sf::Clock update_texture_clock;
    sf::Clock delta_clock;
    sf::Clock render_clock;

    std::shared_ptr<Lambertian> material_ground;
    std::shared_ptr<Dielectric> material_center;
    std::shared_ptr<Dielectric> material_left;
    std::shared_ptr<Metal> material_right;
    std::shared_ptr<Normals> material_normals;
    std::shared_ptr<Sphere> look_at;

    Color *colors_agg = new Color[max_window_width * max_window_height];
    Color *colors = new Color[max_window_width * max_window_height];

    int pass_number{1};
    bool continuous_render;

    int continuous_render_sample_limit;

    sf::Vector2i before_click_mouse_position;
    sf::Vector2i before_move_mouse_position;
    bool mouse_pressed = false;
    sf::Time dt;

    GUI() {
        // Imgui variables
        render_update_ms = 1;
        t_n = 4;
        pool.reset(t_n);
        camera.samples_per_pixel = 10;
        camera.max_depth = 100;
        camera.vfov = 20;
        camera.look_from = {13, 2, 3};
        camera.look_at = camera.look_from - unit_vector(camera.look_from) * 10;
        camera.defocus_angle = 0.6;
        camera.update();
        continuous_render = true;
        continuous_render_sample_limit = 100;

        material_ground = std::make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
        material_center = std::make_shared<Dielectric>(1.5);
        material_left = std::make_shared<Dielectric>(1.5);
        material_right = std::make_shared<Metal>(Color(0.8, 0.6, 0.2), 0);
        material_normals = std::make_shared<Normals>();

        world.add(std::make_shared<Sphere>(Point3(0.0, -100.5, -1.0), 100.0, material_ground));
        world.add(std::make_shared<Sphere>(Point3(0.0, 0.0, -1.0), 0.5, material_center));
        world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), 0.5, material_left));
        world.add(std::make_shared<Sphere>(Point3(-1.0, 0.0, -1.0), -0.4, material_left));
        world.add(std::make_shared<Sphere>(Point3(1.0, 0.0, -1.0), 0.5, material_right));
        look_at = std::make_shared<Sphere>(camera.look_at, 0.2, material_normals);
        // world.add(look_at);
        world = HittableList();
        auto ground_material = std::make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
        world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = Random::_double();
                Point3 center(a + 0.9 * Random::_double(), 0.2, b + 0.9 * Random::_double());

                if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                    std::shared_ptr<Material> sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = Color::random() * Color::random();
                        sphere_material = std::make_shared<Lambertian>(albedo);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = Color::random(0.5, 1);
                        auto fuzz = Random::_double(0, 0.5);
                        sphere_material = std::make_shared<Metal>(albedo, fuzz);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    } else {
                        // glass
                        sphere_material = std::make_shared<Dielectric>(1.5);
                        world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    }
                }
            }
        }

        auto material1 = std::make_shared<Dielectric>(1.5);
        world.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

        auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
        world.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

        auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
        world.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));
    }

    void run() {
        // Gui stuff
        window.setFramerateLimit(60);
        auto _ = ImGui::SFML::Init(window);
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        texture.create(max_window_width, max_window_height);
        sprite.setTexture(texture, true);

        start_render();

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
                        && sprite.getGlobalBounds().contains((float) event.mouseButton.x, (float) event.mouseButton.y)
                        && sf::Mouse::getPosition(window).x < image_width) {
                        mouse_pressed = true;
                        window.setMouseCursorVisible(false);
                        before_click_mouse_position = sf::Mouse::getPosition(window);
                        before_move_mouse_position = before_click_mouse_position;
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
                           &&
                           sprite.getGlobalBounds().contains((float) event.mouseButton.x, (float) event.mouseButton.y)
                           && sf::Mouse::getPosition(window).x < image_width) {
                    camera.vfov -= event.mouseWheel.delta;

                    update_camera_and_start_render();
                }
            }

            handle_mouse();

            handle_keyboard();


            if (t_state != IDLE) {
                if (continuous_render) {
                    if (pool.get_tasks_total() == 0 && t_state != DONE) {
                        colors_to_pixel();
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

    void colors_to_pixel() {
        for (int i = 0; i < image_width * image_height; ++i) {
            colors_agg[i] = (colors_agg[i] * (pass_number - 1) + colors[i]) / pass_number;
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
                        camera.camera_center - camera.look_at, camera.u, (delta.y) / 500.0);
                new_look_at = rodrigues_rotation(
                        new_look_at, Vec3(0, 1, 0), (delta.x) / 500.0);

                camera.look_at = (camera.camera_center - new_look_at);

                update_camera_and_start_render();
            }
        }
    }

    void handle_keyboard() {
        auto speed = 0.003 * dt.asMilliseconds();
        if (!window.hasFocus()) return;
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

    void imgui() {
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

        ImGui::BeginDisabled(t_state == RENDERING);
        if (ImGui::Checkbox("Continuous render", &continuous_render)) {
            if (continuous_render) {
                continuous_render_sample_limit = camera.samples_per_pixel;
            } else {
                camera.samples_per_pixel = continuous_render_sample_limit;
            }
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(continuous_render);
        ImGui::Text("Render update:");
        ImGui::SliderInt("##a", &render_update_ms, 1, 1000, "%dms");
        ImGui::EndDisabled();

        ImGui::Text("Render threads:");
        if (ImGui::SliderInt("##b", (int *) &t_n, 1, (int) std::thread::hardware_concurrency())) {
            pool.reset(t_n);
        }

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

        ImGui::Text("Fov:");
        if (ImGui::SliderDouble("##e", &camera.vfov, 1, 200)) {
            camera.update();
            start_render();
        }

        ImGui::Separator();

        bool re_render_on_material_change = true;
        ImGui::Checkbox("Render on change", &re_render_on_material_change);
        if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {

            for (auto mat: std::map<char *, Material *>{{"Left",   material_left.get()},
                                                        {"Center", material_center.get()},
                                                        {"Right",  material_right.get()}}) {
                if (ImGui::TreeNode(mat.first)) {
                    ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
                    if (imgui_mat(mat.second) & re_render_on_material_change) {
                        start_render();
                    }
                    ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
                    ImGui::TreePop();
                }
            }
        }
        ImGui::Separator();
        ImGui::Text("%dx%d %d samples, %4.0ffps", image_width, image_height, pass_number * camera.samples_per_pixel,
                    1 / dt.asSeconds());
        ImGui::Text("Render: %dms",
                    t_state == RENDERING ? render_clock.getElapsedTime().asMilliseconds() : render_time);

        ImGui::PopStyleColor();
        ImGui::End();
    }

    template<class T>
    bool imgui_mat(T &material) {
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

        return updated;
    }

};


int main() {
    GUI gui;
    gui.run();
    exit(0);
}

