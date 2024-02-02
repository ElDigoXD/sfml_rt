add_executable(p_exe
        main_profiler.cu
        Vec3.h
        Ray.h
        Camera.h
        Hittable.h
        Sphere.h
        Interval.h
        utils.h
        Material.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        AABB.h
)

set_target_properties(p_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)

target_compile_definitions(p_exe PRIVATE PROFILE)

add_subdirectory(third-party)

add_executable(gui_exe
        main.cpp
        Vec3.h
        Ray.h
        Camera.h
        Hittable.h
        Sphere.h
        Interval.h
        utils.h
        Material.h
        third-party/BS_thread_pool.h
        AABB.h
        AABB.h
        Scene.h
)

target_compile_definitions(gui_exe PRIVATE IS_SFML)

set_target_properties(gui_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)

target_link_libraries(gui_exe
        PRIVATE
        ImGui-SFML::ImGui-SFML
)