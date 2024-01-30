add_executable(sfml_rt_exe #WIN32
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
        third-party/pcg_random/pcg_random.hpp
        third-party/pcg_random/pcg_extras.hpp
        third-party/pcg_random/pcg_uint128.hpp
        AABB.h
        AABB.h
        Scene.h
)
target_compile_definitions(sfml_rt_exe PRIVATE IS_SFML)

add_executable(sfml_rt_profiler_exe #WIN32
        main_profiler.cpp
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
        third-party/pcg_random/pcg_random.hpp
        third-party/pcg_random/pcg_extras.hpp
        third-party/pcg_random/pcg_uint128.hpp
        AABB.h
)
target_compile_definitions(sfml_rt_profiler_exe PRIVATE PROFILE)

# target_link_libraries(sfml_rt_profiler_exe
#         PRIVATE
#         ImGui-SFML::ImGui-SFML
# )

target_link_libraries(sfml_rt_exe
        PRIVATE
        ImGui-SFML::ImGui-SFML
)


if (WIN32)
    if (BUILD_SHARED_LIBS)
        add_custom_command(TARGET sfml_rt_exe POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ImGui-SFML::ImGui-SFML>
                $<TARGET_FILE:sfml-graphics>
                $<TARGET_FILE:sfml-window>
                $<TARGET_FILE:sfml-system>
                $<TARGET_FILE_DIR:sfml_rt_exe>)
        add_custom_command(TARGET sfml_rt_profiler_exe POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ImGui-SFML::ImGui-SFML>
                $<TARGET_FILE:sfml-graphics>
                $<TARGET_FILE:sfml-window>
                $<TARGET_FILE:sfml-system>
                $<TARGET_FILE_DIR:sfml_rt_profiler_exe>)
    endif ()
endif ()