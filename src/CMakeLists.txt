add_executable(sfml_rt_exe
        main.cpp
        Vec3.h
        Ray.h
        Camera.h
)
target_compile_definitions(sfml_rt_exe PRIVATE IS_SFML)

add_executable(sfml_rt_profiler_exe
        main_profiler.cpp
        Vec3.h
        Ray.h
        Camera.h
)
target_compile_definitions(sfml_rt_profiler_exe PRIVATE PROFILE)
target_compile_options(sfml_rt_profiler_exe PRIVATE -O3)

target_link_libraries(sfml_rt_exe
    PRIVATE
        ImGui-SFML::ImGui-SFML
    )

if(WIN32)
    if (BUILD_SHARED_LIBS)
        add_custom_command(TARGET sfml_rt_exe POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ImGui-SFML::ImGui-SFML>
                $<TARGET_FILE:sfml-graphics>
                $<TARGET_FILE:sfml-window>
                $<TARGET_FILE:sfml-system>
                $<TARGET_FILE_DIR:sfml_rt_exe>)
    endif()
endif()