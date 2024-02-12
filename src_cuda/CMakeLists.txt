add_executable(p_exe
        AABB.h
        Camera.h
        Hittable.h
        HittableList.h
        Interval.h
        Material.h
        Ray.h
        Sphere.h
        Vec3.h
        main_profiler.cu
        utils.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        BVHNode.h
)

set_target_properties(p_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)
target_link_libraries(p_exe -lcurand)

target_compile_definitions(p_exe PRIVATE PROFILE)

add_executable(c_exe
        AABB.h
        Camera.h
        Hittable.h
        HittableList.h
        Interval.h
        Material.h
        Ray.h
        Sphere.h
        Vec3.h
        utils.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        main_profiler_cpu.cu
        Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        BVHNode.h
)

set_target_properties(c_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)

target_link_libraries(c_exe -lcurand)
target_compile_definitions(c_exe PRIVATE PROFILE)


add_subdirectory(third-party)

add_executable(gui_exe
        main.cu
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
        HittableList.h
        Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        BVHNode.h
)

target_compile_definitions(gui_exe PRIVATE IS_SFML)

set_target_properties(gui_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)

target_link_libraries(gui_exe
        PRIVATE
        ImGui-SFML::ImGui-SFML
)
