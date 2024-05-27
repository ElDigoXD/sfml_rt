# Dependencies
add_subdirectory(third-party)

# rt_gpu_exe
add_executable(rt_gpu_exe
        main_rt_gpu.cu
        AABB.h
        Camera.h
        hittable/Hittable.h
        hittable/HittableList.h
        Interval.h
        material/Material.h
        Ray.h
        hittable/Sphere.h
        Vec3.h
        utils.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        hittable/Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        hittable/BVHNode.h
        HoloCamera.h
        third-party/stb_image.h
)

set_target_properties(rt_gpu_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)
target_link_libraries(rt_gpu_exe -lcurand)
target_compile_definitions(rt_gpu_exe PRIVATE PROFILE)

# test_exe
add_executable(test_exe
        test.cpp
        third-party/stb_image_write.h
        third-party/stb_image.h
)

set_target_properties(test_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)
target_compile_definitions(test_exe PRIVATE PROFILE)

# cgh_cpu_exe
add_executable(cgh_cpu_exe
        main_cgh_cpu.cu
        AABB.h
        Camera.h
        hittable/Hittable.h
        hittable/HittableList.h
        Interval.h
        material/Material.h
        Ray.h
        hittable/Sphere.h
        Vec3.h
        utils.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        hittable/Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        hittable/BVHNode.h
        HoloCamera.h
        third-party/stb_image.h
        third-party/stb_image.h
)

set_target_properties(cgh_cpu_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)
target_compile_definitions(cgh_cpu_exe PRIVATE PROFILE)

# cgh_cpu_exe
add_executable(cgh_gpu_exe
        main_cgh_gpu.cu
        AABB.h
        Camera.h
        hittable/Hittable.h
        hittable/HittableList.h
        Interval.h
        material/Material.h
        Ray.h
        hittable/Sphere.h
        Vec3.h
        utils.h
        third-party/BS_thread_pool.h
        third-party/stb_image_write.h
        hittable/Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        hittable/BVHNode.h
        HoloCamera.h
        third-party/stb_image.h
)

set_target_properties(cgh_gpu_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)
target_link_libraries(rt_gpu_exe -lcurand)
target_compile_definitions(cgh_gpu_exe PRIVATE PROFILE)


# rt_gui_cpu_exe
add_executable(rt_gui_cpu_exe
        main_rt_gui_cpu.cu
        Vec3.h
        Ray.h
        Camera.h
        hittable/Hittable.h
        hittable/Sphere.h
        Interval.h
        utils.h
        material/Material.h
        third-party/BS_thread_pool.h
        AABB.h
        AABB.h
        Scene.h
        hittable/HittableList.h
        hittable/Triangle.h
        third-party/tiny_obj_loader.h
        obj.h
        hittable/BVHNode.h
        HoloCamera.h
        third-party/stb_image.h
)

target_compile_definitions(rt_gui_cpu_exe PRIVATE IS_SFML)

set_target_properties(rt_gui_cpu_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/)

target_link_libraries(rt_gui_cpu_exe PRIVATE ImGui-SFML::ImGui-SFML)
