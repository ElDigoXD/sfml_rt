include(FetchContent)
set(IMGUI_VERSION 1.90.1)
set(IMGUI_SFML_VERSION 2.6)


FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui
        GIT_TAG d6cb3c923d28dcebb2d8d9605ccc7229ccef19eb #1.90.1
)

FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
        imgui-sfml
        GIT_REPOSITORY https://github.com/SFML/imgui-sfml
        GIT_TAG de565ac8f2b795dedc0307b60830cb006afd2ecd # 2.6
)
add_subdirectory(imgui-sfml)
