#pragma once
#include "utils.h"
namespace Obj {

    int get_vertices(Vec3 **out_vertices, bool cuda) {
        auto vertices = std::vector<Vec3>();

        std::string input_file = "../src_cuda/resources/shuttle.obj";
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, input_file.c_str());
        std::cout << warn << std::endl;
        std::cerr << err << std::endl;

        if (!ret) exit(1);
        auto *mat = new Lambertian({0.2, 0.5, 0.8});
        // Shapes
        for (size_t s = 0; s < shapes.size(); s++) {
            size_t index_offset = 0;
            // Faces
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                // Vertices
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                    vertices.emplace_back(vx, vz, vy);
                }
                index_offset += fv;
            }
        }
        if (cuda){
#ifdef __CUDA_ARCH__
#ifdef CUDA
            CU(cudaMallocManaged(out_vertices, sizeof(Vec3) * vertices.size()));
#endif
#endif
        } else
            *out_vertices = new Vec3 [vertices.size()];

        std::copy(vertices.begin(), vertices.end(), *out_vertices);

        return static_cast<int>(vertices.size());
    }


    std::vector<Hittable *> *get_triangles() {
        auto *triangles = new std::vector<Hittable *>;

        std::string input_file = "../src_cuda/resources/shuttle.obj";
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, input_file.c_str());
        std::cout << warn << std::endl;
        std::cerr << err << std::endl;

        if (!ret) exit(1);
        auto *mat = new Lambertian({0.2, 0.5, 0.8});
        // Shapes
        for (size_t s = 0; s < shapes.size(); s++) {
            size_t index_offset = 0;
            // Faces
            Vec3 vertices[3];
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                // Vertices
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                    vertices[v] = {vx, vz, vy};
                }

                triangles->push_back(new Triangle(vertices, mat));
                index_offset += fv;
            }
        }
        return triangles;
    }
}