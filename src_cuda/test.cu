

#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"

__global__ void test(Hittable **a) {
    //*a = new Sphere({0, 1.3, 0}, 1.3, new Normals());
    std::printf("gpu %f\n", ((Sphere *) (*a))->center.y);
}

__global__ void print(Hittable **a) {
    std::printf("gpu %f\n", ((Sphere *) (*a))->center.y);
}


int main() {
    Hittable **a;
    cudaMalloc(&a, sizeof(Hittable *));
    auto *b = new Sphere({0, 1.3, 0}, 1.3, new Normals());

    cudaMemcpy(a, b, sizeof(Sphere), cudaMemcpyHostToDevice);
    CU(cudaDeviceSynchronize());


    //*a = new Sphere({0, 1.3, 0}, 1.3, new Normals());
    test<<<1, 1>>>(a);
    print<<<1, 1>>>(a);
    CU(cudaDeviceSynchronize());
    //("cpu %f\n", ((Sphere *) (a))->center.y);
}