
#include "fftw3.h"

#define STB_IMAGE_IMPLEMENTATION

#include "third-party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"

#include <complex>
#include <iostream>

void angular_spectrum_kernel(fftw_complex *in, fftw_complex *out) {
    int num_pixels_x = 1920;
    int num_pixels_y = 1080;
    double pixel_size = 8e-6;
    double z = -0.2;
    double wavelength = 0.6328e-6;

    double k = 2 * M_PI / wavelength;

    int size = num_pixels_x * num_pixels_y;

    double slm_size_x = num_pixels_x * pixel_size;
    double slm_size_y = num_pixels_y * pixel_size;

    auto a = wavelength / slm_size_x;
    auto b = wavelength / slm_size_y;

    // fx = (wavelength/slm_size_x) * (np.arange(num_pixels_x) - num_pixels_x // 2)
    // fy = (wavelength/slm_size_y) * (np.arange(num_pixels_y) - num_pixels_y // 2)
    double fx[num_pixels_x];
    for (int i = 0; i < num_pixels_x; ++i) {
        fx[i] = a * (i - num_pixels_x / 2);
    }
    double fy[num_pixels_y];
    for (int i = 0; i < num_pixels_y; ++i) {
        fy[i] = b * (i - num_pixels_y / 2);
    }

    // fxx, fyy = np.meshgrid(fx, fy)
    auto *fxx = new double[size];
    auto *fyy = new double[size];

    for (int j = 0; j < num_pixels_y; ++j) {
        for (int i = 0; i < num_pixels_x; ++i) {
            fxx[i + j * num_pixels_x] = fx[i];
            fyy[i + j * num_pixels_y] = fy[j];
        }
    }

    // mod_fxfy= fxx*fxx+fyy*fyy
    auto *mod = new double[size];
    for (int i = 0; i < size; ++i) {
        mod[i] = fxx[i] * fxx[i] + fyy[i] * fyy[i];
    }



    // kernel = np.exp(1j*((k*z)*np.sqrt(1-mod)))
    fftw_complex *tmp = fftw_alloc_complex(size);
    fftw_complex *kernel_fft = fftw_alloc_complex(size);

    for (int i = 0; i < size; ++i) {
        auto c = std::exp(static_cast<std::complex<double>>(1.0j * (k * z) * std::sqrt(1 - mod[i])));
        kernel_fft[i][0] = c.real();
        kernel_fft[i][1] = c.imag();
    }

    // kernel = fftshift(kernel)
    for (int i = 0; i < size; ++i) {
        fftw_complex tmp_c;
        int src = i;
        int dst = (i + size/2 - 1) % size;

        tmp_c[0] = kernel_fft[src][0];
        tmp_c[1] = kernel_fft[src][1];

        kernel_fft[src][0] = kernel_fft[dst][0];
        kernel_fft[src][1] = kernel_fft[dst][1];

        kernel_fft[dst][0] = tmp_c[0];
        kernel_fft[dst][1] = tmp_c[1];
    }

    // Propagate_wave = np.fft.ifft2(fft2(input_image)*kernel)
    auto plan = fftw_plan_dft_2d(num_pixels_x, num_pixels_y, in, tmp, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int i = 0; i < size; ++i) {
        tmp[i][0] *= kernel_fft[i][0];
        tmp[i][1] *= kernel_fft[i][1];
    }

    plan = fftw_plan_dft_2d(num_pixels_x, num_pixels_y, tmp, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    fftw_free(kernel_fft);
    free(fxx);
    free(fyy);
    free(mod);
    free(tmp);
}

int main() {
    int w, h, comp;

    unsigned char *image = stbi_load("test.png", &w, &h, &comp, STBI_grey);
    printf("%d %d %d\n", w, h, comp);
    std::flush(std::cout);

    int size = w * h;

    fftw_complex *in = fftw_alloc_complex(size);
    fftw_complex *out = fftw_alloc_complex(size);

    for (int i = 0; i < size; ++i) {
        in[i][0] = static_cast<double>(image[i]) / 255;
        in[i][1] = 0;
    }

    angular_spectrum_kernel(in, out);

    for (int i = 0; i < size; ++i) {
        image[i] = static_cast<unsigned char>(out[i][0] / size * 255);
    }

    stbi_write_png("test_out1.png", w, h, 1, image, 0);

    stbi_write_png("test_out1.png", w, h, 1, image, 0);

    fftw_free(in);
    fftw_free(out);

    return 0;
}

int main2() {

    int w, h, comp;

    unsigned char *image = stbi_load("test.png", &w, &h, &comp, STBI_grey);

    printf("%d %d %d\n", w, h, comp);

    int size = w * h;

    // fft
    fftw_complex *in = fftw_alloc_complex(size);
    fftw_complex *out = fftw_alloc_complex(size);
    fftw_complex *inverse = fftw_alloc_complex(size);

    fftw_plan plan = fftw_plan_dft_2d(w, h, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < size; ++i) {
        in[i][0] = static_cast<double>(image[i]) / 255;
        in[i][1] = 0;
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int i = 0; i < size; ++i) {
        out[i][0] /= size;
        out[i][1] /= size;
    }

    plan = fftw_plan_dft_2d(w, h, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < size; ++i) {
        image[i] = static_cast<unsigned char>(inverse[i][0] * 255);
    }

    stbi_write_png("test_out1.png", w, h, 1, image, 0);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 0;
}

