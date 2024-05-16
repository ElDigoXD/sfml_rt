
#include "fftw3.h"

#define STB_IMAGE_IMPLEMENTATION

#include "third-party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third-party/stb_image_write.h"

#include <complex>
#include <iostream>
#include "vector"
#include "algorithm"

// https://stackoverflow.com/questions/5915125/fftshift-ifftshift-c-c-source-code
static inline
void fftshift2D(std::complex<double> *data, size_t xdim, size_t ydim) {
    size_t xshift = xdim / 2;
    size_t yshift = ydim / 2;
    if ((xdim * ydim) % 2 != 0) {
        // temp output array
        std::vector<std::complex<double> > out;
        out.resize(xdim * ydim);
        for (size_t x = 0; x < xdim; x++) {
            size_t outX = (x + xshift) % xdim;
            for (size_t y = 0; y < ydim; y++) {
                size_t outY = (y + yshift) % ydim;
                // row-major order
                out[outX + xdim * outY] = data[x + xdim * y];
            }
        }
        // copy out back to data
        copy(out.begin(), out.end(), &data[0]);
    } else {
        // in and output array are the same,
        // values are exchanged using swap
        for (size_t x = 0; x < xdim; x++) {
            size_t outX = (x + xshift) % xdim;
            for (size_t y = 0; y < yshift; y++) {
                size_t outY = (y + yshift) % ydim;
                // row-major order
                swap(data[outX + xdim * outY], data[x + xdim * y]);
            }
        }
    }
}

// https://stackoverflow.com/questions/5915125/fftshift-ifftshift-c-c-source-code
template<class ty>
void circshift(const ty *in, ty *out, int xdim, int ydim, int xshift, int yshift) {
    for (int i = 0; i < xdim; i++) {
        int ii = (i + xshift) % xdim;
        for (int j = 0; j < ydim; j++) {
            int jj = (j + yshift) % ydim;
            out[ii * ydim + jj] = in[i * ydim + j];
        }
    }
}

void fftshift(std::complex<double> *data, int xdim, int ydim) {
    auto *tmp = new std::complex<double>[xdim * ydim];
    circshift(data, tmp, xdim, ydim, xdim / 2, ydim / 2);
    copy(tmp, tmp + xdim * ydim, data);
    free(tmp);
}


void angular_spectrum_kernel(fftw_complex *in, fftw_complex *out) {
    int num_pixels_x = 1920;
    int num_pixels_y = 1080;
    double pixel_size = 8e-6;
    double z = -0.1;
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
        fx[i] = a * (i - std::floor(num_pixels_x / 2.0));
    }
    double fy[num_pixels_y];
    for (int i = 0; i < num_pixels_y; ++i) {
        fy[i] = b * (i - std::floor(num_pixels_y / 2.0));
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
    auto *kernel = new std::complex<double>[size];

    for (int i = 0; i < size; ++i) {
        auto c = std::exp(std::complex<double>(0, (k * z) * std::sqrt(1 - mod[i])));
        kernel[i] = c;
    }

    // kernel = fftshift(kernel)
    printf("kernel 0 0: %f %f\n", kernel[0].real(), kernel[0].imag());
    fftshift(kernel, num_pixels_x, num_pixels_y);
    printf("kernel 0 0: %f %f\n", kernel[0].real(), kernel[0].imag());

    // Propagate_wave = np.fft.ifft2(fft2(input_image)*kernel)
    auto plan = fftw_plan_dft_2d(num_pixels_x, num_pixels_y, in, tmp, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int i = 0; i < size; ++i) {
        tmp[i][0] *= kernel[i].real() / size;
        tmp[i][1] *= kernel[i].imag() / size;
    }

    plan = fftw_plan_dft_2d(num_pixels_x, num_pixels_y, tmp, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    fftw_free(tmp);
    free(fxx);
    free(kernel);
    free(fyy);
    free(mod);
}

int main() {
    int w, h, comp;

    unsigned char *image = stbi_load("ph_1cpu_166s.png", &w, &h, &comp, STBI_grey);
    printf("%d %d %d\n", w, h, comp);
    std::flush(std::cout);

    int size = w * h;

    fftw_complex *in = fftw_alloc_complex(size);
    fftw_complex *out = fftw_alloc_complex(size);


    // [0, 255] -> [-1, 1] -> [-pi, pi]
    for (int i = 0; i < size; ++i) {
        auto c = std::exp(std::complex<double>(0, M_PI * (image[i] - 127.5) / 127.5));
        in[i][0] = c.real();
        in[i][1] = c.imag();
    }
    printf("in 0 0: %f %f\n", in[0][0], in[0][1]);

    angular_spectrum_kernel(in, out);

    printf("out 0 0: %f %f\n", out[0][0], out[0][1]);
    printf("pp 0 0: %f\n", std::abs(std::complex{out[0][0], out[0][1]}));
    for (int i = 0; i < size; ++i) {
        std::complex c = {out[i][0], out[i][1]};
        image[i] = static_cast<unsigned char>(std::abs(c) * 255);
    }

    stbi_write_png("test_out1.png", w, h, 1, image, 0);

    //stbi_write_png("test_out1.png", w, h, 1, image, 0);

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

