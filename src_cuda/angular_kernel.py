#Angular spectrum kernel function
#it is used to propagate a plane wave from 0 to z distance

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import fftshift



def angular_spectrum_kernel(input_image,dx, dy, wavelength, num_pixels_x, num_pixels_y, slm_size_x, slm_size_y, z):
    k = 2 * np.pi / wavelength

    # fx = dx * (wl/Lx) * (np.arange(Nx) - Nx // 2)
    # fy = dy * (wl/Ly) * (np.arange(Ny) - Ny // 2)
    fx = (wavelength/slm_size_x) * (np.arange(num_pixels_x) - num_pixels_x // 2)
    fy = (wavelength/slm_size_y) * (np.arange(num_pixels_y) - num_pixels_y // 2)

    fxx, fyy = np.meshgrid(fx, fy)

    mod_fxfy= fxx*fxx+fyy*fyy

    kernel = np.exp(1j*((k*z)*np.sqrt(1-mod_fxfy)))
    kernel_scale = np.abs(np.sum(kernel))
    print('angular spectrum kernel_scale = ', kernel_scale)
    kernel = fftshift(kernel)

    Propagate_wave = np.fft.ifft2(fft2(input_image)*kernel)
    print('angular_spectrum_int = ', np.sum(np.absolute(Propagate_wave*np.conj(Propagate_wave))))
    # np.save('CGHs/results/angular_spectrum_out',Propagate_wave)
    return (Propagate_wave)