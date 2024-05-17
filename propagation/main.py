import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft.helper import fftshift
from numpy.fft import fft2, ifft2


mm = 1e-3
um = 1e-6
nm = 1e-9

slm_z = -1000 * mm
pixelsize = 8 * um
wl = 632.8 * nm
k = 2 * np.pi/wl

# z_srt = input('Propagation distance, z (mm): ')

# focal?

lx = 15.36 * mm
ly = 8.64 * mm

dx = pixelsize
dy = pixelsize

# Import CGH

image = Image.open('../out/ph_16cpu_387s.png')
data_im = np.array(image)

print("Imported image: ", data_im.shape, " pixels.")

# [0, 255] --> [-1, 1]
data_norm = (data_im - 127.5) / 127.5

# [-1, 1] --> [-pi, pi]
data_phase = np.exp(1j * np.pi * data_norm)
print("in 0 0:", data_phase[0, 0].real, data_phase[0, 0].imag)

nx = int(data_phase.shape[1])
ny = int(data_phase.shape[0])

lx = nx*dx
ly = ny*dy

# Propagation kernel


def propagation_kernel(slm_z):
    fx = (wl/lx) * (np.arange(nx) - nx // 2)
    fy = (wl/ly) * (np.arange(ny) - ny // 2)

    fxx, fyy = np.meshgrid(fx, fy)

    mod_fxfy = fxx * fxx + fyy * fyy
    kernel = np.exp(1j*((k*slm_z)*np.sqrt(1-mod_fxfy)))
    print("kernel 0 0:", kernel[0, 0])
    kernel = fftshift(kernel)
    print("kernel 0 0:", kernel[0, 0])

    propagated = ifft2(fft2(data_phase)*kernel)

    return propagated


p = propagation_kernel(-100*mm)
print("out 0 0: ", p[0, 0].real, p[0, 0].imag)
pp = np.abs(p)
print("pp: ", pp[0, 0])
# exit(0)
# Show an image with pyplot

zs = [200, 400, 600, 800, 1000, 1200]
zs = range(600, 1001, 50)
zs = range(760, 850, 10)
zs = range(200, 218, 2)

max_columns = 3
num_rows = int(np.ceil(len(zs) / max_columns))
num_cols = min(len(zs), max_columns)

print(f"Plotting {len(zs)} images in a grid of {
      num_rows} rows and {num_cols} columns.")

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(5 * num_cols, 3*num_rows))
if num_rows == 1:
    axes = np.reshape(axes, (1, -1))
elif num_cols == 1:
    axes = np.reshape(axes, (-1, 1))

for i, ax in enumerate(axes.flat):
    if i < len(zs):
        ax.imshow(np.abs(propagation_kernel(-zs[i]*mm)), cmap='gray')
        ax.axis('off')
        ax.set_title('z = {} mm'.format(zs[i]))
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig("out.png")