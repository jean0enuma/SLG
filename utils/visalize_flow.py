import numpy as np
import matplotlib.pyplot as plt

# Define make_color_wheel function
def make_color_wheel():
    """
    Generates a color wheel for visualizing optical flow.
    Returns:
        numpy array: A color wheel (n x 3) where n is the number of distinct colors.
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # R, G, B

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

# Function to compute optical flow visualization using color wheel
def compute_color(u, v):
    """
    Compute optical flow color map
    Args:
        u (np.array): Horizontal optical flow
        v (np.array): Vertical optical flow
    Returns:
        np.array: Optical flow visualization in RGB color space
    """
    height, width = u.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255.0
        col1 = tmp[k1 - 1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img

# Create a sample vector field (u, v)
height, width = 100, 100
x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
u = -y
v = x

# Compute optical flow visualization
flow_image = compute_color(u, v)

# Display the result
plt.figure(figsize=(6, 6))
plt.imshow(flow_image)
plt.axis('off')
plt.title("Optical Flow Visualization using Color Wheel")
plt.show()