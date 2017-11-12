import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import cv2
import mutual_information as minfo

# Program parameters
img_fn = os.path.join('Lenna.png')
samples = 100   # Number of angles (0, 2*pi)
bins = 50       # Number of bins in the histogram


# Define variables
angles = np.linspace(0, 360, samples)
mi_data = []
angle_data = []
i = 0

# Initialize plot
fig = plt.figure()
img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
# Plot original image
plt.subplot(221)
plt.title('Original')
plt.axis('off')
plt.imshow(img, cmap='gray')

# Plot image to be rotated
plt.subplot(222)
im = plt.imshow(img, animated=True, cmap='gray')
plt.title('Rotated')
plt.axis('off')

# Mutual information plot
plt.subplot(212)
mi_fig, = plt.plot(angle_data, mi_data, label='Mutual Information', lw=2)
plt.legend()
plt.xlim(0, 360)
plt.ylim(0, 4)
plt.xlabel('Rotation [Degrees]')
plt.ylabel('Mutual Information')


def update_fig(*args):
    global i
    if i < samples:
        cols, rows = np.shape(img)
        m_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angles[i], 1)
        im_rot = cv2.warpAffine(img, m_rot, (cols, rows))

        im.set_array(im_rot)
        # print 'Img. rot. shape: ', np.shape(im_rot)

        hist, x_edges, y_edges = np.histogram2d(img.ravel(), im_rot.ravel(), bins=bins)
        mi_data.append(minfo.mutual_information(hist))
        angle_data.append(angles[i])

        # print 'Mutual information: ', mi_data, 'Shape: ', np.shape(mi_data)

        mi_fig.set_data(angle_data, mi_data)

        i += 1
    return im, mi_fig,

ani = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)
plt.show()
