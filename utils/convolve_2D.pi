#
# JLC jean-luc.charles@ensam.eu 2021/01
#
# adapted from github.com/SamratSahoo/UT-Arlington-Research/blob/master/Week 6 - Convolutions & Wavelet Transforms/Convolutions.py
#

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def plot_images(images, r, L=1, C=1, coeff=1):
    plt.figure(figsize=(coeff*C,coeff*L))
    for i in range(L*C):
        plt.subplot(L, C, i+1)
        plt.imshow(images[r+i], cmap='gray')
        plt.xticks([]); plt.yticks([]);
    plt.show()

if __name__ == "__main__":

    import tensorflow as tf
    from tensorflow import keras
    import sys, cv2
    import matplotlib.pyplot as plt
    import numpy as np

    (im_train, lab_train), (im_test, lab_test) = tf.keras.datasets.mnist.load_data()

    # Edge Detection Kernel
    kernels = [np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]]),
               np.array([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]),
               np.array([[ 0, 0, 0], [ 1, 1, 1], [-1,-1,-1]]),
               np.array([[ 0, 1,-1], [ 0, 1,-1], [ 0, 1,-1]])]

    # Convolve and Save Output
    out = []
    for kernel in kernels:
        out.append(convolve2D(im_train[608], kernel))
    plot_images(out,0,1,len(out))    

