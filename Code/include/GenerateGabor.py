import numpy as np
import cv2
import matplotlib.pyplot as plt

class Gabor:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma


    def generateGabor(self, rotation, Lambda=6, psi=1, gamma=1):
        rotations = [(i*np.pi)/180 for i in rotation]
        filters = list()
        for i in range(len(self.sigma)):
            for j in range(len(rotations)):
                sigma_x = self.sigma[i]
                sigma_y = self.sigma[i]/gamma

                lower_limit = int(-((self.size-1)/2))
                upper_limit = abs(lower_limit) + 1
                (y, x) = np.meshgrid(np.arange(lower_limit, upper_limit), np.arange(lower_limit, upper_limit))

                # Rotation
                theta = rotations[j]
                x_theta = (x * np.cos(theta)) + (y * np.sin(theta))
                y_theta = (-(x * np.sin(theta))) + (y * np.cos(theta))
                # gb = np.exp(-((x_theta**2/(2*sigma_x**2))+(y_theta**2/(2*sigma_y**2)))) * np.cos(((2*np.pi/Lambda)*x_theta) + psi)
                gb = np.exp(-((x_theta**2/(2*sigma_x**2))+(y_theta**2/(2*sigma_y**2)))) * np.cos(((2*np.pi/Lambda)*x_theta) + psi)
                filters.append(gb)
                # plt.imshow(gb, cmap='gray')
                # plt.pause(0.2)
        # plt.show()

        return filters

    def plotGabor(self, filters):
        fig, ax = plt.subplots(5,8, figsize=(15,15))
        for i in range(5):
            for j in range(8):
                ax[i][j].imshow(filters[(i*8)+j], cmap='gray')
                ax[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig("results/Gabor.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return
