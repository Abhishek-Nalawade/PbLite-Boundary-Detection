import numpy as np
import cv2
import matplotlib.pyplot as plt

class HalfDisks:
    def __init__(self, size, radius, rotations):
        self.radius = radius
        self.rotations = rotations
        self.size = size

    def generateMask(self, size, theta):
        centerX = size//2
        centerY = size//2
        ind = np.arange(0,size)
        row = np.reshape(ind, (ind.shape[0],1)) + np.zeros((1,ind.shape[0]))
        col = np.reshape(ind, (1,ind.shape[0])) + np.zeros((ind.shape[0],1))

        cover = ((size-1)-row-centerY)>=np.tan(theta)*((size-1)-col-centerX)
        cover2 = ((size-1)-row-centerY)<=np.tan(theta)*((size-1)-col-centerX)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        return cover, cover2


    def generateHDs(self):
        disks = list()
        for i in range(len(self.radius)):
            for j in range(len(self.rotations)):
                disk = np.zeros((self.size[i], self.size[i]))
                disk2 = np.zeros((self.size[i], self.size[i]))
                centerX = (self.size[i]-1)/2
                centerY = (self.size[i]-1)/2
                ind = np.arange(0,self.size[i])
                row = np.reshape(ind, (ind.shape[0],1)) + np.zeros((1,ind.shape[0]))
                col = np.reshape(ind, (1,ind.shape[0])) + np.zeros((ind.shape[0],1))
                row_sq = (row-centerY)**2
                col_sq = (col-centerX)**2
                dist = (row_sq + col_sq)**(1/2)
                circle = dist<=self.radius[i]
                disk[circle] = 1
                disk2[circle] = 1
                cover, cover2 = self.generateMask(self.size[i], self.rotations[j])
                disk[cover] = 0
                disk2[cover2] = 0
                disks.append(disk)
                disks.append(disk2)
        return disks

    def plotHDS(self, disks):
        fig, ax = plt.subplots(6,8, figsize=(15,15))
        for i in range(6):
            for j in range(8):
                ax[i][j].imshow(disks[(8*i)+j], cmap='gray')
                ax[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig("results/HDMasks.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return
