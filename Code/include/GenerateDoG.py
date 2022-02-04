import numpy as np
import cv2
import matplotlib.pyplot as plt

class DoG:
    def __init__(self, size, scales):
        self.size = size
        self.scales = scales


    def generateGaussian(self, scaleX, scaleY):
        lower_limit = int(-((self.size-1)/2))
        upper_limit = abs(lower_limit) + 1
        ind = np.arange(lower_limit, upper_limit)
        row = np.reshape(ind, (ind.shape[0],1)) + np.zeros((1,ind.shape[0]))
        col = np.reshape(ind, (1,ind.shape[0])) + np.zeros((ind.shape[0],1))
        # G = (1/(2*np.pi*(scaleX*scaleY))) * np.exp(-(((col)**2 + (row)**2)/(2*(scaleX*scaleY))))
        G = (1/(2*np.pi*(scaleX*scaleY))) * np.exp(-(((col)**2/(2*(scaleX**2))) + ((row)**2/(2*(scaleY**2)))))
        return G

    def convolve(self, G, sobel, padding='same'):
        pad_size = int((sobel.shape[0]-1)/2)
        # inverting for convolution
        sobel = sobel[::-1,:]
        sobel = sobel[:,::-1]
        if padding == 'zeros':
            paddedG = np.zeros(((2*pad_size)+G.shape[0],(2*pad_size)+G.shape[1]))
            paddedG[pad_size:pad_size+G.shape[0], pad_size:pad_size+G.shape[1]] = G
            DoG = np.zeros((G.shape[0], G.shape[1]))
        elif padding == False:
            paddedG = G
            DoG = np.zeros((G.shape[0]-(2*pad_size), G.shape[1]-(2*pad_size)))
        elif padding == 'same':
            paddedG = np.zeros(((2*pad_size)+G.shape[0],(2*pad_size)+G.shape[1]))
            paddedG[pad_size:pad_size+G.shape[0], pad_size:pad_size+G.shape[1]] = G
            DoG = np.zeros((G.shape[0], G.shape[1]))
            for i in range(pad_size):
                # top left
                paddedG[i,i] = G[0,0]
                paddedG[i, pad_size:pad_size+G.shape[1]] = G[0,:]
                # bottom right
                paddedG[-i-1,-i-1] = G[-1,-1]
                paddedG[paddedG.shape[0]-1-i, pad_size:pad_size+G.shape[1]] = G[-1,:]
                # bottom left
                paddedG[paddedG.shape[0]-1-i, i] = G[-1,-1]
                paddedG[pad_size:pad_size+G.shape[0], i] = G[:,0]
                # right
                paddedG[i,paddedG.shape[1]-1-i] = G[-1,-1]
                paddedG[pad_size:pad_size+G.shape[0], paddedG.shape[1]-1-i] = G[:,-1]



        # convolving
        for i in range(DoG.shape[0]):
            for j in range(DoG.shape[1]):
                upperi = i + sobel.shape[0]
                upperj = j + sobel.shape[1]
                # print(i," : ",upperi,"  ",j," : ",upperj)
                DoG[i,j] = np.sum(paddedG[i:upperi, j:upperj] * sobel)
        return DoG

    # rotatedDoG = (np.cos(angle) * DoGx) + (np.sin(angle) * DoGy)
    def rotateDoG(self, DoG, angle):
        h, w = DoG.shape
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotatedDoG = cv2.warpAffine(DoG, M, (w, h))
        return rotatedDoG

    def generateDoG(self, rotations):
        dog = list()
        for scale in self.scales:
            gaussian = self.generateGaussian(scale, scale)
            # defining sobel operator
            Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            # DoGx = self.convolve(gaussian,Sx, padding='same')
            # DoGy = self.convolve(gaussian,Sy, padding='same')
            DoGx = cv2.filter2D(gaussian, -1, Sx)
            DoGy = cv2.filter2D(gaussian, -1, Sy)
            DoG = (0.5*DoGx) + (0.5*DoGy)

            for angle in rotations:
                r = self.rotateDoG(DoG, angle)
                dog.append(r)
                # plt.imshow(r, cmap='gray')
                # plt.pause(0.2)

        return dog

    def plotDoG(self, filters):
        fig, ax = plt.subplots(2,16, figsize=(15,15))
        for i in range(2):
            for j in range(16):
                ax[i][j].imshow(filters[(i*16)+j], cmap='gray')
                ax[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig("results/DoG.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return
