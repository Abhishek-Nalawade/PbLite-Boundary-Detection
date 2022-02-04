from include.GenerateDoG import *

class LM(DoG):
    LMscales = list()
    scales3x = list()

    def generateDerivatives(self, gaussians):
        first_order = list()
        second_order = list()
        Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        # first derivative
        for gaussian in gaussians:
            # DoGx = self.convolve(gaussian, Sx, padding='same')
            # DoGy = self.convolve(gaussian, Sy, padding='same')
            DoGx = cv2.filter2D(gaussian, -1, Sx)
            DoGy = cv2.filter2D(gaussian, -1, Sy)
            DoG = (0.5*DoGx) + (0.5*DoGy)
            first_order.append(DoG)
        # second derivative
        for first_derivative in first_order:
            # DoGx = self.convolve(first_derivative, Sx, padding='same')
            # DoGy = self.convolve(first_derivative, Sy, padding='same')
            DoGx = cv2.filter2D(first_derivative, -1, Sx)
            DoGy = cv2.filter2D(first_derivative, -1, Sy)
            DoG = (0.5*DoGx) + (0.5*DoGy)
            second_order.append(DoG)
        return first_order, second_order

    def generateRotations(self, first_derivative, second_derivative, rotations):
        first_derivative_rot = list()
        second_derivative_rot = list()
        for i in range(len(first_derivative)):
            for j in range(len(rotations)):
                first_derivative_rot.append(self.rotateDoG(first_derivative[i], rotations[j]))
                second_derivative_rot.append(self.rotateDoG(second_derivative[i], rotations[j]))
        return first_derivative_rot, second_derivative_rot


    def generateLOG(self,gaussians, scales):
        LOG = list()
        for scale in scales:
            lower_limit = int(-((self.size-1)/2))
            upper_limit = abs(lower_limit) + 1
            ind = np.arange(lower_limit, upper_limit)
            row = np.reshape(ind, (ind.shape[0],1)) + np.zeros((1,ind.shape[0]))
            col = np.reshape(ind, (1,ind.shape[0])) + np.zeros((ind.shape[0],1))
            log = (-(1/(np.pi*(scale**4)))) * (1 - (((col)**2 + (row)**2)/(2*(scale**2)))) * np.exp(-(((col)**2 + (row)**2)/(2*(scale**2))))
            LOG.append(log)
        return LOG

    def generateLM(self, rotations):
        gaussians_for_DoG = list()
        [self.LMscales.append([i,3*i]) for i in self.scales]
        [self.scales3x.append(3*i) for i in self.scales]
        # generating gaussians for first three scales
        for i in range(len(self.LMscales)):
            gaussians_for_DoG.append(self.generateGaussian(self.LMscales[i][0],self.LMscales[i][1]))

        first_derivative, second_derivative = self.generateDerivatives(gaussians_for_DoG[:3])
        rotations_1D, rotations_2D = self.generateRotations(first_derivative, second_derivative, rotations)

        gaussians = list()
        for i in range(len(self.scales)):
            gaussians.append(self.generateGaussian(self.scales[i],self.scales[i]))

        LOG = self.generateLOG(gaussians, self.scales)
        LOG3x = self.generateLOG(gaussians, self.scales3x)

        LMfilters = list()
        for i in range(48):
            if i//12 == 0:
                if i<6:
                    LMfilters.append(rotations_1D[i])
                else:
                    LMfilters.append(rotations_2D[i-6])
            elif i//12 == 1:
                if i<18:
                    LMfilters.append(rotations_1D[i-6])
                else:
                    LMfilters.append(rotations_2D[i-12])
            elif i//12 == 2:
                if i<30:
                    LMfilters.append(rotations_1D[i-12])
                else:
                    LMfilters.append(rotations_2D[i-18])
            elif i//12 == 3:
                if i < 40:
                    LMfilters.append(LOG[i-36])
                elif 40 <= i < 44:
                    LMfilters.append(LOG3x[i-40])
                else:
                    LMfilters.append(gaussians[i-44])
        return LMfilters

    def plotfilters(self, LMfilters,  bank_type):
        fig, ax = plt.subplots(4,12, figsize=(15,15))
        for i in range(4):
            for j in range(12):
                ax[i][j].imshow(LMfilters[(i*12)+j], cmap='gray')
                # ax[i][j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                ax[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig("results/LM%s.png"%(bank_type))
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return
