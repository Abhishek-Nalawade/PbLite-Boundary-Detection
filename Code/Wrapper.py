#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s):
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu)
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:
from include.GenerateDoG import *
from include.GenerateLM import *
from include.GenerateGabor import *
from include.GenerateHD import *
from include.kmeans import *
import sklearn.cluster

def chi_sqr_dist_func(disks, map, num_of_IDs):
	all_dist = list()
	for j in range(int(len(disks)/2)):
		chi_sqr_dist = np.zeros((map.shape[0],map.shape[1]))
		for i in range(num_of_IDs):
			tmp = np.zeros((map.shape[0],map.shape[1]))
			ind = np.where(map==i)
			tmp[ind] = 1
			g_i = cv2.filter2D(tmp, -1, disks[j])
			h_i = cv2.filter2D(tmp, -1, disks[j+1])
			dn = g_i + h_i
			zer = np.where(dn==0)
			dn[zer] = 0.0001
			chi_sqr_dist = chi_sqr_dist + (((g_i - h_i)**2)/dn)
		chi_sqr_dist = (1/2) * chi_sqr_dist
		all_dist.append(chi_sqr_dist)
	all_dist = np.array(all_dist)
	meanTg = np.sum(all_dist, axis=0)/(len(disks)/2)
	# print("shape of meanMapg: ",meanTg.shape)
	# plt.imshow(meanTg)
	# plt.show()
	return meanTg

def main():
	for k in range(1,11):
		image_num = k

		"""
		Generate Difference of Gaussian Filter Bank: (DoG)
		Display all the filters in this filter bank and save image as DoG.png,
		use command "cv2.imwrite(...)"
		"""
		kernel_size = 55
		scales = [4,8]		# along X and Y
		div = 360/16
		rotations = [div*i for i in range(1,17)]
		dog = DoG(kernel_size, scales)
		dogFilters = dog.generateDoG(rotations)

		dog.plotDoG(dogFilters)
		# print(len(dogFilters))

		"""
		Generate Leung-Malik Filter Bank: (LM)
		Display all the filters in this filter bank and save image as LM.png,
		use command "cv2.imwrite(...)"
		"""
		# LMS
		scales_small = [1, 2**(1/2), 2, 2*(2**(1/2))]
		rotations = [0,30,60,90,120,150]
		lm = LM(55,scales_small)
		LMSfilters = lm.generateLM(rotations)

		lm.plotfilters(LMSfilters, bank_type='S')

		# LML
		large_small = [2**(1/2), 2, 2*(2**(1/2)), 4]
		rotations = [0,30,60,90,120,150]
		lm = LM(55,large_small)
		LMLfilters = lm.generateLM(rotations)

		lm.plotfilters(LMLfilters, bank_type='L')
		"""
		Generate Gabor Filter Bank: (Gabor)
		Display all the filters in this filter bank and save image as Gabor.png,
		use command "cv2.imwrite(...)"
		"""
		scales = [2,4,5,6,10]
		rotations = [22.5*i for i in range(8)]
		gabor = Gabor(21, scales)
		gaborFilters = gabor.generateGabor(rotations)

		gabor.plotGabor(gaborFilters)

		"""
		Generate Half-disk masks
		Display all the Half-disk masks and save image as HDMasks.png,
		use command "cv2.imwrite(...)"
		"""
		radius = [5,7,16]
		size = [(radius[i]*2) for i in range(len(radius))]
		angles = [22.5*i for i in range(8)]
		rotations = [(i*np.pi)/180 for i in angles]
		hd = HalfDisks(size, radius, rotations)
		disks = hd.generateHDs()

		hd.plotHDS(disks)


		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		allFilters = list()
		allFilters = dogFilters + LMSfilters + LMLfilters + gaborFilters# + disks
		total_len = len(allFilters)

		# total_len = 40
		# allFilters = gaborFilters

		img = cv2.imread("../BSDS500/Images/%s.jpg"%(image_num))
		# img = cv2.resize(img, (450,450))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# cv2.imshow("wind", img)
		# cv2.waitKey(0)
		# i = 0
		convolved = list()
		print("\nConvolving using the filter banks........\n")
		for filter in allFilters:
			# print("convolving using filter %d from a total of %d filters"%(i,total_len))
			# edge = dog.convolve(img, filter)
			edge = cv2.filter2D(gray, -1, filter)
			convolved.append(edge)
			cv2.imshow("wind", edge)
			cv2.waitKey(1)
			# i += 1
		cv2.destroyAllWindows()
		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		# cluster = Kmeans(64)
		# cluster.fit(convolved)
		print("Generating Texton Map.......\n")
		num_of_textonIDs = 64
		convolved = np.array(convolved)
		convolved = np.reshape(convolved, (convolved.shape[0],(convolved.shape[1]*convolved.shape[2])))
		convolved = np.moveaxis(convolved, 0, -1)
		kmeans = sklearn.cluster.KMeans(n_clusters = num_of_textonIDs)
		kmeans.fit(convolved)
		labels = kmeans.labels_
		texton = np.reshape(labels, (img.shape[0],img.shape[1]))
		plt.imsave("results/%s/TextonMap_%s.png"%(image_num, image_num), texton)
		plt.imshow(texton)
		plt.show(block=False)
		plt.pause(3)
		plt.close()


		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		print("Generating Tg.......\n")
		Tg = chi_sqr_dist_func(disks, texton, num_of_textonIDs)
		plt.imsave("results/%s/Tg_%s.png"%(image_num, image_num), Tg)
		plt.imshow(Tg)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		"""
		Generate Brightness Map
		Perform brightness binning
		"""
		print("Generating Brightness Map.......\n")
		num_of_IDs = 16
		tmp_img = np.reshape(gray, ((img.shape[0]*img.shape[1]), 1))
		kmeans = sklearn.cluster.KMeans(n_clusters = num_of_IDs)
		kmeans.fit(tmp_img)
		labels = kmeans.labels_
		# print("from herere: ",labels.shape)
		brightness = np.reshape(labels, (img.shape[0],img.shape[1]))
		plt.imsave("results/%s/BrightnessMap_%s.png"%(image_num, image_num), brightness)
		plt.imshow(brightness)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		print("Generating Bg.......\n")
		Bg = chi_sqr_dist_func(disks, brightness, num_of_IDs)
		plt.imsave("results/%s/Bg_%s.png"%(image_num, image_num), Bg)
		plt.imshow(Bg)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		print("Generating Color Map.......\n")
		num_of_IDs = 16
		tmp_img = np.reshape(img, ((img.shape[0]*img.shape[1]), 3))
		kmeans = sklearn.cluster.KMeans(n_clusters = num_of_IDs)
		kmeans.fit(tmp_img)
		labels = kmeans.labels_
		color = np.reshape(labels, (img.shape[0],img.shape[1]))
		plt.imsave("results/%s/ColorMap_%s.png"%(image_num, image_num), color)
		plt.imshow(color)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		print("Generating Cg Map.......\n")
		Cg = chi_sqr_dist_func(disks, color, num_of_IDs)
		plt.imsave("results/%s/Cg_%s.png"%(image_num, image_num), Cg)
		plt.imshow(Cg)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobelPb = cv2.imread("../BSDS500/SobelBaseline/%s.png"%(image_num))
		# sobelPb = cv2.resize(sobelPb, (img.shape[0], img.shape[1]))
		sobelPb = cv2.cvtColor(sobelPb, cv2.COLOR_BGR2GRAY)

		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		cannyPb = cv2.imread("../BSDS500/CannyBaseline/%s.png"%(image_num))
		# cannyPb = cv2.resize(cannyPb, (img.shape[0], img.shape[1]))
		cannyPb = cv2.cvtColor(cannyPb, cv2.COLOR_BGR2GRAY)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		PbEdges = ((Tg + Bg + Cg)/3) * ((0.4*cannyPb) + (0.6*sobelPb))
		# cv2.imshow("final",PbEdges)
		# print(PbEdges)
		plt.imsave("results/%s/PbLite_%s.png"%(image_num, image_num), PbEdges, cmap='gray')
		plt.imshow(PbEdges, cmap='gray')
		plt.show(block=False)
		plt.pause(3)
		plt.close()

if __name__ == '__main__':
	main()
