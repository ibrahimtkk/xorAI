# coding=utf-8
import numpy as np
import pickle

def dec_to_stringBinary(number):
	return "{0:b}".format(number)

def addZero(x, maxi):
	maxLeng = len(dec_to_stringBinary(maxi))
	dist = maxLeng - len(dec_to_stringBinary(x))
	x = '0'*dist + dec_to_stringBinary(x)
	return x

def find_training_output(liste):
	training_output = [[]]
	for arr in liste:
		arr = list(arr)
		top = 0
		for i in arr:
			top += i
		if top%2 == 0:
			training_output[0].append(0)
		else:
			training_output[0].append(1)

	training_output = np.array(training_output)
	return training_output

def num_to_ndarray(num):
	arr = []
	for i in range(num+1):
		stringBin = addZero(i, num)
		stringBin = list(stringBin)
		stringBin = map(int, stringBin)
		arr.append(list(stringBin))
	arr = np.array(arr)
	return arr

class NeuralNetwork():
	leng = -1
	weightList, adjList, layerList, deltaList = [[]], [[]], [[]], [[]]
	training_input, training_output = [], []
	repeatCount, iterationCount = 2000, 216

	def __init__(self, leng):
		self.leng = leng
		self.weightList = [[] for x in range(self.leng+1)]
		self.adjList    = [[] for x in range(self.leng+1)]
		self.layerList  = [[] for x in range(self.leng+1)]
		self.deltaList  = [[] for x in range(self.leng+1)]
		self.defWeightList()

	def defWeightList(self):
		for j in range(self.leng):
			self.weightList[j] = 2 * np.random.random((self.leng, self.leng)) - 1
		self.weightList[self.leng-1] = 2 * np.random.random((self.leng, 1)) - 1
	

	

	def think(self):
		for iteration in range(self.iterationCount):
			m1 = np.dot(self.training_input, self.weightList[0])						
			layer1 = 1 / (1 + np.exp(-m1))									
			self.layerList[1] = layer1					

			for j in range(1, self.leng):
				mJarti1 = np.dot(self.layerList[j], self.weightList[j]) 			
				self.layerList[j+1] = 1 / (1 + np.exp(-mJarti1))					


			errorLast = self.training_output - self.layerList[self.leng]
			deltaLast = errorLast * self.layerList[self.leng] * (1 - self.layerList[self.leng])
			self.deltaList[self.leng] = deltaLast										

			for j in range(self.leng-1, 0, -1):
				errorJ = np.dot(self.deltaList[j+1], self.weightList[j].T)
				self.deltaList[j] = errorJ * self.layerList[j] * (1 - self.layerList[j])


			self.adjList[0] = np.dot(self.training_input.T, self.deltaList[1])
			for j in range(1, self.leng):
				self.adjList[j] = np.dot(self.layerList[j].T, self.deltaList[j+1])

			for j in range(self.leng):
				self.weightList[j] += self.adjList[j]

		return self.weightList

	def say(self, test_input):
		for j in range(self.repeatCount):
			self.weightList = self.think()
			print "weightList: ", self.weightList

			m0 = np.dot(test_input, self.weightList[0])
			output0 = 1 / (1 + np.exp(-m0))


			outputi = output0
			for i in range(1, self.leng):
				mi = np.dot(outputi, self.weightList[i])
				outputi = 1 / (1 + np.exp(-mi))


			if (j%50==0):
				print(j, outputi)

		print(' test_input -> binary: ', test_input)


if __name__ == '__main__':

	# Girdileri düzenlemek için yapılan ayarlamalar
	# ========================================================================
	learnNumber = int(input("learnNumber: "))
	testNumber = int(input("testNumber: "))

	training_input = num_to_ndarray(learnNumber)
	training_output = find_training_output(training_input).T

	test_input_add_zero = addZero(testNumber, learnNumber)
	test_input_add_zero = list(test_input_add_zero)
	test_input = map(int, test_input_add_zero)
	print testNumber,"-> ",test_input
	test_input = np.array(test_input)
	# ========================================================================

	leng = len(training_input[0])
	neural_network = NeuralNetwork(leng)
	neural_network.training_input = training_input
	neural_network.training_output = training_output

	neural_network.say(test_input)