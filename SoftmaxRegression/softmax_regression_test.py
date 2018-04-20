#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model_utils import * 
from SoftmaxRegression import *
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt



def test():
	"""
	Builds a model on a toy dataset (iris)
	"""
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	clf = SoftmaxRegression()
	print("Fitting Model now...")
	clf.fit(X,y.reshape(-1, 1))
	score = clf.score(X,y.reshape(-1, 1))
	print("On Iris Dataset, accuracy score is {}".format(score))
	# should easily beat this
	assert score >= .95
	print("Saving Learning Curve Plot")
	clf.plot_learning_curve()




if __name__ == '__main__':
	test()

