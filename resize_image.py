from os import listdir
import cv2
import numpy as np

folder1 = 'train_all/'
folder2 = 'train/'
folder3 = 'test/'
count_cat = 1
count_dog = 1
width = 250
height = 250
dim = (width, height)
train_size = 250
test_size = 500

for file in listdir(folder1):
	if count_cat <= train_size:
		if file.startswith('cat'):
			img = cv2.imread(folder1 + file, cv2.IMREAD_UNCHANGED)
			# resize image
			resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			fname = folder2 + 'cat.' + str(count_cat) + '.jpg'
			cv2.imwrite(fname, img=(resized).clip(0.0, 255.0).astype(np.uint8))
			count_cat += 1
	elif count_cat <= train_size + test_size:
		if file.startswith('cat'):
			img = cv2.imread(folder1 + file, cv2.IMREAD_UNCHANGED)
			# resize image
			resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			fname = folder3 + 'cat.' + str(count_cat) + '.jpg'
			cv2.imwrite(fname, img=(resized).clip(0.0, 255.0).astype(np.uint8))
			count_cat += 1
	else: 
		break

for file in listdir(folder1):
	if count_dog <= train_size:
		if file.startswith('dog'):
			img = cv2.imread(folder1 + file, cv2.IMREAD_UNCHANGED)
			# resize image
			resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			fname = folder2 + 'dog.' + str(count_dog) + '.jpg'
			cv2.imwrite(fname, img=(resized).clip(0.0, 255.0).astype(np.uint8))
			count_dog += 1
	elif count_dog <= train_size + test_size:
		if file.startswith('dog'):
			img = cv2.imread(folder1 + file, cv2.IMREAD_UNCHANGED)
			# resize image
			resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			fname = folder3 + 'dog.' + str(count_dog) + '.jpg'
			cv2.imwrite(fname, img=(resized).clip(0.0, 255.0).astype(np.uint8))
			count_dog += 1
	else:
		break
