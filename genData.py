#Import required packages
import numpy as np
import cv2
import random
import os
import sys
import shutil
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import configparser
import secrets

try:
	config = configparser.ConfigParser()
	config.read('config.ini')

	gen_per_image = int(config['AUGMENTATION']['GEN_PER_IMAGE'])
	gen_per_class = int(config['AUGMENTATION']['GEN_PER_CLASS'])
	#path to the folder containing the data ready for augmentation
	path = config['DEFAULT']['DATASET_PATH']
	rotation_range = float(config['AUGMENTATION']['ROTATION_RANGE'])
	width_shift_range = float(config['AUGMENTATION']['WIDTH_SHIFT_RANGE'])
	height_shift_range = float(config['AUGMENTATION']['HEIGHT_SHIFT_RANGE'])
	shear_range = float(config['AUGMENTATION']['SHEAR_RANGE'])
	zoom_range = float(config['AUGMENTATION']['ZOOM_RANGE'])
	horizontal_flip = (config['AUGMENTATION']['HORIZONTAL_FLIP'] == "True")
	fill_mode = config['AUGMENTATION']['FILL_MODE']
except:
	print("Please check your config file for issues")
	exit()

def increase_brightness(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value

	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

def change_contrast(img, level):
	img = Image.fromarray(img.astype('uint8'))
	factor = (259 * (level + 255)) / (255 * (259 - level))
	def contrast(c):
		return 128 + factor * (c - 128)
	return np.array(img.point(contrast))

def pad_img(img):
	h, w = img.shape[:2]
	new_h = int((5 + secrets.randbelow(16)) * h / 100) + h
	new_w = int((5 + secrets.randbelow(16)) * w / 100) + w

	full_sheet = np.ones((new_h, new_w, 3)) * 255

	p_X = secrets.randbelow(new_h - img.shape[0])
	p_Y = secrets.randbelow(new_w - img.shape[1])

	full_sheet[p_X : p_X + img.shape[0], p_Y : p_Y + img.shape[1]] = img

	full_sheet = cv2.resize(full_sheet, (w, h), interpolation = cv2.INTER_AREA)

	return full_sheet.astype(np.uint8)

def preprocess_img(img):
	img = np.array(img)

	x = secrets.randbelow(2)

	if x == 0:
		# img = pad_img(img)
		img = increase_brightness(img, secrets.randbelow(26))
		img = change_contrast(img, secrets.randbelow(51))
	else:
		# img = pad_img(img)
		img = change_contrast(img, secrets.randbelow(51))
		img = increase_brightness(img, secrets.randbelow(26))

	return img

def copy_org(doc_type):
	files = os.listdir(os.path.join(path, doc_type))

	for file in files:
		shutil.copy(os.path.join(path, doc_type, file), os.path.join(os.getcwd(),"augmented_data",doc_type, file))


#Initialise the parameters for Augmentation.
datagen = ImageDataGenerator(
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range = shear_range,
        zoom_range = zoom_range,
        horizontal_flip = horizontal_flip,
        fill_mode = fill_mode,
        preprocessing_function = preprocess_img)

def generator(doc_type, total):
	# print(doc_type + " " + set_type)
	print(doc_type)
	# src_path = os.path.join(path, doc_type, set_type)
	src_path = os.path.join(path,doc_type)
	dst_path = os.path.join(os.getcwd(), "augmented_data",doc_type)
	# files = os.listdir(src_path)
	files = os.listdir(src_path)
	m = len(files)

	for i in range(total):
		k = secrets.randbelow(m)
		img_cv = cv2.resize(cv2.imread(os.path.join(src_path, files[k])), (500, 500), interpolation = cv2.INTER_AREA)
		cv2.imwrite("temp_img.jpg", img_cv)
		img = load_img("temp_img.jpg")  # this is a PIL image
		# img = load_img(os.path.join(src_path, files[k]))  # this is a PIL image
		imgarr = img_to_array(img)  # this is a Numpy array with shape (?, ?, ?)

		gen_file_name = doc_type + "_" + str(i)

		# cv2.imwrite(os.path.join(dst_path, gen_file_name + ".jpg"), cv2.imread(os.path.join(src_path, files[k])))

		imgarr = imgarr.reshape((1,) + imgarr.shape)  # this is a Numpy array with shape (1, ?, ?, ?)

		n = 1
		for batch in datagen.flow(imgarr, batch_size=1, save_to_dir=dst_path, save_prefix=gen_file_name, save_format='jpeg'):
		    n += 1
		    if n > gen_per_image:
		        break  # otherwise the generator would loop indefinitely




#Contains all the labels
doc_types = os.listdir(path)

for doc_type in doc_types:
	if not os.path.exists(os.path.join(os.getcwd(), "augmented_data",doc_type)):
		os.makedirs(os.path.join(os.getcwd(),"augmented_data",doc_type))
	# generator(doc_type, set_types[0], gen_per_class)
	generator(doc_type,gen_per_class)
	copy_org(doc_type)
