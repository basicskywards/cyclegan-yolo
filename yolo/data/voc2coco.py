# Prepare COCO annotation for training YOLOv3
# To convert VOC annotation format to COCO

from __future__ import print_function, division
import os
#import pandas as pd 
import numpy as np

def read_txt(txt_path):
	f = open(txt_path, "r")
	for line in f:
		yield line 


def parse_line(line):
	'''
	Inputs: a line from train/test txt file
		line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
	
	Returns:
		image_path: string
		boxes: [N, 4], N = number of boxes, 4 = x_min, y_min, x_max, y_max
		labels: class index
		img_width: int
		img_height: int		
	'''	
	if 'str' not in str(type(line)):
		line = line.decode()
	str_line = line.strip().split(' ')
	assert len(str_line) > 8, 'Annotation error 1!'
	image_path = str_line[1]
	img_width = int(str_line[2])
	img_height = int(str_line[3])
	bboxes = str_line[4:]
	assert len(bboxes) % 5 == 0, 'Annotation error 2!'
	box_cnt = len(bboxes) // 5
	boxes = []
	labels = []
	for i in range(box_cnt):
		label = int(bboxes[i * 5])
		x_min = float(bboxes[i * 5 + 1])
		y_min = float(bboxes[i * 5 + 2])
		x_max = float(bboxes[i * 5 + 3])
		y_max = float(bboxes[i * 5 + 4])
		boxes.append([x_min, y_min, x_max, y_max])
		labels.append(label)
	boxes = np.asarray(boxes, np.float32)
	labels = np.asarray(labels, np.int64)
	return image_path, boxes, labels, img_width, img_height

def convert_voc2coco(bbox, img_width, img_height):
	'''
	Here is COCO label format:
	label <1> <2> <3> <4>

	Here is how to convert COCO to VOC label format
    <1>*w = (xmax-xmin)/2 + xmin
    <2>*h = (ymax-ymin)/2 + ymin
    <3> = (xmax-xmin)/w
    <4> = (ymax-ymin)/h

	Returns: x_min, y_min, w, h scaled into [0, 1]

	'''
	x = (((bbox[2] - bbox[0])/2 + bbox[0]) - 1) / img_width
	y = (((bbox[3] - bbox[1]) / 2 + bbox[1]) - 1) / img_height
	w = (bbox[2] - bbox[0]) / img_width
	h = (bbox[3] - bbox[1]) / img_height

	return x, y, w, h 

# def gen_coco_txt(save_path):
# 	f = open(save_path, 'w')

# 	pass

def main():
	txt_path = '/home/basic/YOLOv3_Tensorflow_Traffic_Cones_ITRI/data/traffic_cones_real/test_full_without_difficult.txt'
	label_path = '/home/basic/PyTorch-YOLOv3/data/traffic_cones/labels/'
	train_test_path = '/home/basic/PyTorch-YOLOv3/data/traffic_cones/'
	name_txt = 'test.txt'
	f1 = open(train_test_path + name_txt, 'w')
	for line in read_txt(txt_path):
		img_path, boxes, labels, img_w, img_h = parse_line(line)
		img_name = img_path.split('/')[-1]
		#tmp = img_name + '\n'
		f1.write(img_path + '\n')
		img_txt = img_name.split('.')[0] + '.txt'
		f2 = open(label_path + img_txt, 'w')
		#object_line = []
		for i in range(len(labels)):
			bbox = boxes[i, :]
			#print('bbox = ', bbox[0])
			label = labels[i]
			#object_line.append(str(label))
			x, y, w, h = convert_voc2coco(bbox, img_w, img_h)
			#object_line.append([str(label), str(x), str(y), str(w), str(h)])
			tmp_line = [str(label), str(x), str(y), str(w), str(h)]
			#print('object_line = ', object_line[0])
			tmp_line = ' '.join(v for v in tmp_line) + '\n'
			#print('tmp_line = ', tmp_line)
			f2.write(tmp_line)
	f1.close()

if __name__ == '__main__':
	main()
