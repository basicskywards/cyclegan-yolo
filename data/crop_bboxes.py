# bbox loss
# rescale_bbox
# crop_by_bbox
# resize


from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np 
import os
import torch

# def rescale_boxes(boxes, current_dim, original_shape):
#     """ Rescales bounding boxes to the original shape """
#     orig_h, orig_w = original_shape
#     # The amount of padding that was added
#     pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
#     pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
#     # Image height and width after padding is removed
#     unpad_h = current_dim - pad_y
#     unpad_w = current_dim - pad_x
#     # Rescale bounding boxes to dimension of original image
#     boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
#     boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
#     boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
#     boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
#     return boxes


# def xywh2xyxy(x):
#     y = x.new(x.shape)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] + x[..., 2] / 2
#     y[..., 3] = x[..., 1] + x[..., 3] / 2
#     return y

# def rescale_bbox(bbox, original_img_size, current_img_size=416):
# 	orig_h, orig_w = original_img_size
# 	# The amount of padding added
# 	pad_x = max(orig_h - orig_w, 0) * (current_img_size / max(original_img_size))
# 	pad_y = max(orig_w - orig_h, 0) * (current_img_size / max(original_img_size))
# 	# Image height & width after padding removed
# 	unpad_h = current_img_size - pad_y
# 	unpad_w = current_img_size - pad_x
# 	# Rescale bbox to dimension of original image
# 	bbox[:, 0] = ((bbox[:, 0] - pad_x // 2) / unpad_w) * orig_w
# 	bbox[:, 1] = ((bbox[:, 1] - pad_y // 2) / unpad_h) * orig_h
# 	bbox[:, 2] = ((bbox[:, 2] - pad_x // 2) / unpad_w) * orig_w
# 	bbox[:, 3] = ((bbox[:, 3] - pad_y // 2) / unpad_h) * orig_h	
# 	return bbox

#---------------------------------
# def pad_to_square_x(img, pad_value):
# 	c, h, w = img.shape
# 	dim_diff = np.abs(h - w)
# 	# (upper / left) padding and (lower / right) padding
# 	pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
# 	# Determine padding
# 	pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
# 	# Add padding
# 	img = F.pad(img, pad, "constant", value=pad_value)

# 	return img, pad



#---------------------------------
def get_img_label_paths(img_path_txt):
	with open(img_path_txt, "r") as file:
		img_files = file.readlines()

	label_files = []
	for path in img_files:
		img_id = path.split('/')[-1].split('.')[0]
		# synthetic
		label_path_tmp = '/home/basic/PyTorch-YOLOv3/data/traffic_cones_syn_yololoss/labels/'
		
		# real
		#label_path_tmp = '/home/basic/PyTorch-YOLOv3/data/traffic_cones/labels/'

		label_path = label_path_tmp + img_id + '.txt'
		label_files.append(label_path)
	return img_files, label_files

def get_label(label_path):
	if os.path.exists(label_path):
		boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
	return boxes

def get_img(img_path):
	img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
	return img 

def bbox_rescale(boxes, img_size=416):
	''' To rescale bboxes in accordance with the image'''

	# resized 416x416
	h_factor, w_factor = img_size, img_size

	# synthetic
	#h_factor, w_factor = 1080, 1920

	# real
	#h_factor, w_factor = 1536, 2048

	# Extract coordinates for unpadded + unscaled image
	# bbox for original images (can use this bbox to extract objects from orig images)
	#print('\n boxes info: ', boxes)
	x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
	y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
	x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
	y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
	boxes_scaled = np.zeros(boxes.shape)
	boxes_scaled[:, 1] = x1 + 1
	boxes_scaled[:, 2] = y1 + 1
	boxes_scaled[:, 3] = x2 + 1
	boxes_scaled[:, 4] = y2 + 1
	return boxes_scaled

def resize_img(image, size=416):
	#print('\nimage info: ', image.shape, type(image))
	image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
	#image = F.interpolate(image, size=(size, size))#, mode="nearest")

	return image

def save_traffic_cone(image_tensor, image_path, idx):
	# image_pil = Image.fromarray(image_numpy)
	# image_pil.save(image_path)
	try:
		transforms.ToPILImage()(image_tensor).save(image_path + 'test%d.png' %(idx), mode='png')
	except:
		print('ValueError: tile cannot extend outside image')

def crop_object_by_bbox(real_img, boxes):
	# real_img: 416x416
	# process 1 image with its boxes
	list_real_cones = []
	#print('\n boxes info in crop_by_bbox: ', boxes)
	for i in range(len(boxes)):
		#print('\nboxes info: ', boxes)
		#x_min, y_min, x_max, y_max = int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]), int(boxes[i, 4])
		x_min, y_min, x_max, y_max = boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]

		#print('\nx1, y1, x2, y2: ', x_min, y_min, x_max, y_max)
		cone = real_img[..., int(y_min):int(y_max), int(x_min):int(x_max)]
		list_real_cones.append(cone)
		#print('\n cone info: ', cone.shape, type(cone))
	
	return list_real_cones

#-----------------------------------------------------------------------------#
# define VGG to extract features from cropped traffic cones
#-----------------------------------------------------------------------------#







#-----------------------------------------------------------------------------#
def main():
	# synthetic images
	img_path_txt = '/media/basic/ssd256/cyclegan_data/A_train.txt'
	
	# real images
	#img_path_txt = '/media/basic/ssd256/cyclegan_data/B_train.txt'

	# save path
	cropped_cone_path = '/home/basic/cyclegan/data/tmp_cones/'



	img_files, label_files = get_img_label_paths(img_path_txt)
	idx = 20 

	img_txt = img_files[idx].rstrip()
	label_txt = label_files[idx].rstrip()

	#print('img_txt, label_txt: ', img_txt, label_txt)
	img = get_img(img_txt)
	img_resized = resize_img(img, 416)
	#print('img: ', img.shape)
	#print('img_resized: ', img_resized.shape)
	boxes = get_label(label_txt)
	boxes_rescaled = bbox_rescale(boxes, 416)
	list_cones = crop_object_by_bbox(img_resized, boxes_rescaled) 
	for i, cone in enumerate(list_cones):
		save_traffic_cone(cone, cropped_cone_path, i)

if __name__ == '__main__':
	main()