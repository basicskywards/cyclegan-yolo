import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np 
from models import network
## Given a txt file of paths of real traffic cone images, extract 1 vector feature representing all traffic cones
## Save that vector feature into disk

'''
1. img_id - img_path - img_annotation
2. img original size + original bboxes = crop
3. img resized 410x410 + resize bboxes (?) = crop
'''
# use original real traffic cone images - size h=1536, w=2048
def coco2voc(x, y, w, h, img_w=2048, img_h=1536):
    ## converts the normalized positions  into integer positions
    x_max = int((x*img_w) + (w * img_w)/2.0)
    x_min = int((x*img_w) - (w * img_w)/2.0)
    y_max = int((y*img_h) + (h * img_h)/2.0)
    y_min = int((y*img_h) - (h * img_h)/2.0)
    return x_min, y_min, x_max, y_max

# def scale_bbox(bbox, img_size): # if necessary
#     pass
#     return x_min, y_min, x_max, y_max

def crop_real_img_bbox(real_img, bboxes):
    # process 1 image
    list_real_cones = []
    for box in bboxes:
        x_min, x_max, y_min, y_max = box
        cone = real_img[..., y_min:y_min+y_max, x_min:x_min+x_max]
        list_real_cones.append(cone)
    
    return list_real_cones

def resize_for_vgg(img, h_resnet=224, w_resnet=224):
    # find size vgg
    img = F.interpolate(img, size=(h_resnet, w_resnet), mode="nearest")
    return img

def get_real_cones(real_img, bboxes, h_resnet=224, w_resnet=224):
    # process 1 path at a time
    list_real_cones = crop_real_img_bbox(real_img, bboxes)
    list_real_cones_resnet = []
    for cone in list_real_cones:
        img_resnet = resize_for_vgg(cone, h_resnet, w_resnet)
        list_real_cones_resnet.append(img_resnet)
    return list_real_cones_resnet

def get_img_bboxes(real_img_path):
    # parse image & bboxes (x_min, x_max, y_min, y_max)
    # TODO - get image & bboxes from a single path
    return real_img, bboxes


# def resnet_extract(cone):
#     # TODO - get ResNet34
#     resnet = networks.define_feature_network('resnet34', 0)
#     cone_feature = resnet(cone)
#     return cone_feature

# def compute_mean_cones_feature(list_real_cones):
#     mean_cones_features_vector = []
#     for cone in list_real_cones:
#         feat = resnet_extract(cone)
#         mean_cones_features_vector.append(feat)
#     mean_cones_features_vector.stack(dim=0).mean(dim=0)
#     return mean_cones_features_vector

# def extract_mean_real_cone_features(real_img_file_of_paths):
#     # TODO - get list of all paths


#     list_of_all_real_cones = []
#     for path in list_paths:
#         real_img, bboxes = get_img_bboxes(path)
#         list_cones_per_img = get_real_cones(real_img, bboxes)
#         list_of_all_real_cones.extend(list_cones_per_img)
#     mean_cones_features_vector = compute_mean_cones_feature(list_of_all_real_cones)

#     return mean_cones_features_vector

# def main():
#     real_img_file_of_paths = ''
#     save_path = ''
#     mean_cones_features_vector = extract_mean_real_cone_features(real_img_file_of_paths)

#     # save vector
#     np.save(save_path, mean_cones_features_vector)

#     # load vector
#     #np.load(save_path)
#     return print('Saved to ', save_path)

if __name__ == '__main__':
    main()
