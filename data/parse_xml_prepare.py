
# coding: utf-8

import xml.etree.ElementTree as ET
import os

names_dict = {}
cnt = 0
f = open('/home/basic/YOLOv3_Tensorflow_Traffic_Cones/data/traffic_cones_real_1125/traffic_cone.txt', 'r').readlines()
for line in f:
    line = line.strip()
    names_dict[line] = cnt
    cnt += 1

#cones = '/media/basic/ssd256/traffic_cone_real'
cones = '/media/basic/ssd256/traffic_cone_syn'

#cones = '/media/basic/ssd256/cyclegan_data'

anno_path = [os.path.join(cones, 'annotations')]
img_path = [os.path.join(cones, 'images')]

trainval_path = [cones + '/A_train.txt']
test_path = [cones + '/A_test.txt']


def parse_xml(path):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]
    #print('img_name = ', img_name)

    #folder_path = tree.findtext("./path")

    #print('folder_path = ', folder_path)
    tmp = path.split('/')[-1]
    folder_path = path.replace(tmp, '').replace('annotations', 'images')
    #print('folder_path: ', folder_path)
    #print('folder_path: ', path.split('/')[-1][:-4])

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        # difficult = obj.find('difficult').text
        # if difficult == '1':
        #     continue
        difficult = obj.find('difficult').text
        #print('==> type: ', type(difficult))
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        name = 'traffic_cone'
        name = str(names_dict[name])
        objects.extend([name, xmin, ymin, xmax, ymax])

    # if difficult == '1':
    #     return None, None
    # elif difficult == '0':
    #     if len(objects) > 1:
    #         return objects, folder_path
    #     else:
    #         return None, None
    if len(objects) > 1:
        return objects, folder_path
    else:
        return None, None

test_cnt = 0
def gen_test_txt(txt_path):
    global test_cnt
    f = open(txt_path, 'w')

    for i, path in enumerate(test_path):
        img_names = open(path, 'r').readlines()
        for img_name in img_names:
            img_name = img_name.strip()
            #folder_path = img_name.split('/')[-2]
            img_name = img_name.split('/')[-1]
            img_name = img_name.split('.')[0]
            xml_path = anno_path[i] + '/' + img_name + '.xml'
            #folder_path = img_name.split('/')[-2]
            objects, folder_path = parse_xml(xml_path)
            if objects:
                #objects[0] = img_path[i] + '/' + folder_path + '/' + img_name + '.png' # for real traffice cones
                objects[0] = img_path[i] + '/' + img_name + '.png'  # for synthesis

                if os.path.exists(objects[0]):
                    objects.insert(0, str(test_cnt))
                    test_cnt += 1
                    objects = ' '.join(objects) + '\n'
                    f.write(objects)
    f.close()


train_cnt = 0
def gen_train_txt(txt_path):
    global train_cnt
    f = open(txt_path, 'w')

    for i, path in enumerate(trainval_path):
        img_names = open(path, 'r').readlines()
        #print('img len = ', len(img_names))
        for img_name in img_names:
            #print('img_name: ', img_name)
            img_name = img_name.strip()
            #folder_path = img_name.split('/')[-2]
            img_name = img_name.split('/')[-1]
            img_name = img_name.split('.')[0]
            xml_path = anno_path[i] + '/' + img_name + '.xml'
            #print('xml_path = ', xml_path)

            objects, folder_path = parse_xml(xml_path)
            if objects:
                #objects[0] = img_path[i] + '/' + folder_path + '/' + img_name + '.png'
                objects[0] = img_path[i] + '/' + img_name + '.png'  # for synthesis

                if os.path.exists(objects[0]):
                    objects.insert(0, str(train_cnt))
                    train_cnt += 1
                    print('img no = ', train_cnt)
                    objects = ' '.join(objects) + '\n'
                    f.write(objects)
    f.close()


gen_train_txt(cones + '/voc/train_full.txt')
gen_test_txt(cones + '/voc/test_full.txt')