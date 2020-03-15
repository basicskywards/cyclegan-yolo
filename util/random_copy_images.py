
import shutil
import glob

path = "/home/basic/Downloads/datasets/synthetic_traffic_cones/Real"
imgs_list = glob.glob(path + "/*.png")
out_path = "/home/basic/Downloads/datasets/cyclegan_data_big/trainA"

copy_ratio = .5
count = 0
for i in range(int(len(imgs_list) * copy_ratio)):
    print(imgs_list[i])
    shutil.copy(imgs_list[i], out_path)
    count += 1

print("Total images copied: {}".format(count))
