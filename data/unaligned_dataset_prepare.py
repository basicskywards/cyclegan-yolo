
import glob

def gen_paths_txt(folder_path, save_name):
	#tmp = ['jpg', 'png']
	list_files = glob.glob(folder_path + '**/*.png', recursive=True)
	file = open(folder_path + save_name, 'w')
	for file_path in list_files:
		print(file_path)
		file.write(file_path + '\n')
	print('list_files = ', len(list_files))
	file.close()

def output_txt(file_name, name_list):
    with open(file_name, "w") as f:
    	
    	for path in name_list:
    		#print(path)
    		f.write("%s" %(str(path)))
    	f.close()

def split_data(path_files, train = 0.9):
	#data_list = glob.glob(path_files + "/*.xml")
	data_list = open(path_files, 'r').readlines()
	total = len(data_list)
	#total = len(glob.glob(args.path_files + '/*.xml')) # count n_all_images '/*.png'
	#print(glob.glob(args.path_files + '/*.xml'))
	print("Total traffic cones: ", total)
	n_train = total * train
	train_set = []
	test_set = []
	counter = 0
	for i, path in enumerate(data_list):
		if counter <= n_train:
			train_set.append(path)
			counter += 1
			#print(path)
		else:
			test_set.append(path)
	return train_set, test_set


def main():
	save_name = 'A.txt'
	folder_path = '/media/basic/ssd256/traffic_cone_syn/'
	#folder_path = '/media/basic/ssd256/cyclegan_data/trainA/'
	gen_paths_txt(folder_path, save_name)
	

	path_files = folder_path + save_name	
	train_set, test_set = split_data(path_files, .1)

	print("Train set: ", len(train_set))
	print("Test set: ", len(test_set))

	save_train = folder_path + 'A_train.txt'
	save_test = folder_path + 'A_test.txt'
	output_txt(save_train, train_set)
	output_txt(save_test, test_set) 

if __name__ == '__main__':
	main()
	