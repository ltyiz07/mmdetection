import os
import glob
import random


get_filename = lambda x: os.path.basename(x).split(".")[0]


if __name__ == "__main__":
    image_path = r"./data/kitti/training/image_2/"
    label_path = r"./data/kitti/"

    image_suffix = ".png"

    dir_list = os.listdir(image_path)
    dir_list.sort()

    random.seed("first")
    random.shuffle(dir_list)

    train_val_ratio = 0.8   # ratio of train
    
    index = int(0.8 * len(dir_list))
    
    # train 
    with open(os.path.join(label_path, "train.txt"), "w") as file:
        for dir in dir_list[:index]:
            file.write(get_filename(dir) + "\n")

    # val
    with open(os.path.join(label_path, "val.txt"), "w") as file:
        for dir in dir_list[index:]:
            file.write(get_filename(dir) + "\n")
