import cv2
import matplotlib.pyplot as plt
import os
def get_coord(img_label):
    img_name = img_label.split("txt")[0] + "jpg"
    label_name = img_label
    img_path = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train/" + img_name
    label_file = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train/" + label_name

    img = cv2.imread(img_path)
    img_height, img_width = img.shape[0], img.shape[1]
    lfile = open(label_file)
    coords = []
    all_coords = []

    for line in lfile:
        l = line.split(" ")
        label = l[0]
        coords = list(map(float, list(map(float, l[1:5]))))
        x1 = float(img_width) * (2.0 * float(coords[0]) - float(coords[2])) / 2.0
        y1 = float(img_height) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
        x2 = float(img_width) * (2.0 * float(coords[0]) + float(coords[2])) / 2.0
        y2 = float(img_height) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0
        tmp = [ x1, y1, x2, y2, label]
        all_coords.append(list(map(int, tmp)))
    lfile.close()
    return all_coords

def save_crop(coord_list, img_name):
    img_path = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train/" + img_name
    img = cv2.imread(img_path)
    for coord in coord_list:
        y1 = coord[1]
        y2 = coord[3]
        x1 = coord[0]
        x2 = coord[2]
        label = coord[4]
        img_crop = img[y1:y2, x1:x2]
        # plt.imshow(img_crop)
        # plt.show()
        if label == 0:
            save_crop_img(label)
            img_crop = img[y1:y2, x1:x2]
            cv2.imwrite("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_0/img{}.jpg".format(img_name), img_crop)
        elif label == 1:
            save_crop_img(label)
            img_crop = img[y1:y2, x1:x2]
            cv2.imwrite("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_1/img{}.jpg".format(img_name), img_crop)
        elif label == 2:
            save_crop_img(label)
            img_crop = img[y1:y2, x1:x2]
            cv2.imwrite("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_2/img{}.jpg".format(img_name), img_crop)
        elif  label == 3:
            save_crop_img(label)
            img_crop = img[y1:y2, x1:x2]
            cv2.imwrite("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_3/img{}.jpg".format(img_name), img_crop)
        elif  label == 4:
            save_crop_img(label)
            img_crop = img[y1:y2, x1:x2]
            cv2.imwrite("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_4/img{}.jpg".format(img_name), img_crop)

def save_crop_img(label):
    if label == 0:
        os.makedirs("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_{}".format(0), exist_ok = True)
    elif label == 1:
        os.makedirs("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_{}".format(1), exist_ok = True)
    elif label == 2:
        os.makedirs("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_{}".format(2), exist_ok = True)
    elif label == 3:
        os.makedirs("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_{}".format(3), exist_ok = True)
    elif label == 4:
        os.makedirs("E:/Project/pythonProject/datastructure/few-shot/data_dbd/class_{}".format(4), exist_ok = True)

img_path = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train/Chase-1-_jpg.rf.ab18e4d1cea256c13a7483d89da63fa5.jpg"
label_path = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train/Chase-1-_jpg.rf.ab18e4d1cea256c13a7483d89da63fa5.txt"
path = "E:/Project/pythonProject/datastructure/few-shot/new_dbd/train"

list_txt = [i for i in os.listdir(path) if i.endswith(".txt")]
for text in list_txt:
    img_name = text.split("txt")[0] + "jpg"
    coord = get_coord(text)
    print(coord)
    save_crop(coord, img_name)
