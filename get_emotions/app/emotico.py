import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil


def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print(f'{path} already exist')


path_to_photos = '/home/user/Documents/to_server/photos'
path_to_data = '/home/user/Documents/to_server/abobus'
path_to_yolo_emo = '/home/user/Documents/to_server/yolo_emo_val'
create_directory(path_to_yolo_emo)

s, e, = 0, 100

emotions_dict = {'Surprise': 0,
                 'Anger': 1,
                 'Sadness': 2,
                 'Disquietment': 3,
                 'Fear': 4,
                 'Peace': 5, }

arr_val = pd.read_csv(f'{path_to_data}/annot_arrs_val.csv')
biggest_emotion = arr_val.iloc[:][emotions_dict.keys()].idxmax(axis=1)


def find_emotion(idx):
    return emotions_dict[biggest_emotion[idx]] \
        if not len(set(arr_val.iloc[0][emotions_dict.keys()])) else len(emotions_dict.keys())


def bigger_than_one(val):
    if val > 1:
        return 1
    else:
        return val


def create_file(data, path, mode='w'):
    with open(f'{path}', mode) as f:
        f.write(f'{data}\n')


def bbox_to_yolo(data, n_class=0, w=1280, h=720):
    # TODO add width, height
    """
        translate bbox coordinates to yolo format
        """
    # x1, y1 = data['x1'], data['y1']
    # x2, y2 = data['x2'], data['y2']
    x1, y1 = data[0], data[1]
    x2, y2 = data[2], data[3]

    yolo_x = bigger_than_one((int(abs(x1 + x2)) / 2) / w)
    yolo_y = bigger_than_one((int(abs(y1 + y2)) / 2) / h)
    yolo_w = bigger_than_one(int(abs(x1 - x2)) / w)
    yolo_h = bigger_than_one(int(abs(y1 - y2)) / h)

    return [n_class, yolo_x, yolo_y, yolo_w, yolo_h]


def main():
    for idx in range(s, e, 1):

        shutil.copy(os.path.join(path_to_photos, arr_val['Crop_name'][idx].replace('npy', 'png')),
                    f"{path_to_yolo_emo}")
        create_file(data=" ".join([str(x) for x in bbox_to_yolo([arr_val.iloc[idx]['X_min'], arr_val.iloc[idx]['Y_min'],
                                                                 arr_val.iloc[idx]['X_max'],
                                                                 arr_val.iloc[idx]['Y_max']],
                                                                n_class=find_emotion(idx),
                                                                w=arr_val.iloc[idx]['Width'],
                                                                h=arr_val.iloc[idx]['Height'])]),
                    path=os.path.join(path_to_yolo_emo, arr_val['Crop_name'][idx].replace('npy', 'txt')))


if __name__ == '__main__':
    main()
