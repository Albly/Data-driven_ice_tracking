import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
import cv2
from tqdm import tqdm
import os
from PIL import Image

def parse_fragment_filename(filename):
    filename = filename.split(".")[0]
    _, idx, date_time_str = filename.split("_")
    try:
        idx = int(idx)
    except:
        pass
    try:
        date_time = datetime.strptime(date_time_str, '%Y-%m-%d-h%H').date()
        hour = int(date_time_str[-2:])
    except:
        date_time = datetime.strptime(date_time_str, '%Y-%m-%d').date()
        hour = -1
    return idx, date_time, hour


def get_line_length(coors):
    x1, y1, x2, y2 = coors
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def get_border_line(img):

    edges = cv2.Canny(np.uint8(img),0.,1.) # 0 and 1 - min and max values
    minLineLength = 20
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, rho=1, theta=0.1*np.pi/180, threshold=5,
                        minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is None:
        return None, None, None, None
    lines = [line[0] for line in lines]
    longest_line = max(lines, key=get_line_length)

    return longest_line


def detect_line_and_norm(idx, days_all,
                           date_dict, arr):

    orig = arr[idx]
    day = days_all[idx].date()
    my_orig = deepcopy(orig)
    normed_orig = deepcopy(orig)
    mask = np.zeros(orig.shape)
    if day in date_dict.keys():
        hours = list(date_dict[day].keys())
        if len(hours) > 1:
            mask = date_dict[day][hours[-1]]
            mask[mask>0] = 1
            my_orig = deepcopy(orig)
            normed_orig = deepcopy(orig)

            x1, y1, x2, y2 = get_border_line(mask)

            if x1:
                up_pixels = []
                down_pixels = []
                k, m = np.polyfit([y1,y2], [x1,x2], 1)
                for x in range(my_orig.shape[0]):
                    for y in range(my_orig.shape[1]):
                        if np.abs(y-(k*x+m)) <= 1:
                            my_orig[x][y] = 1
                        if y-(k*x+m)<7 and y-(k*x + m)>=0:
                            up_pixels.append(my_orig[x][y])
                        if y-(k*x+m)>=-7 and y-(k*x + m)<0:
                            down_pixels.append(my_orig[x][y])

                up_mean, down_mean = np.mean(up_pixels), np.mean(down_pixels)
                up_coef = (up_mean+down_mean)/(2*up_mean)
                down_coef = (up_mean+down_mean)/(2*down_mean)


                for x in range(normed_orig.shape[0]):
                    for y in range(normed_orig.shape[1]):
                        if y-(k*x+m)>=0:
                            normed_orig[x][y] = normed_orig[x][y] * up_coef
                        else:
                            normed_orig[x][y] = normed_orig[x][y] * down_coef

    return my_orig, normed_orig, mask



def preprocess_files_to_arr(path="/sea-ice/radars_fragmented/test",
                            path_to_gfs="gfs_frag0",
                            max_pixels=200, need_to_norm=False, fragment_idx=0,
                            gfs_names=[], pred_hours=[], gfs_shape=[158, 200],
                            gfs_norm_coefs={"Temperature":300, "U component of wind":30, "V component of wind":30},
                            train_size=0.8, path_to_preprocessed_arr=None):

    if not path_to_preprocessed_arr:
        all_files = os.listdir(path)
        files = []
        for file in all_files:
            if file.endswith('.npy'):
                files.append(file)
        files.sort()

        date_dict = {}
        for file in tqdm(files):
            idx, date_time, hour = parse_fragment_filename(file)
            #if idx != 0 :
                #print("hehehe", idx)
            if idx == fragment_idx:
                #print(idx, fragment_idx)
                a = np.load(os.path.join(path, file))[:, :, 0]
                if date_time not in date_dict.keys():
                    date_dict[date_time] = {}
                date_dict[date_time][hour] = a

        days = date_dict.keys()
        first_day, last_day = min(days), max(days)

        image_size = a.shape
        days_all = np.arange(first_day, last_day, timedelta(days=1)).astype(datetime)

        arr = np.zeros((len(days_all), image_size[0], image_size[1]))

    else:
        arr = np.load(path_to_preprocessed_arr)    #"/sea-ice/fragments_processed/fragment_0-5_2017-05-01_2023-03-01.npy")
        filename = os.path.basename(path_to_preprocessed_arr)
        filename = filename.split(".")[0]
        _, _, first_day, last_day = filename.split("_")
        first_day = datetime.strptime(first_day, '%Y-%m-%d').date()
        last_day = datetime.strptime(last_day, '%Y-%m-%d').date()
        days_all = np.arange(first_day, last_day, timedelta(days=1)).astype(datetime)

    arr_gfs = np.zeros((len(days_all), len(gfs_names)*len(pred_hours), gfs_shape[0], gfs_shape[1]))

    for idx, day in tqdm(enumerate(days_all)):

        if path_to_preprocessed_arr is None:
            day_int = day.date()
            if day_int in days:
                data = date_dict[day_int]

                for hour in data.keys():
                    a = data[hour]
                    arr[idx][a != 0] = a[a != 0]

        for name_idx, name in enumerate(gfs_names):
            for pred_hour_idx, pred_hour in enumerate(pred_hours):
                gfs_layer_idx = name_idx*len(pred_hours) + pred_hour_idx
                gfs_filename = f"frag{fragment_idx}_{day.strftime('%Y%m%d')}000000_pred{pred_hour:03d}_{name}.npy"
                gfs_filename = os.path.join(path_to_gfs, gfs_filename)

                '''if name in gfs_norm_coefs.keys():
                    norm_coef = gfs_norm_coefs[name]
                else:
                    norm_coef = 1'''

                try:
                    #print("HEHEHE")
                    #a_gfs = np.load(gfs_filename)/norm_coef
                    a_gfs = np.load(gfs_filename)
                    #print(a_gfs.shape)
                    #print(gfs_filename, np.sum(a_gfs))
                    #if np.sum(a_gfs) == 0:
                        #print(gfs_filename)
                    arr_gfs[idx][gfs_layer_idx] = a_gfs.T
                except:
                    pass
                    # print("PASSED: ", gfs_filename)


    for name_idx in range(len(gfs_names)):
        gfs = arr_gfs[:, name_idx]
        #gfs_mean = np.mean(gfs[:int(train_size*len(gfs))])
        gfs_max = np.max(gfs[:int(train_size*len(gfs))])
        gfs_min = np.min(gfs[:int(train_size*len(gfs))])
        if gfs_max != gfs_min:
            arr_gfs[:, name_idx] = (arr_gfs[:, name_idx] - gfs_min)/(gfs_max - gfs_min)
        #if gfs_max == 0:
            #gfs_max = 1
        #arr_gfs[:, name_idx] = (arr_gfs[:, name_idx] - gfs_mean)/gfs_max
    if np.max(arr)>32.:
        arr = arr / 256.

    arr_compressed = []
    for a in arr:
        im = Image.fromarray(a)
        im.thumbnail((max_pixels, max_pixels))
        arr_compressed.append(np.array(im))
    arr_compressed = np.array(arr_compressed)

    if need_to_norm:
        pass

    '''for idx, a in enumerate(arr_gfs):
        print(np.sum(a[0]))
        if np.sum(a) == 0:
            print("HEHEHEHE")'''

    return arr_compressed, arr_gfs


