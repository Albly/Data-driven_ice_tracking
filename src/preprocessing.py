import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


def load_mask(mask_path):
    '''
    loads binary mask with river from mask_path.
    
    ## Returns:
        - mask_norm: (2d np.array) normed (0-1) mask image
    '''
    mask = imageio.v3.imread(mask_path)
    mask = mask.mean(-1)
    v_min, v_max = mask.min(), mask.max()
    new_min, new_max = 1,0

    mask_norm = (mask - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    mask_norm = (mask_norm > 0.5).astype(int)
    return mask_norm

def min_max_scaller(radar_data, new_min = 0, new_max = 1 ):
    v_min = np.min(radar_data, (1,2)).reshape(-1,1,1)
    v_max = np.max(radar_data, (1,2)).reshape(-1,1,1)
    new_max = np.array(new_max).reshape(-1,1,1)
    new_min = np.array(new_min).reshape(-1,1,1)

    denum = v_max - v_min
    denum[denum == 0] = 1
    normed_data = (radar_data - v_min) / denum * (new_max - new_min) + new_min
    return normed_data

def mean_scaller(radar_data):
    v_min  = np.min(radar_data, (1,2)).reshape(-1,1,1)
    v_max  = np.max(radar_data, (1,2)).reshape(-1,1,1)
    v_mean = np.mean(radar_data, (1,2)).reshape(-1,1,1)

    normed_data = (radar_data - v_mean)/(v_max - v_min)
    return normed_data

def standard_scaller(radar_data):
    v_mean = np.mean(radar_data, (1,2)).reshape(-1,1,1)
    v_std  = np.std(radar_data, (1,2)).reshape(-1,1,1)

    normed_data = (radar_data - v_mean)/v_std
    return normed_data

def norm_scaller(radar_data, p = 2):
    norms = np.sum(np.abs(radar_data)**p, (1,2))**(1/p)
    norms = norms.reshape(-1,1,1)
    return radar_data / norms

def get_clear_data_idxs(radar_data, zero_th_percent , mask = None):
    '''
    Returns radar idxs where number of zeros less then @zero_th_percent in percent
    
    ## Inputs:
        - radar_data: (np.array) radar data (3d np array)
        - zero_th_percent : (float) threshold in percent. How many zeros are allowed. In interval (0, 100)
        - mask : (np.array) 2D mask where zeros should be counted. In mask is None - counts zeros on whole image
    
    ## Returns:
        - idxs of clear images
    '''

    assert 0 < zero_th_percent <= 100, 'zero_th_percent must be within interval 0 and 100'
    
    radar_post = radar_data * (radar_data >= 0)     # turn negatives to zeros
    zeros = radar_post == 0                         # get zeros positions 

    if mask is None:                                # if we have a mask
        N_pixls = zeros.sum(-1).sum(-1)             # number of all pixels
        zeros_selected = zeros                      # select all pixels 
    else:                                           # if no Mask
        zeros_selected = zeros * mask               # select zeros within mask
        N_pixls = mask.sum(-1).sum(-1)              # number of all pixels in mask

    n_zeros = zeros_selected.sum(-1).sum(-1)        # number of zeros per image
    p_zeros = n_zeros/N_pixls * 100                 # percentage of zeros per image
    nz_mask = p_zeros < zero_th_percent             # find required image positions

    return np.where(nz_mask)                        # return idxs of required images


def get_idxs_pairs(idxs):
    pairs = []
    i = 0
    while i < (len(idxs[0]) - 1):
        if (idxs[0][i] - idxs[0][i + 1]) == -1:
            pairs.append([idxs[0][i], idxs[0][i+1]])
            i += 2
        else:
            i += 1
    
    return pairs

def generate_diff_data(radar_data, mask):
    MASK_SUM = mask.sum()
    zeros_in_mask = np.logical_and(mask, np.logical_not(radar_data))                    # zeros-mask inside river 

    zero_percentage = np.sum(zeros_in_mask.astype(int), (1,2)) * 100 / MASK_SUM         # zeros ratio inside river
    filled_images_idxs = np.where(zero_percentage < 0.1)                                # idxs of fully filled images inside river

    dataset = []                                                                        # var where we store final data

    for fill_idx in filled_images_idxs[0]:                                              # go through all filled images
        output_image = torch.tensor([])                                                 # difference masks with unique data 
        input_radar = torch.tensor([])
        used_idxs = []                                                                  # idxs of radar data that have unique data
        filled_mask_bool = np.zeros((825, 200))                                              # mask-flag 
        filled_mask_days = np.zeros((825, 200)) 

        target_img = radar_data[fill_idx]                                               # target filled image that we want to predict
        mask = np.squeeze(mask)     

        for j in range(1,14):                                                           # 2 weeks memory
            if fill_idx - j < 0 : break                                                 # stop if we outside the memory

            cur_img = radar_data[fill_idx - j]                                          # past image 
            cur_mask = np.logical_and(mask, cur_img)                                    # past image inside river
            
            if np.sum(cur_mask) == 0:                                                   # if past image is empty inside river
                        continue                                                        # ommit this past image

            delta_cur = target_img - cur_img                                            # find differece between target and past image
            delta_mask = np.logical_and(abs(delta_cur) > 1e-6, cur_mask)                # find non-zero samples mask of the difference

            #zeros_in_deltas = np.logical_and(mask, np.logical_not(delta_mask))         
            other_in_delats = np.logical_and(mask, delta_mask)                          # non-zero samples mask of the difference inside the river
            
            new_data_mask = np.logical_and(np.logical_not(filled_mask_bool), other_in_delats)# select samples that has no information
            
            if new_data_mask.astype(int).sum() != 0:                                    # if there is new info
                data_layer = torch.from_numpy(delta_cur * new_data_mask).unsqueeze(0)   # select difference with new info
                cur_radar = torch.from_numpy(cur_img * new_data_mask).unsqueeze(0)
                
                filled_mask_days = filled_mask_days + new_data_mask * 1/j
                input_radar = torch.cat((input_radar, cur_radar), dim = 0)
                output_image = torch.cat((output_image, data_layer), dim = 0)           # store difference with new info
                used_idxs.append(fill_idx-j)                                            # store idx of data
                filled_mask_bool = np.logical_or(filled_mask_bool, other_in_delats)               # fill mask about having data

        current_data = {}
        current_data['target_idx'] = fill_idx
        current_data['input_data'] = input_radar.sum(0).unsqueeze(0)
        current_data['input_idxs'] = used_idxs
        current_data['difference'] = output_image.sum(0).unsqueeze(0)
        current_data['mask_days'] = torch.from_numpy(filled_mask_days).unsqueeze(0)

        dataset.append(current_data)

    return dataset



def generate_simple_data(radar_data, mask):
    MASK_SUM = mask.sum()
    zeros_in_mask = np.logical_and(mask, np.logical_not(radar_data))                    # zeros-mask inside river 

    zero_percentage = np.sum(zeros_in_mask.astype(int), (1,2)) * 100 / MASK_SUM         # zeros ratio inside river
    filled_images_idxs = np.where(zero_percentage < 0.1)                                # idxs of fully filled images inside river

    dataset = []                                                                        # var where we store final data

    for fill_idx in filled_images_idxs[0]:                                              # go through all filled images
        output_image = torch.tensor([])                                                 # difference masks with unique data 
        input_radar = torch.tensor([])
        used_idxs = []                                                                  # idxs of radar data that have unique data
        filled_mask_bool = np.zeros((825, 200))                                              # mask-flag 
        filled_mask_days = np.zeros((825, 200)) 

        target_img = radar_data[fill_idx]                                               # target filled image that we want to predict
        mask = np.squeeze(mask)     

        for j in range(1,14):                                                           # 2 weeks memory
            if fill_idx - j < 0 : break                                                 # stop if we outside the memory

            cur_img = radar_data[fill_idx - j]                                          # past image 
            cur_mask = np.logical_and(mask, cur_img)                                    # past image inside river
            
            if np.sum(cur_mask) == 0:                                                   # if past image is empty inside river
                        continue                                                        # ommit this past image

            delta_cur = target_img - cur_img                                            # find differece between target and past image
            delta_mask = np.logical_and(abs(delta_cur) > 1e-6, cur_mask)                # find non-zero samples mask of the difference

            #zeros_in_deltas = np.logical_and(mask, np.logical_not(delta_mask))         
            other_in_delats = np.logical_and(mask, delta_mask)                          # non-zero samples mask of the difference inside the river
            
            new_data_mask = np.logical_and(np.logical_not(filled_mask_bool), other_in_delats)# select samples that has no information
            
            if new_data_mask.astype(int).sum() != 0:                                    # if there is new info
                data_layer = torch.from_numpy(target_img * new_data_mask).unsqueeze(0)   # select difference with new info
                cur_radar = torch.from_numpy(cur_img * new_data_mask).unsqueeze(0)
                
                filled_mask_days = filled_mask_days + new_data_mask * 1/j
                input_radar = torch.cat((input_radar, cur_radar), dim = 0)
                output_image = torch.cat((output_image, data_layer), dim = 0)           # store difference with new info
                used_idxs.append(fill_idx-j)                                            # store idx of data
                filled_mask_bool = np.logical_or(filled_mask_bool, other_in_delats)               # fill mask about having data

        current_data = {}
        current_data['target_idx'] = fill_idx
        current_data['input_data'] = input_radar.sum(0).unsqueeze(0)
        current_data['input_idxs'] = used_idxs
        current_data['target_image'] = output_image.sum(0).unsqueeze(0)
        current_data['mask_days'] = torch.from_numpy(filled_mask_days).unsqueeze(0)

        dataset.append(current_data)

    return dataset



def generate_weather_data(radar_data, mask, gfs ,glorys):
    MASK_SUM = mask.sum()
    zeros_in_mask = np.logical_and(mask, np.logical_not(radar_data))                    # zeros-mask inside river 

    zero_percentage = np.sum(zeros_in_mask.astype(int), (1,2)) * 100 / MASK_SUM         # zeros ratio inside river
    filled_images_idxs = np.where(zero_percentage < 0.1)                                # idxs of fully filled images inside river

    dataset = []                                                                        # var where we store final data

    for fill_idx in filled_images_idxs[0]:                                              # go through all filled images
        output_image = torch.tensor([])                                                 # difference masks with unique data 
        input_radar = torch.tensor([])
        used_idxs = []                                                                  # idxs of radar data that have unique data
        filled_mask_bool = np.zeros((825, 200))                                              # mask-flag 
        filled_mask_days = np.zeros((825, 200)) 

        target_img = radar_data[fill_idx]                                               # target filled image that we want to predict
        mask = np.squeeze(mask)     

        for j in range(1,14):                                                           # 2 weeks memory
            if fill_idx - j < 0 : break                                                 # stop if we outside the memory

            cur_img = radar_data[fill_idx - j]                                          # past image 
            cur_mask = np.logical_and(mask, cur_img)                                    # past image inside river
            
            if np.sum(cur_mask) == 0:                                                   # if past image is empty inside river
                        continue                                                        # ommit this past image

            delta_cur = target_img - cur_img                                            # find differece between target and past image
            delta_mask = np.logical_and(abs(delta_cur) > 1e-6, cur_mask)                # find non-zero samples mask of the difference

            #zeros_in_deltas = np.logical_and(mask, np.logical_not(delta_mask))         
            other_in_delats = np.logical_and(mask, delta_mask)                          # non-zero samples mask of the difference inside the river
            
            new_data_mask = np.logical_and(np.logical_not(filled_mask_bool), other_in_delats)# select samples that has no information
            
            if new_data_mask.astype(int).sum() != 0:                                    # if there is new info
                data_layer = torch.from_numpy(delta_cur * new_data_mask).unsqueeze(0)   # select difference with new info
                cur_radar = torch.from_numpy(cur_img * new_data_mask).unsqueeze(0)
                
                filled_mask_days = filled_mask_days + new_data_mask * 1/j
                input_radar = torch.cat((input_radar, cur_radar), dim = 0)
                output_image = torch.cat((output_image, data_layer), dim = 0)           # store difference with new info
                used_idxs.append(fill_idx-j)                                            # store idx of data
                filled_mask_bool = np.logical_or(filled_mask_bool, other_in_delats)               # fill mask about having data

        current_data = {}
        current_data['gfs'] = gfs[fill_idx]
        current_data['glorys'] = glorys[fill_idx]
        current_data['target_idx'] = fill_idx
        current_data['input_data'] = input_radar.sum(0).unsqueeze(0)
        current_data['input_idxs'] = used_idxs
        current_data['difference'] = output_image.sum(0).unsqueeze(0)
        current_data['mask_days'] = torch.from_numpy(filled_mask_days).unsqueeze(0)

        dataset.append(current_data)

    return dataset



class Weather_Dataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) :
        input_data = self.dataset[index]['input_data']
        target_difference = self.dataset[index]['difference']
        days_mask = self.dataset[index]['mask_days']
        target_idx = self.dataset[index]['target_idx']
        gfs_data = torch.from_numpy(self.dataset[index]['gfs'])
        glorys_data = torch.from_numpy(self.dataset[index]['glorys'])
        input_data = torch.cat((input_data, gfs_data, glorys_data), dim = 0)

        return input_data, target_difference, days_mask, target_idx


class Difference_Dataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        input_data = self.dataset[index]['input_data']
        target_difference = self.dataset[index]['difference']
        days_mask = self.dataset[index]['mask_days']
        target_idx = self.dataset[index]['target_idx']
        #input_idxs = self.dataset[index]['input_idxs']

        return input_data, target_difference, days_mask, target_idx


class Sequential_dataset(Dataset):
    def __init__(self, radar_data, gfs, glorys):
        self.radar_data = []
        self.gfs = []
        self.glorys = []
        self.idxs = []

        for i in range(radar_data.shape[0]):
            self.radar_data.append(radar_data[i])
            self.gfs.append(gfs[i])
            self.glorys.append(glorys[i])
            self.idxs.append(i)

    def __len__(self):
        return len(self.radar_data)
         
    def __getitem__(self, index):
        input_data = self.radar_data[index]
        gfs_data = self.gfs[index]
        glorys_data = self.glorys[index]
        target_data = self.radar_data[index + 1]


class Simple_Dataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        input_data = self.dataset[index]['input_data']
        target_difference = self.dataset[index]['target_image']
        days_mask = self.dataset[index]['mask_days']
        target_idx = self.dataset[index]['target_idx']
        #input_idxs = self.dataset[index]['input_idxs']

        return input_data, target_difference, days_mask, target_idx

