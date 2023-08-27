import imageio
import numpy as np


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
    return mask_norm

def min_max_scaller(radar_data, new_min = 0, new_max = 1 ):
    v_min = np.min(radar_data, (1,2)).reshape(-1,1,1)
    v_max = np.max(radar_data, (1,2)).reshape(-1,1,1)
    new_max = np.array(new_max).reshape(-1,1,1)
    new_min = np.array(new_min).reshape(-1,1,1)

    normed_data = (radar_data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
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