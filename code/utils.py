##### 2.17.2019 ######
#### PYTHON3 VERSION of MIBI low level analysis from Leeat's Matlab code
#### Translated by 
################## Yubin Xie from MSK Dana Pe'er lab
################## Daniel Li from Columbia Univ. Itsik Pe'er lab
#### Improvement can be made in 1. speed 2. interface.


import pandas as pd
import numpy as np
import matplotlib
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from PIL.TiffTags import TAGS

from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
from sklearn.neighbors import NearestNeighbors

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, label

from skimage.exposure import rescale_intensity

import glob

def ismember(a, b):
	bind = {}
	for i, elt in enumerate(b):
		if elt not in bind:
			bind[elt] = i
	return [bind.get(itm, None) for itm in a]

def read_tiff(massDS,file_name):
	img = Image.open(file_name)
	channel_name=[]
	im_array = []
	number_of_channels=0
	for i, page in enumerate(ImageSequence.Iterator(img)):
		im_array.append(np.array(page))
		number_of_channels+=1
		meta_dict = {TAGS[key] : page.tag[key] for key in page.tag.keys()}
		channel_name.append(meta_dict['PageName'][0].split('(')[0].strip())
	sorted_im_array = np.array(im_array)[[channel_name.index(x) for x in list(massDS.Label.values)]]

	return sorted_im_array.swapaxes(1,2).T

def MibiGetMask(rawMaskData, cap, t, gausRad, plot=False):
	if not cap:
		cap = 10
	if not t:
		t = 0.07
	if not gausRad:
		gausRad = 3
	rawMaskDataCap = rawMaskData
	rawMaskDataCap[np.where(rawMaskDataCap > cap)] = cap
	# smooth channel by gaussian, scale from 0 to 1 and threshold at 0.05
	bw = rescale_intensity(gaussian(rawMaskDataCap,sigma=gausRad))
	level = threshold_otsu(bw)
	if plot:
		plt.hist(bw.flatten(),bins=100)
		plt.axvline(level, color='r')
		plt.show()
	mask = (bw>t)*1
	return mask

def MibiRemoveBackgroundByMaskAllChannels(countsAllSFiltCRSum, mask,removeVal):
	channelNum = countsAllSFiltCRSum.shape[2]
	mask3d = np.array([mask for _ in range(channelNum)]).transpose(1,2,0)
	countsNoBg = np.copy(countsAllSFiltCRSum)
	masked_all_imgs = countsNoBg - removeVal * mask3d
	masked_all_imgs = masked_all_imgs.clip(min=0)
	return masked_all_imgs


def subtract_background(countsAllSFiltCRSum,corePath, gausRad, bgChannel, massDS, capBgChannel, t_threshold, removeVal, evalChannel, plot=False):
	coreNum = len(corePath)
	for i in range(0, coreNum):
		#loaded_data = loadmat('../../matlab_pipeline/codes/SampleData/extracted/Point1/data.mat')
		#countsAllSFiltCRSum = read_tiff(massDS,corePath[0]) #loaded_data['countsAllSFiltCRSum']
		bgChannelInd = ismember(bgChannel, massDS.Label)[0]
		mask = MibiGetMask(countsAllSFiltCRSum[:,:,bgChannelInd], capBgChannel, t_threshold, gausRad, plot=plot)
		countsNoBg = MibiRemoveBackgroundByMaskAllChannels(countsAllSFiltCRSum, mask, removeVal)
		evalChannelInd= ismember(evalChannel, massDS.Label)[0]
		if plot:
			plt.figure(figsize=(20,10),dpi=100)
			plt.subplot(1,2,1)
			plt.imshow(countsAllSFiltCRSum[:,:,evalChannelInd])
			plt.title('Before')
			plt.subplot(1,2,2)
			plt.imshow(countsNoBg[:,:,evalChannelInd])
			plt.title('after')
			plt.show()
			
	return countsNoBg, countsAllSFiltCRSum

def noise_level(countsNoBg,massDS):
	for j in range(len(massDS)): 
		level = threshold_otsu(countsNoBg[:,:,j])
		print(level)



def get_intND(countsNoBg, massDS):
	K = 25
	n2joinStart=2
	n2joinEnd=K
	intND=dict([])
	for j in range(len(massDS)):
		
		if massDS['NoiseT'][j]>0:
			print('Working on',j)
			dataA = countsNoBg[:,:,j]
			[xa,ya] = np.where(dataA>0)
			xAexpand = np.repeat(xa,dataA[dataA>0])
			yAexpand = np.repeat(ya,dataA[dataA>0])
			search_space = np.dstack([xAexpand, yAexpand]).squeeze()
			given_coordinates = np.dstack([xa, ya]).squeeze()
			[closestDvecBA,IDX] = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(search_space).kneighbors(given_coordinates)
			if len(closestDvecBA) == 0:
				intND[j] = []
			else:
				intND[j] = np.mean(closestDvecBA[:,n2joinStart-1:n2joinEnd-1], axis=1)
	return intND


def MibiFilterImageByNNThreshold(counts_all_filts_ipt, int_norm_d_i, thresh_vec_i):
	counts_all_filts_i = np.copy(counts_all_filts_ipt)
	posSignal = np.argwhere(counts_all_filts_i>0)
	if len(posSignal)!=0:
		inds = np.argsort(int_norm_d_i)[::-1]
		nnScorS = np.sort(int_norm_d_i)[::-1]
		pos = np.where(nnScorS>thresh_vec_i)
		indsToRemove = inds[pos]
		pos2Remove=posSignal[indsToRemove]
		counts_all_filts_i[pos2Remove[:,0],pos2Remove[:,1]] = 0
	return counts_all_filts_i
		
		
def denoise(countsNoBg, intND, massDS):
	countsNoNoise = np.zeros(countsNoBg.shape)
	for i in range(countsNoBg.shape[2]):
		
		if massDS['NoiseT'][i]>0:
			print('Working on',i)
			countsNoNoise[:,:,i] =  MibiFilterImageByNNThreshold(countsNoBg[:,:,i],intND[i],massDS['NoiseT'][i])
		else:
			countsNoNoise[:,:,i] = countsNoBg[:,:,i]
	return countsNoNoise



def MibiFilterAggregates(countsNoNoise, gausRad, t):
	countsNoAgg = np.copy(countsNoNoise)
	for idx in range(countsNoNoise.shape[2]):
		data =  countsNoNoise[:,:,idx]
		data_gauss = gaussian(data, sigma=gausRad,truncate=2)
		dataNoAgg = data.copy()
		#if np.sum(data_gauss)==0:
		threshold=0
		# else:
		# 	threshold = threshold_otsu(data_gauss)
		binarized = (data_gauss > threshold) * 1
		label_img = label(binarized)
		#print('Total connected components: %d' % (len(np.unique(label_img))))
		stats = regionprops(label_img)
		for i in stats:
			if i.area<t:
				coords = np.array( i.coords)
				dataNoAgg[coords[:,0], coords[:,1]]=0
		countsNoAgg[:,:,idx] = dataNoAgg
	return countsNoAgg



def read_Angelo_image(massDS,folder):
    img_array=[]
    image_list = glob.glob(folder)
    name_list = []
    for image_path in image_list:
        name=image_path.split('/')[-1].split('.')[0]
        if name == 'totalIon':
            continue
        name_list.append(name)
        img = Image.open(image_path)
        img_array.append(np.array(img))
    order = [[name_list.index(x) for x in list(massDS.Label.values)]]
    img_array = np.array(img_array)  
    sorted_im_array = img_array[tuple(order)]
    
#     print(sorted(name_list),len(set(name_list)),img_array.shape)
#     print(sorted_im_array.shape)
    return sorted_im_array.swapaxes(1,2).T

def read_MIBI_tracker_tiff(massDS,corePath):
    channel_name=[]
    im_array = []
    number_of_channels=0
    im = Image.open(corePath)
    for i, page in enumerate(ImageSequence.Iterator(im)):
        im_array.append(np.array(page))
        number_of_channels+=1
        meta_dict = {TAGS[key] : page.tag[key] for key in page.tag.keys()}
        channel_name.append(meta_dict['PageName'][0].split('(')[0].strip())
    sorted_im_array = np.array(im_array)[[channel_name.index(x) for x in list(massDS.Label.values)]]
    
    return sorted_im_array.swapaxes(1,2).T







