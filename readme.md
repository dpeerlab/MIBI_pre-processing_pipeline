# Multiplex Ion Beam Imaging pre-processing tool - Python version 

Introduction: This tool is for MIBI data preprocessing and is the Python 3 version implementation of its original MATLAB version (https://github.com/lkeren/MIBIAnalysis) from Keren et al Cell 2018. Check the [jupyter notebook](https://github.com/dpeerlab/MIBI_pre-processing_pipeline/blob/master/code/MIBI_preprocessing_demo-2019-11-03.ipynb) in the code folder to see the usage. It involes background subtraction, noise removal, and aggregates removal.

Last update: 2019.11.03

Package: The versions of the package used in this tool are shown here.
scikit-learn==0.21.2
Pillow==6.0.0
pandas==0.24.2
numpy==1.16.0
matplotlib==3.1.0
scikit-image==0.16.2
scipy==1.2.1

## Example

![Example](./resource/example.png)

## Background subtraction: 

The background in MIBI image across all channels is highly similar to the blank channels and is usually located in the bare area on the slide. If not carefully handled, the noise can obscure the data. First, we define the blank channels (no antibody, usually mass 128-132) as the background channel. Then the background image was smoothed using a Gaussian kernel with radius as 3 pixels and masked as a binary image using thresholding method. All the signals in other channels are then subtracted by two counts and finally we recover negative counts as 0. Some channels have special background noise, which is usually from strong signal channels like gold. The same procedure should be additionally applied to these channels with the target channel as the background channel.

## Noise removal: 

The signal in MIBI data can be very low intensity values, e.g., single count, and the signal can also be very sparse. This property makes the noise and true signal in the data very similar. Therefore, to address the noise removal problem, we aim to look at the density of signal instead of the intensity of signal. For each pixel in the image, we calculate the distance to the nearest 25 counts. By thresholding the distribution of the distance in a bimodal distribution method, the counts with low confidence are removed.


## Aggregates removal:

During staining, some antibodies aggregate together and exhibited small, dense staining. These signals are not true signal and needs to be removed. We apply a Gaussian kernel with a radius being 1 pixel and binarize the image with Otsu thresholding method. The original signals in the bare area and small connected components area were removed. However, this step is optional.




## Acknowledgement 
This work is supported by Parker Institute for Cancer Immunotherapy
