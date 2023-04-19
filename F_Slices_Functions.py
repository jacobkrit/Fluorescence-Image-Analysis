from PIL import Image, ImageOps # load and show an image with Pillow
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats import beta # Distribution type
from sklearn import preprocessing # Normalization
from scipy.stats import pearsonr # Pearson Correlation
 
def Read_Image_Gray_Down_Scale(path, scale_down_percent):
    # Open the image form working directory
    image = Image.open(path)
    # convert image to numpy array
    image_data = asarray(image) 
    print("Original Image Shape:",image_data.shape)
    # applying grayscale method
    gray_image = ImageOps.grayscale(image)
    # convert image to numpy array
    gray_image_data = asarray(gray_image)
    print("Gray Original Image Shape:",gray_image_data.shape)
    # resize image
    scale_percent = scale_down_percent # percent of original size
    width = int(gray_image_data.shape[1] * scale_percent / 100)
    height = int(gray_image_data.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray_image_data_resized = cv.resize(gray_image_data, dim, interpolation = cv.INTER_AREA)
    print("Gray Image Resized Shape: ",gray_image_data_resized.shape)
    gray_image_data_resized_flattened = gray_image_data_resized.reshape(-1)
    print("Gray Image Resized Flattend Shape: ",gray_image_data_resized_flattened.shape)
    return gray_image_data_resized, gray_image_data_resized_flattened

def Edges_of_Images(array_2d, depth_num): # returns the edges of an Images fla
    edge_array = array_2d[:,0:depth_num].reshape(-1)
    edge_array = np.concatenate((edge_array, array_2d[:,array_2d.shape[1]-depth_num:array_2d.shape[1]].reshape(-1))) # returns the last depth_num columms
    edge_array = np.concatenate((edge_array, array_2d[0:depth_num,:].reshape(-1))) # returns the first depth_num rows
    edge_array = np.concatenate((edge_array, array_2d[array_2d.shape[0]-depth_num:array_2d.shape[0],:].reshape(-1))) # returns the last depth_num rows
    return edge_array

def Delete_Values_Lower_Than_Threshold(array_1d,threshold):
    list_1d = list(array_1d)
    i = 0
    while i < len(list_1d):
        if(list_1d[i]<=threshold):
            del list_1d[i]
        else:
            i += 1
    return list_1d

def Remove_From_Image_Edges_Background(array_2d,array_1d,edges_depth,percentile_threshold):
    # Edges of an Image
    array_edges_list = Edges_of_Images(array_2d,edges_depth)
    value_percentile = np.percentile(array_edges_list, percentile_threshold)
    print(str(percentile_threshold)+"th percentile of arr : ", value_percentile)
    array_1d_without_edges = Delete_Values_Lower_Than_Threshold(array_1d,value_percentile)
    return array_1d_without_edges

def Delete_Values_Array_From_Another(array1_1d,array2_1d): #delete all values from array1 which are in array2
    list1 = list(array1_1d)
    list2 = list(array2_1d)
    for j in range(0,len(list2)):
        value_check = list2[j]
        i = 0
        while i < len(list1):
            if(list1[i]==value_check):
                del list1[i]
                break
            else:
                i += 1
    return list1

def Add_Black_White_Limits(array_1d): #ADD 0 (back) and 255 (white) to the list of values
    array_1d = np.append(array_1d, [0])
    array_1d = np.append(array_1d, [255])
    return array_1d

def Array_1d_From_Pick_And_After(array_1d):# Takes an list (distribution list), finds the max value and returns the same list from this max value and after
    max_value = 0
    max_index = 0
    for i in range(0,len(array_1d)):
        if(array_1d[i]>max_value):
            max_value=array_1d[i]
            max_index=i
    print("Max value:",max_value,"  with Max index:",max_index)  
    array_1d_short = array_1d[max_index:]
    print("Distribution list Shape from Max value and after:",array_1d_short.shape)
    return array_1d_short

def Calculate_Mean_SD(array_1d,print_text):
    #calculate standard deviation and means of images
    array_std = np.std(array_1d)
    array_mean = np.mean(array_1d)
    print("Mean "+print_text+":",array_mean)
    print("SD "+print_text+":",array_std)
    return array_mean,array_std

def Plot_Basic(array_1d,print_title,print_y,print_x):
    # matplotlib histogram
    plt.figure(1, figsize=(7,5))
    plt.plot(array_1d)
    plt.title(print_title)
    plt.xlabel(print_x)
    plt.ylabel(print_y)
    plt.show()
    
def Plot_Basic_Histogram(array_1d,print_title,print_y,print_x):
    # matplotlib histogram
    plt.figure(1, figsize=(7,5))
    plt.hist(array_1d, color = 'blue', edgecolor = 'black', bins=100)
    plt.title("Histogram "+print_title)
    plt.xlabel(print_x)
    plt.ylabel(print_y)
    plt.show()
    
def Normalization_of_List(array_1d):   
    # Normalization
    reshape_list = np.array(array_1d.reshape(-1,1))
    #print('List:',reshape_list.shape)
    scaler_list = preprocessing.MinMaxScaler()
    array_1d_normalized = scaler_list.fit_transform(reshape_list)
    return array_1d_normalized

def Beta_Distribution_Best_Fit(array_1d):
    length_list = len(array_1d)
    a_best = 0
    b_best = 0
    corr_best = 0
    x_best = []
    y_best = []
    for a in np.arange(0.01, 5, 0.2):
        for b in np.arange(1, 100, 0.5):
            beta_x_val = np.linspace(0.01,0.99, length_list)
            beta_y_val = beta.pdf(beta_x_val, a, b)
            beta_y_val_normalized = Normalization_of_List(beta_y_val)
            beta_y_val_normalized = beta_y_val_normalized.reshape(-1)
            corr, _ = pearsonr(array_1d, beta_y_val_normalized)
            if(corr>corr_best):
                corr_best = corr
                a_best = a
                b_best = b
                x_best = beta_x_val
                y_best = beta_y_val_normalized
    print("Best Correlation:",corr_best, " with (a,b):", a_best, b_best)
    mean_best, var_best, skew_best, kurt_best = beta.stats(a_best, b_best, moments='mvsk')
    print("Mean, Variance, Skew, Kurtosis:",mean_best, var_best, skew_best, kurt_best)
    return a_best, b_best, mean_best, var_best, skew_best, kurt_best, x_best, y_best