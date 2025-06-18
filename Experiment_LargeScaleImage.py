import copy
import numpy as np
import torch
import datetime
import Anomaly_Detection
import Pattern_Learning as PELr
from numpy import fft
from matplotlib import pyplot as plt
from os import listdir, path

import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest
from scipy import stats
from scipy.stats import shapiro
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


import hnswlib
import random
from numba import jit

def Show30Img(img_list):
    plt.figure(figsize=(10, 10))
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_list[i])
        plt.colorbar()
    plt.show()
def plot_lambda_2(lambda1,lambda2):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(lambda1)
    plt.title('lambda_1')
    plt.subplot(2, 1, 2)
    plt.plot(lambda2)
    plt.title('lambda_2')
    plt.show()
    return
def show_image_colorbar_no_ticks(image,title = ''):
    plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title(title)
    plt.grid(False)
    plt.show()
    return
def Cal_AUC(a_score, label):
    y_true = np.zeros((label.shape[0])).astype(int)
    cond = np.where(label > 0.5)
    y_true[cond[0]] = 1
    fpr, tpr, thresholds = roc_curve(y_true, a_score)
    auc_value = auc(fpr, tpr)
    return auc_value, fpr, tpr, thresholds
def show_image_colorbar(image,title = ''):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.title(title)
    plt.grid(False)
    plt.show()
    return
def Draw_Roc(a_score, label):
    y_true = np.zeros((label.shape[0])).astype(int)
    cond = np.where(label > 0.5)
    y_true[cond[0]] = 1
    RocCurveDisplay.from_predictions(
        y_true,
        a_score,
        name="Anomaly ROC",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    return

def show_image(image,title = ''):
    plt.figure()
    plt.imshow(image,cmap='gray')
    # plt.imshow(image, cmap='jet', vmin=0, vmax=10)
    # plt.colorbar()
    plt.title(title)
    plt.grid(False)
    plt.show()
    return

def get_label_paths(gt_dir):
    gt_filepaths_ = []

    test_images = [path.splitext(file)[0]
                   for file
                   in listdir(gt_dir)
                   if path.splitext(file)[1] == '.png']
    gt_filepaths_.extend(
        [path.join(gt_dir, file+'.png')
         for file in test_images])

    return gt_filepaths_


def Build_Image(width = 4096, length = 4096,anomaly_num = 3,a_width = 50,a_hight = 200,anomaly_amp = 0.5,sigma = 0.05):
    T1 = random.randint(25, 40)
    T2 = random.randint(25, 40)

    size_y1 = width;
    size_y2 = length
    Y0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    A0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    gt = np.zeros((size_y1, size_y2)).astype(np.float64)
    for i in range(size_y1):
        for j in range(size_y2):
            Y0[i, j] = np.sin(2 * np.pi / T1 * i) + np.cos(2 * np.pi / T2 * j)
    for item in range(anomaly_num):

        start1 = random.randint(0, size_y1 - a_hight)
        start2 = random.randint(0, size_y2 - a_width)
        for i in range(a_hight):
            for j in range(a_width):
                A0[start1 + i, start2 + j] = anomaly_amp
                gt[start1 + i, start2 + j] = 1
    Y = Y0 + A0 + np.random.normal(0,sigma,Y0.shape)
    return Y,gt

def Build_Image_Shape(width = 4096, length = 4096,label_list=None,anomaly_amp = 0.5,sigma = 0.01):
    random_numbers = random.sample(range(1,256), len(label_list))
    T1 = 65
    T2 = 65

    size_y1 = width
    size_y2 = length
    Y0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    A0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    gt = np.zeros((size_y1, size_y2)).astype(np.float64)
    for i in range(size_y1):
        for j in range(size_y2):
            Y0[i, j] = np.sin(2 * np.pi / T1 * i) + np.cos(2 * np.pi / T2 * j)
    for i in range(len(label_list)):
        row_i = int(random_numbers[i]/16)
        col_j = random_numbers[i]%16
        A0[row_i * 256:(row_i + 1) * 256, col_j * 256:(col_j + 1) * 256] = (-(Y0[row_i * 256:(row_i + 1) * 256, col_j * 256:(col_j + 1) * 256])*label_list[i])
        gt[row_i * 256:(row_i + 1) * 256, col_j * 256:(col_j + 1) * 256] = label_list[i]
    Y = Y0 + A0 + np.random.normal(0, sigma, Y0.shape)
    return Y, gt


def Draw_A_score(a_score):
    # draw a_score
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(a_score, cmap='jet', vmin=0, vmax=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.show()
    return

def FPR_FNR(A_list,label_list,threshold = 4):
    label_pred_list = list()
    FPR_list = list()
    FNR_list = list()
    DICE_list = list()
    FOR_list = list()
    BA_list = list()
    CM_list = list()
    for i in range(len(A_list)):
        A_list_reshape = A_list[i].reshape(A_list[i].shape[0]*A_list[i].shape[1])
        cond = np.where(A_list_reshape > threshold)
        label_pred = np.zeros((A_list[i].shape[0]*A_list[i].shape[1])).astype(int)
        label_pred[cond[0]] = 1
        label_true = np.zeros((A_list[i].shape[0]*A_list[i].shape[1])).astype(int)
        label_list_reshape = label_list[i].reshape(A_list[i].shape[0]*A_list[i].shape[1])
        cond2 = np.where(label_list_reshape > 0.1)
        label_true[cond2[0]] = 1
        CM = confusion_matrix(label_true, label_pred)
        tn, fp, fn, tp = CM.ravel()
        FPR = fp/(fp+tn+ 1e-9)
        FNR = fn/(tp+fn+ 1e-9)
        DICE = 2*tp/(fp+fn+2*tp+ 1e-9)
        FOR = fp / (fp + tp + 1e-9)
        BA = tn / (2 * (tn + fp) + 1e-9) + tp / (2 * tp + fn + 1e-9)
        FPR_list.append(FPR)
        FNR_list.append(FNR)
        CM_list.append(CM)
        DICE_list.append(DICE)
        FOR_list.append(FOR)
        BA_list.append(BA)
        label_pred_list.append(label_pred.reshape(A_list[i].shape[0] , A_list[i].shape[1]))
    return FPR_list,FNR_list,DICE_list,FOR_list,BA_list,CM_list,label_pred_list


if __name__ == "__main__":
    label_path = "./Different_Shape_Labels"
    anomaly_amp_list = list([0.5])
    anomaly_nums = [1,3,5]
    Ts=[50,65,80]
    gt_filepaths = get_label_paths(label_path)
    label_list = list()
    pixel_0 = np.arange(256) * 4
    for i in range(len(gt_filepaths)):
        gt_path = gt_filepaths[i]
        label = cv2.imread(gt_path, 0)
        label = label[pixel_0, :]
        label = label[:, pixel_0]
        label = label / 255
        label_list.append(label)

    whole_img, whole_label = Build_Image_Shape(label_list=label_list)
    img =whole_img[0:256,0:256]
    label = whole_label[0:256,0:256]
    lable_int = label.astype(int)
    whole_label_int = whole_label.astype(int)

    RI = Anomaly_Detection.Resample_Image(img)
    RI.Resample(img)
    Y = RI.Y_ReS
    size_y1 = Y.shape[0]
    size_y2 = Y.shape[1]
    # Grid parameters
    lr_l = 0.0005  # 0.01, 0.001
    lr_A = 0.01  # 0.01, 0.001
    rou_l1_lambda = 1000
    rou_l1_A = 0.1  # ,0.1
    rou_l2_E = 0.5  # ,0.1
    rou_row_sum = 100
    dtype = torch.double
    device = torch.device("cuda")
    max_iteration = 100
    dtype = torch.double
    device = torch.device("cpu")
    endtime = datetime.datetime.now()

    starttime = datetime.datetime.now()
    PE = PELr.Pattern_Extract(Y=Y, lr_l=lr_l, lr_A_E=lr_A,
                              rou_l1_lambda=rou_l1_lambda,
                              rou_l1_A=rou_l1_A,
                              rou_row_sum=rou_row_sum,
                              dtype=dtype, device=device, max_iteration=max_iteration, epsilon_lambda=1e-4,
                              rou_l2_E=rou_l2_E)

    PE.opti_iteration()
    RI.Reconstruction_params(PE.lambda1, PE.lambda2, PE.Y_A_E)
    RI.Reconstruction()
    endtime = datetime.datetime.now()
    print('Pattern_Extract time',(endtime - starttime).seconds)


    whole_a_score = np.zeros(whole_img.shape)
    patch_width = 11
    kernel_std = 5
    ASE = Anomaly_Detection.A_Score_Estimate(RI.Y_recon, img, patch_width_=patch_width, kernel_std_=kernel_std)
    whole_ref_images = np.zeros(whole_img.shape)
    starttime = datetime.datetime.now()
    for img_numi in range(16):
        for img_numj in range(16):
            cur_img = whole_img[img_numi*256:(img_numi+1)*256,img_numj*256:(img_numj+1)*256]
            RI.Resample(cur_img)
            Y_A_E = PE.Decomposition(RI.Y_ReS,rou_A=0.1,lr = 0.1)
            RI.Reconstruction(Y_A_E)
            # RI.Reconstruction(RI.Y_ReS)
            whole_a_score[img_numi*256:(img_numi+1)*256,img_numj*256:(img_numj+1)*256] = ASE.Anomaly_score_map(cur_img,RI.Y_recon)
            whole_ref_images[img_numi*256:(img_numi+1)*256,img_numj*256:(img_numj+1)*256] = RI.Y_recon
    endtime = datetime.datetime.now()
    print('total process img time',(endtime - starttime).seconds)
    show_image(whole_img)

    show_image(whole_label)
    show_image(whole_ref_images)

    Draw_A_score(whole_a_score)

    A_std = np.sqrt(np.sum(np.square(whole_a_score)) / (whole_a_score.size))
    saliency_list = list()
    gt_list = list()
    saliency_list.append(np.abs(whole_a_score) / A_std)
    gt_list.append(whole_label)
    FPR_list, FNR_list, DICE_list, FOR_list, BA_list, CM_list, label_pred_list = FPR_FNR(saliency_list, gt_list,threshold=2)

    show_image(label_pred_list[0])







    # auc_value, fpr, tpr, thresholds = Cal_AUC(ProbE.a_score.flatten(), lable_int.flatten())
    # average_precision = average_precision_score(lable_int.flatten(), ProbE.a_score.flatten())
    # Draw_Roc(ProbE.a_score.flatten(), lable_int.flatten())
    # precision, recall, pr_thresholds = precision_recall_curve(lable_int.flatten(), ProbE.a_score.flatten())
    # plt.figure()
    # plt.plot(precision,recall)
    # plt.show()

    # i = 1
    # j = 2
    #
    # cur_img = whole_img[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
    # st = datetime.datetime.now()
    # RI.Resample(cur_img)
    # et = datetime.datetime.now()
    # print('Resample time', (et - st).seconds,'s',(et - st).microseconds,'us')
    #
    # st = datetime.datetime.now()
    # RI.Reconstruction()
    # et = datetime.datetime.now()
    # print('Reconstruction time', (et - st).seconds,'s',(et - st).microseconds,'us')
    # # show_image(RI.Y_recon)
    #
    # st = datetime.datetime.now()
    # whole_a_score[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = ASE.Anomaly_score_map(cur_img,RI.Y_recon)
    # et = datetime.datetime.now()
    # print('Anomaly_score_map time', (et - st).seconds,'s',(et - st).microseconds,'us')

    # Draw_A_score(whole_a_score[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256])
    # show_image(whole_label[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256])







