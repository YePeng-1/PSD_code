import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
# import pandas as pd
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch
# import ImageCut
import Pattern_Learning as PELr
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
import datetime
from sklearn.metrics import roc_curve, auc
import read_results

def Show30Img(img_list, title, a_amp, a_num):
    plt.figure()
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        # plt.title(title+' #'+str(i))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_list[i],cmap='gray')
        # plt.colorbar()

    # filename = 'Figure/Case_Result_'+str(int(a_amp * 10)) + '_' + str(a_num) + title +'.png'
    # plt.savefig(filename)
    plt.show()

def Show30ImgAbs(img_list,title):
    plt.figure()
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        # plt.title(title + ' #' + str(i))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.abs(img_list[i]),cmap='gray')
        plt.colorbar()
    plt.show()

def ShowYAEImg(Y,Y0,A,E,GT,SETTING):

    plt.figure(figsize=(18, 2.5))
    plt.title(SETTING, fontsize=20)
    plt.rcParams['font.size'] = 12

    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Y,cmap='jet', vmin=-2.1, vmax=2.6)
    plt.colorbar()

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Y0, cmap='jet', vmin=-2.1, vmax=2.6)
    plt.colorbar()

    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(E,cmap='jet', vmin=-0.1, vmax=0.1)
    plt.colorbar()

    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(A,cmap='jet', vmin=-0.5, vmax=0.5)
    plt.colorbar()



    plt.subplot(1, 5, 5)
    # plt.title(title + ' #' + str(i))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(GT, cmap='gray')
    plt.colorbar()

    # filename = 'Figure/YAECase.png'
    # plt.savefig(filename)
    plt.show()

def ShowPSDImg(Y,Y_r,A,E,lable,SETTING):
    plt.figure(figsize=(18, 2.5))
    plt.title(SETTING, fontsize=20)
    plt.rcParams['font.size'] = 12

    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Y,cmap='jet', vmin=-2.1, vmax=2.6)
    plt.colorbar()

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Y_r,cmap='jet', vmin=-2.1, vmax=2.6)
    plt.colorbar()

    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(E,cmap='jet', vmin=-0.2, vmax=0.2)
    plt.colorbar()

    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(A,cmap='jet', vmin=-0.5, vmax=0.5)
    plt.colorbar()

    plt.subplot(1, 5, 5)
    # plt.title(title + ' #' + str(i))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(lable, cmap='gray')
    plt.colorbar()

    # filename = 'Figure/PSDCase.png'
    # plt.savefig(filename)
    plt.show()



def Show30Plot(arr_list, title):
    plt.figure()
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        # plt.title(title + ' #' + str(i), fontsize=5)
        plt.grid(False)
        plt.plot(arr_list[i])
    # filename = 'Figure/Case_Result_'+ title +'.png'
    # plt.savefig(filename)
    plt.show()

def PlotBoxPlot(FPR,FNR):
    plt.figure()
    plt.title('FPR and FNR Box Plot', fontsize=20)
    labels = 'FPR', 'FNR'
    plt.boxplot([FPR, FNR], labels=labels)
    plt.show()

def Showcase(FPR, FNR, DICE):
    # plt.figure()

    plt.figure()
    # plt.subplot(1,3,1)
    # plt.grid(False)
    # plt.title('#0 Origial Img')
    # plt.imshow(Y_list[0], cmap='gray')

    # plt.subplot(1, 3, 2)
    # plt.grid(False)
    # plt.title('#0 Ground Truth')
    # plt.imshow(gt_list[0], cmap='gray')

    # plt.title('FPR and FNR Box Plot')
    labels = 'FPR', 'FNR', 'DICE'
    plt.boxplot([FPR, FNR, DICE], labels=labels)
    # filename = 'Figure/Case_Result.png'
    # plt.savefig(filename)
    plt.show()

def Showcase2(FOR, FNR, BA, DICE,SETTING):
    plt.figure(figsize=(4, 4))
    plt.title(SETTING, fontsize=10)
    flierprops = dict(marker='o', markersize=16)
    labels = 'FOR', 'FNR', 'BA', 'DICE'
    plt.boxplot([FOR, FNR, BA, DICE], labels=labels, flierprops=flierprops)
    plt.show()

def FPR_FNR(A_list,label_list):
    label_pred_list = list()
    FPR_list = list()
    FNR_list = list()
    DICE_list = list()
    FOR_list = list()
    BA_list = list()
    CM_list = list()
    for i in range(len(A_list)):
        A_list_reshape = A_list[i].reshape(256*256)
        cond = np.where(A_list_reshape > 3)
        label_pred = np.zeros((256*256)).astype(int)
        label_pred[cond[0]] = 1
        label_true = np.zeros((256*256)).astype(int)
        label_list_reshape = label_list[i].reshape(256 * 256)
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
        label_pred_list.append(label_pred.reshape(256 , 256))
    return FPR_list,FNR_list,DICE_list,FOR_list,BA_list,CM_list,label_pred_list

def showimg(img, title = None):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    if title is None:
        plt.savefig("Figure/curimg.png")
    else:
        filename = 'Figure/'+ title +".png"
        plt.savefig(filename)

def BuildSignal(anomaly_amp, anomaly_num,T1,T2):
    sigma = 0.01 # 0.01 - 0.05

    # T1 = random.uniform(50, 80)
    # T2 = random.uniform(50, 80)

    size_y1 = 256
    size_y2 = 256
    Y0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    A0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    gt = np.zeros((size_y1, size_y2)).astype(np.float64)
    for i in range(size_y1):
        for j in range(size_y2):
            Y0[i, j] = np.sin(2 * np.pi / T1 * i) + np.cos(2 * np.pi / T2 * j)
    a_width = 32
    a_hight = 32
    for item in range(anomaly_num):

        start1 = random.randint(0, size_y1 - a_hight)
        start2 = random.randint(0, size_y2 - a_width)
        for i in range(a_hight):
            for j in range(a_width):
                A0[start1 + i, start2 + j] = anomaly_amp
                gt[start1 + i, start2 + j] = 1
    Y = Y0 + A0 + np.random.normal(0, sigma, Y0.shape)
    anomaly_per = a_width*a_hight*anomaly_num/(size_y1*size_y2)
    return Y, Y0, A0, gt ,anomaly_per


def Draw_Roc(a_score, label_, title_):
    y_true = np.zeros((label_.shape[0])).astype(int)
    cond = np.where(label_ > 0.5)
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
    plt.title(title_)
    plt.legend()
    plt.show()
    return




if __name__ == "__main__":


    # optimization parameters
    # learning_rate = 0.005 # 0.01, 0.001
    # learning_rate_A = 0.005 # 0.01, 0.001
    # sigma_sparse_A = 0.05  # 0.01,0.02,0.05
    # sigma_sparse_lambda = 10
    # sigma_sumlambda = 100
    anomaly_amp_list = list([0.5])
    anomaly_nums = [1,3,5]
    Ts=[50,65,80]

    dtype = torch.double
    device = torch.device("cpu")
    # anomaly_num = 5
    # anomaly_amp = 0.4
    st_all = datetime.datetime.now()
    for anomaly_num in anomaly_nums:
        for anomaly_amp in anomaly_amp_list:
            for T in Ts:
                T1 = T
                T2 = T
                anomaly_per_list = list()

                FPR_mean_list = list()
                FPR_std_list = list()
                FPR_max_list = list()
                FPR_min_list = list()

                FNR_mean_list = list()
                FNR_std_list = list()
                FNR_max_list = list()
                FNR_min_list = list()

                DICE_mean_list = list()
                DICE_std_list = list()
                DICE_max_list = list()
                DICE_min_list = list()

                FOR_mean_list = list()
                FOR_std_list = list()
                FOR_max_list = list()
                FOR_min_list = list()

                BA_mean_list = list()
                BA_std_list = list()
                BA_max_list = list()
                BA_min_list = list()
                anomaly_amp_appear = list()
                anomaly_num_appear = list()

                lr_l = 0.0005  # 0.01, 0.001
                lr_A = 0.005  # 0.01, 0.001
                rou_l1_lambda = 1000
                rou_l1_A = 0.15#,0.1
                rou_l2_E = 0.5  # ,0.1
                rou_row_sum = 100
                dtype = torch.double
                device = torch.device("cuda")
                max_iteration = 500


                saliency_list = list()
                Y0_list = list()
                E0_list = list()
                A0_list = list()
                images_list = list()
                label_list = list()
                lambda1_list = list()
                lambda2_list = list()
                A_list = list()
                label_pre_list = list()
                Y_r_list = list()
                E_list = list()
                anomaly_amp_appear.append(anomaly_amp)
                anomaly_num_appear.append(anomaly_num)
                for i in range(30):
                    Y, Y0, A0,gt,anomaly_per = BuildSignal(anomaly_amp,anomaly_num,T1,T2)
                    anomaly_per_list.append(anomaly_per)
                    E0_list.append(Y-Y0-A0)
                    A0_list.append(A0)
                    images_list.append(Y)
                    label_list.append(gt)
                    Y0_list.append(Y0)
                    label_gt = A0

                    st = datetime.datetime.now()
                    PE = PELr.Pattern_Extract(Y=Y, lr_l=lr_l, lr_A_E=lr_A,
                                              rou_l1_lambda=rou_l1_lambda,
                                              rou_l1_A=rou_l1_A,
                                              rou_row_sum=rou_row_sum,
                                              dtype=dtype, device=device, max_iteration=max_iteration, epsilon_lambda=1e-4,rou_l2_E=rou_l2_E)
                    PE.opti_iteration()
                    et = datetime.datetime.now()
                    # print('No.', i, ' image complete, detection time: ', (et - st).seconds, 's', (et - st).microseconds,
                    #       'us')

                    A_list.append(PE.A)
                    lambda1_list.append(PE.lambda1)
                    lambda2_list.append(PE.lambda2)
                    Y_r_list.append(PE.Y_r)
                    E_list.append(Y-PE.Y_r-PE.A)
                    A_std = np.sqrt(np.sum(np.square(PE.A))/(PE.A.size))
                    saliency_list.append(np.abs(PE.A)/A_std)
                print('anomaly_num',anomaly_num,)
                FPR_list, FNR_list,DICE_list,FOR_list,BA_list, CM_list,label_pred_list = FPR_FNR(saliency_list, label_list)
                # # Show30Img(label_pred_list,'Predict Label')
                anomaly_per_arr = np.array(anomaly_per_list)
                anomaly_per_mean = np.mean(anomaly_per_arr)
                FPR_arr = np.array(FPR_list)
                FNR_arr = np.array(FNR_list)
                DICE_arr = np.array(DICE_list)
                FOR_arr = np.array(FOR_list)
                BA_arr = np.array(BA_list)
                FPR_mean = np.mean(FPR_arr)
                FPR_std = np.std(FPR_arr)
                FPR_max = np.max(FPR_arr)
                FPR_min = np.min(FPR_arr)
                FPR_mean_list.append(FPR_mean)
                FPR_std_list.append(FPR_std)
                FPR_max_list.append(FPR_max)
                FPR_min_list.append(FPR_min)
                print('Experiment Setting - Anomaly Number:',anomaly_num, 'T1=T2:',T)
                print('FPR mean: ',str(FPR_mean),' std: ',str(FPR_std),' max: ', str(FPR_max),' min: ',str(FPR_min))

                FNR_mean = np.mean(FNR_arr)
                FNR_std = np.std(FNR_arr)
                FNR_max = np.max(FNR_arr)
                FNR_min = np.min(FNR_arr)
                FNR_mean_list.append(FNR_mean)
                FNR_std_list.append(FNR_std)
                FNR_max_list.append(FNR_max)
                FNR_min_list.append(FNR_min)
                print('FNR mean: ',str(FNR_mean),' std: ',str(FNR_std),' max: ', str(FNR_max),' min: ',str(FNR_min))

                DICE_mean = np.mean(DICE_arr)
                DICE_std = np.std(DICE_arr)
                DICE_max = np.max(DICE_arr)
                DICE_min = np.min(DICE_arr)
                DICE_mean_list.append(DICE_mean)
                DICE_std_list.append(DICE_std)
                DICE_max_list.append(DICE_max)
                DICE_min_list.append(DICE_min)
                print('DICE mean: ', str(DICE_mean), ' std: ', str(DICE_std), ' max: ', str(DICE_max), ' min: ', str(DICE_min))

                FOR_mean = np.mean(FOR_arr)
                FOR_std = np.std(FOR_arr)
                FOR_max = np.max(FOR_arr)
                FOR_min = np.min(FOR_arr)
                FOR_mean_list.append(FOR_mean)
                FOR_std_list.append(FOR_std)
                FOR_max_list.append(FOR_max)
                FOR_min_list.append(FOR_min)
                print('FOR mean: ', str(FOR_mean), ' std: ', str(FOR_std), ' max: ', str(FOR_max), ' min: ', str(FOR_min))

                BA_mean = np.mean(BA_arr)
                BA_std = np.std(BA_arr)
                BA_max = np.max(BA_arr)
                BA_min = np.min(BA_arr)
                BA_mean_list.append(BA_mean)
                BA_std_list.append(BA_std)
                BA_max_list.append(BA_max)
                BA_min_list.append(BA_min)
                print('BA mean: ', str(BA_mean), ' std: ', str(BA_std), ' max: ', str(BA_max), ' min: ', str(BA_min))

                # show
                # Showcase(FPR_arr, FNR_arr, DICE_arr)
                SETTING = 'Anomaly Number:'+str(anomaly_num)+'T1=T2:'+str(T)
                Showcase2(FOR_arr, FNR_arr, BA_arr, DICE_arr,SETTING)
                # Show30Img(label_list, 'GroudTruth', anomaly_amp, anomaly_num)
                # Show30Img(label_pred_list, 'PredictLabel', anomaly_amp, anomaly_num)
                # Show30Plot(lambda1_list, 'lambda_1')
                # Show30Plot(lambda2_list, 'lambda_2')
                ShowYAEImg(images_list[1],Y0_list[1],A0_list[1],E0_list[1],label_list[1],SETTING)
                ShowPSDImg(images_list[1],Y_r_list[1],A_list[1],E_list[1],label_pred_list[1],SETTING)
    et_all = datetime.datetime.now()
    print('Running time: ', (et_all - st_all).seconds, 's', (et_all - st_all).microseconds, 'us')


    # print("Complete")

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 生成数据
    FOR = np.asarray([[0.0,0.0,0.0],
                      [0.0,0.0,0.0],
                      [0.0029,0.0,0.0]]).transpose()
    FNR = np.asarray([[0.0, 0.0, 0.0395],
                      [0.0, 0.0, 0.0478],
                      [0.0001, 0.0796, 0.2588]]).transpose()
    BA = np.asarray([[1.0, 1.0, 0.9897],
                      [1.0, 1.0, 0.9876],
                      [0.9999, 0.9785, 0.9246]]).transpose()
    DICE = np.asarray([[1.0, 1.0, 0.9795],
                     [1.0, 1.0, 0.9751],
                     [0.9984, 0.9571, 0.8493]]).transpose()

    a_num = np.asarray([1,3,5]).astype(np.int32)
    t = np.asarray([50, 65, 80]).astype(np.int32)
    [T, A_num] = np.meshgrid(t, a_num)

    # FOR
    # 创建图形和3D轴
    figsize = 5
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制曲面
    for i in range(t.shape[0]):
        ax.plot(T[i],A_num[i],FOR[i],'-o')
        ax.plot(T[i],A_num[i],np.zeros_like(FOR[i]),'-o',color='gray',alpha=0.5)
        for j in range(a_num.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0],
                [T[i, j+1], A_num[i, j+1], 0],
                [T[i, j+1], A_num[i, j+1], FOR[i,j+1]],
                [T[i, j], A_num[i, j], FOR[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i/len(a_num)),alpha=0.3))
        for j in range(a_num.shape[0]):
            ax.text(T[i,j],A_num[i,j],FOR[i,j]+0.0003,f'{FOR[i,j]:.4f}',color='black',ha='center')

    for j in range(a_num.shape[0]):
        # ax.plot(T[i,j],A_num[i,j],FOR[:,j],'--',color='gray')
        ax.plot(T[:,j], A_num[:,j], np.zeros_like(FOR[:,j]), '-o', color='gray', alpha=0.5)
        for i in range(t.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0],
                [T[i+1, j], A_num[i+1, j], 0],
                [T[i+1, j], A_num[i+1, j], FOR[i+1,j]],
                [T[i, j], A_num[i, j], FOR[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.plasma(j/len(a_num)),alpha=0.3))

    # 设置标签和标题
    ax.set_xlabel('T')
    ax.set_ylabel('Anomaly Number')
    ax.set_zlabel('  FOR')
    ax.set_title('FOR')
    # # 设置视角
    ax.view_init(elev=30, azim=120)
    # 显示图形
    plt.show()

    # FNR
    # 创建图形和3D轴
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制曲面
    for i in range(t.shape[0]):
        ax.plot(T[i],A_num[i],FNR[i],'-o')
        ax.plot(T[i],A_num[i],np.zeros_like(FNR[i]),'-o',color='gray',alpha=0.5)
        for j in range(a_num.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0],
                [T[i, j+1], A_num[i, j+1], 0],
                [T[i, j+1], A_num[i, j+1], FNR[i,j+1]],
                [T[i, j], A_num[i, j], FNR[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i/len(a_num)),alpha=0.3))
        for j in range(a_num.shape[0]):
            ax.text(T[i,j],A_num[i,j],FNR[i,j]+0.0003,f'{FNR[i,j]:.4f}',color='black',ha='center')

    for j in range(a_num.shape[0]):
        # ax.plot(T[i,j],A_num[i,j],FNR[:,j],'--',color='gray')
        ax.plot(T[:,j], A_num[:,j], np.zeros_like(FNR[:,j]), '-o', color='gray', alpha=0.5)
        for i in range(t.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0],
                [T[i+1, j], A_num[i+1, j], 0],
                [T[i+1, j], A_num[i+1, j], FNR[i+1,j]],
                [T[i, j], A_num[i, j], FNR[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.plasma(j/len(a_num)),alpha=0.3))

    # 设置标签和标题
    ax.set_xlabel('T')
    ax.set_ylabel('Anomaly Number')
    ax.set_zlabel('  FNR')
    ax.set_title('FNR')
    # # 设置视角
    ax.view_init(elev=30, azim=120)
    # 显示图形
    plt.show()

    # BA
    # 创建图形和3D轴
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制曲面
    for i in range(t.shape[0]):
        ax.plot(T[i],A_num[i],BA[i],'-o')
        ax.plot(T[i],A_num[i],0.92+np.zeros_like(BA[i]),'-o',color='gray',alpha=0.5)
        for j in range(a_num.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0.92],
                [T[i, j+1], A_num[i, j+1], 0.92],
                [T[i, j+1], A_num[i, j+1], BA[i,j+1]],
                [T[i, j], A_num[i, j], BA[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i/len(a_num)),alpha=0.3))
        for j in range(a_num.shape[0]):
            ax.text(T[i,j],A_num[i,j],BA[i,j]+0.0003,f'{BA[i,j]:.4f}',color='black',ha='center')

    for j in range(a_num.shape[0]):
        # ax.plot(T[i,j],A_num[i,j],BA[:,j],'--',color='gray')
        ax.plot(T[:,j], A_num[:,j], 0.92+np.zeros_like(BA[:,j]), '-o', color='gray', alpha=0.5)
        for i in range(t.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0.92],
                [T[i+1, j], A_num[i+1, j], 0.92],
                [T[i+1, j], A_num[i+1, j], BA[i+1,j]],
                [T[i, j], A_num[i, j], BA[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.plasma(j/len(a_num)),alpha=0.3))

    # 设置标签和标题
    ax.set_xlabel('T')
    ax.set_ylabel('Anomaly Number')
    ax.set_zlabel('  BA')
    ax.set_title('BA')
    # # 设置视角
    ax.view_init(elev=30, azim=120)
    # 显示图形
    plt.show()

    # DICE
    # 创建图形和3D轴
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制曲面
    for i in range(t.shape[0]):
        ax.plot(T[i],A_num[i],DICE[i],'-o')
        ax.plot(T[i],A_num[i],0.84+np.zeros_like(DICE[i]),'-o',color='gray',alpha=0.5)
        for j in range(a_num.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0.84],
                [T[i, j+1], A_num[i, j+1], 0.84],
                [T[i, j+1], A_num[i, j+1], DICE[i,j+1]],
                [T[i, j], A_num[i, j], DICE[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i/len(a_num)),alpha=0.3))
        for j in range(a_num.shape[0]):
            ax.text(T[i,j],A_num[i,j],DICE[i,j]+0.0003,f'{DICE[i,j]:.4f}',color='black',ha='center')

    for j in range(a_num.shape[0]):
        # ax.plot(T[i,j],A_num[i,j],DICE[:,j],'--',color='gray')
        ax.plot(T[:,j], A_num[:,j], 0.84+np.zeros_like(DICE[:,j]), '-o', color='gray', alpha=0.5)
        for i in range(t.shape[0]-1):
            verts = [
                [T[i,j], A_num[i,j], 0.84],
                [T[i+1, j], A_num[i+1, j], 0.84],
                [T[i+1, j], A_num[i+1, j], DICE[i+1,j]],
                [T[i, j], A_num[i, j], DICE[i,j]]]
            ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.plasma(j/len(a_num)),alpha=0.3))

    # 设置标签和标题
    ax.set_xlabel('T')
    ax.set_ylabel('Anomaly Number')
    ax.set_zlabel('  DICE')
    ax.set_title('DICE')
    # # 设置视角
    ax.view_init(elev=30, azim=120)
    # 显示图形
    plt.show()

