import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
import Pattern_Learning_PA as PELr
import random
import pickle

def BuildSignal(sigma = 0.01,anomaly_amp = 0):
    anomaly_num = 1
    a_width = 32
    a_hight = 32
    T1 = 40
    T2 = 50
    size_y1 = 256
    size_y2 = 256
    Y0 = 0.2*np.ones((size_y1, size_y2)).astype(np.float64)
    A0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    gt = np.zeros((size_y1, size_y2)).astype(np.float64)
    for i in range(size_y1):
        for j in range(size_y2):
            Y0[i, j] = ((np.sin(2 * np.pi / T1 * i) + np.cos(2 * np.pi / T2 * j))/4*0.6+0.5)

    for item in range(anomaly_num):
        start1 = random.randint(0, size_y1 - a_hight)
        start2 = random.randint(0, size_y2 - a_width)
        for i in range(a_hight):
            for j in range(a_width):
                A0[start1 + i, start2 + j] = anomaly_amp
                gt[start1 + i, start2 + j] = 1
    Y = Y0 + A0 + np.random.normal(0,sigma,Y0.shape)
    return Y, Y0, A0, gt,T1,T2

def BuildSignal_square(sigma = 0.01,anomaly_amp = 0):
    anomaly_num = 1
    a_width = 32
    a_hight = 32
    T1 = 40
    T2 = 50
    size_y1 = 256
    size_y2 = 256
    Y0 = 0.2*np.ones((size_y1, size_y2)).astype(np.float64)
    A0 = np.zeros((size_y1, size_y2)).astype(np.float64)
    gt = np.zeros((size_y1, size_y2)).astype(np.float64)
    for i in range(size_y1):
        for j in range(size_y2):
            a = 0
            if i%T1 >=T1/2:a =1
            b = 0
            if j%T2 >=T2/2:b=1
            c = a+b
            if c==1: Y0[i, j]=0.8
            else: Y0[i, j]=0.2

    for item in range(anomaly_num):

        start1 = random.randint(0, size_y1 - a_hight)
        start2 = random.randint(0, size_y2 - a_width)
        for i in range(a_hight):
            for j in range(a_width):
                A0[start1 + i, start2 + j] = anomaly_amp
                gt[start1 + i, start2 + j] = 1
    Y = Y0 + A0 + np.random.normal(0,sigma,Y0.shape)
    return Y, Y0, A0, gt,T1,T2

def Draw_Images(images_list, vmin = 0, vmax = 1):
    plt.figure()
    for i in range(len(images_list)):
        # draw a_score
        plt.subplot(2, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # plt.imshow(images_list[i])
        plt.imshow(images_list[i],cmap = 'jet', vmin = vmin, vmax= vmax)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
    plt.show()
    return

def Draw_Figures(image_list,Y0_list,Yr_list,E_list,Er_list):
    plt.figure()
    num = len(image_list)
    column_num = 5
    for i in range(num):
        # draw img
        plt.subplot(num, column_num, i*column_num+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_list[i], cmap = 'jet', vmin = 0, vmax= 1)
        # draw Y0_list
        plt.subplot(num, column_num, i * column_num+2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Y0_list[i], cmap = 'jet', vmin = 0, vmax= 1)
        # draw Y_r
        plt.subplot(num, column_num, i * column_num+3)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Yr_list[i], cmap = 'jet', vmin = 0, vmax= 1)
        # draw E
        plt.subplot(num, column_num, i * column_num+4)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(E_list[i], cmap = 'jet', vmin = -0.1, vmax= 0.1)
        # draw E_r
        plt.subplot(num, column_num, i * column_num+5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Er_list[i], cmap = 'jet', vmin = -0.1, vmax= 0.1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6)
    plt.show()
    return

def Draw_Lambdas(lambda1_list,lambda2_list):
    num = len(lambda1_list)
    plt.figure()
    for i in range(num):

        plt.subplot(5, 3, i+1+(i//3)*6)
        plt.grid(False)
        # plt.imshow(images_list[i])
        plt.plot(lambda1_list[i])
        # plt.xlabel(r'$ \lambda_1 $',fontsize=6)


        plt.subplot(5, 3, i+4+(i//3)*6)
        plt.grid(False)
        plt.plot(lambda2_list[i])
        # plt.xlabel(r'$ \lambda_2 $',fontsize=6)

    plt.show()
    return

def Draw_Yr_E(Yr_list,Er_list):
    num = len(Yr_list)
    plt.figure()
    for i in range(num):
        plt.subplot(5, 3, i+1+(i//3)*6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Yr_list[i], cmap='jet', vmin=0, vmax=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6)

        plt.subplot(5, 3, i+4+(i//3)*6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Er_list[i], cmap='jet', vmin=-0.3, vmax=0.3)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6)

    plt.show()
    return

def Draw_Yr_E_Sparse(Yr_list,Er_list):
    num = len(Yr_list)
    plt.figure()
    for i in range(num):
        plt.subplot(4, 3, i//2+1+(i%2)*6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Yr_list[i], cmap='jet', vmin=0, vmax=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6)

        plt.subplot(4, 3, i//2+4+(i%2)*6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Er_list[i], cmap='jet', vmin=-0.2, vmax=0.2)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=6)

    plt.show()
    return

def Draw_Lambdas_Sparse(lambda1_list,lambda2_list):
    num = len(lambda1_list)
    plt.figure()
    for i in range(num):

        plt.subplot(4, 3, i//2+1+(i%2)*6)
        plt.grid(False)
        # plt.imshow(images_list[i])
        plt.plot(lambda1_list[i])
        # plt.xlabel(r'$ \lambda_1 $',fontsize=6)


        plt.subplot(4, 3, i//2+4+(i%2)*6)
        plt.grid(False)
        plt.plot(lambda2_list[i])
        # plt.xlabel(r'$ \lambda_2 $',fontsize=6)

    plt.show()
    return


def Simulaton():
    image_list = []
    Y0_list = []
    A0_list = []
    Yr_list = []
    E_list = []
    Er_list = []
    lambda1_list = []
    lambda2_list = []
    A_list = []
    img1, Y01, A01, gt,T1,T2 = BuildSignal(sigma = 0)
    image_list.append(img1)
    Y0_list.append(Y01)
    A0_list.append(A01)
    img2, Y02, A02, gt,T1,T2 = BuildSignal(sigma = 0.01)
    image_list.append(img2)
    Y0_list.append(Y02)
    A0_list.append(A02)
    img3, Y03, A03, gt,T1,T2 = BuildSignal(sigma = 0.05)
    image_list.append(img3)
    Y0_list.append(Y03)
    A0_list.append(A03)
    img4, Y04, A04, gt,T1,T2 = BuildSignal_square(sigma = 0)
    image_list.append(img4)
    Y0_list.append(Y04)
    A0_list.append(A04)
    img5, Y05, A05, gt,T1,T2 = BuildSignal_square(sigma = 0.01)
    image_list.append(img5)
    Y0_list.append(Y05)
    A0_list.append(A05)
    img6, Y06, A06, gt,T1,T2 = BuildSignal_square(sigma = 0.05)
    image_list.append(img6)
    Y0_list.append(Y06)
    A0_list.append(A06)

    lr_l = 0.0001  # 0.01, 0.001
    lr_A = 0.001  # 0.01, 0.001
    rou_l1_lambda = 100
    rou_l1_A = 100#,0.1
    rou_row_sum = 10
    dtype = torch.double
    device = torch.device("cuda")
    max_iteration = 100

    for i in range(3):
        img = image_list[i]
        E_list.append(image_list[i]-Y0_list[i])
        st = datetime.datetime.now()
        PE = PELr.Pattern_Extract(Y=img, lr_l=lr_l, lr_A_E=lr_A,
                                  rou_l1_lambda=rou_l1_lambda,
                                  rou_l1_A=rou_l1_A,
                                  rou_row_sum=rou_row_sum,
                                  dtype=dtype, device=device, max_iteration=max_iteration,epsilon_lambda=1e-3,rou_l2_E = 1)
        PE.opti_iteration()
        et = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(et - st).seconds, 's', (et - st).microseconds, 'us')

        Yr_list.append(PE.Y_r)
        Er_list.append(image_list[i]-Yr_list[i])
        A_list.append(PE.A)
        lambda1_list.append(PE.lambda1)
        lambda2_list.append(PE.lambda2)

    for i in range(3,6):
        img = image_list[i]
        E_list.append(image_list[i]-Y0_list[i])
        st = datetime.datetime.now()
        PE = PELr.Pattern_Extract(Y=img, lr_l=lr_l, lr_A_E=lr_A,
                                  rou_l1_lambda=rou_l1_lambda,
                                  rou_l1_A=rou_l1_A,
                                  rou_row_sum=rou_row_sum,
                                  dtype=dtype, device=device, max_iteration=max_iteration,epsilon_lambda=1e-4)
        PE.opti_iteration()
        et = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(et - st).seconds, 's', (et - st).microseconds, 'us')

        Yr_list.append(PE.Y_r)
        Er_list.append(Y0_list[i]-Yr_list[i])
        A_list.append(PE.A)
        lambda1_list.append(PE.lambda1)
        lambda2_list.append(PE.lambda2)

    return image_list,Y0_list,Yr_list,E_list,Er_list,A_list,lambda1_list,lambda2_list

def Simulaton_test_sparsity():
    image_list = []
    Y0_list = []
    A0_list = []
    Yr_list = []
    E_list = []
    Er_list = []
    lambda1_list = []
    lambda2_list = []
    A_list = []
    img1, Y01, A01, gt,T1,T2 = BuildSignal(sigma = 0.01)
    image_list.append(img1)
    Y0_list.append(Y01)
    A0_list.append(A01)
    img2, Y02, A02, gt,T1,T2 = BuildSignal_square(sigma = 0.01)
    image_list.append(img2)
    Y0_list.append(Y02)
    A0_list.append(A02)

    lr_l = 0.001  # 0.01, 0.001
    lr_A = 0.001  # 0.01, 0.001
    rou_l1_lambda = 0
    rou_l1_A = 100#,0.1
    rou_row_sum = 10
    dtype = torch.double
    device = torch.device("cuda")
    max_iteration = 100

    for i in range(2):
        img = image_list[i]
        E_list.append(image_list[i]-Y0_list[i])
        st = datetime.datetime.now()
        PE = PELr.Pattern_Extract(Y=img, lr_l=lr_l, lr_A_E=lr_A,
                                  rou_l1_lambda=rou_l1_lambda,
                                  rou_l1_A=rou_l1_A,
                                  rou_row_sum=rou_row_sum,
                                  dtype=dtype, device=device, max_iteration=max_iteration,epsilon_lambda=3e-4)
        PE.opti_iteration()
        et = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(et - st).seconds, 's', (et - st).microseconds, 'us')

        Yr_list.append(PE.Y_r)
        Er_list.append(PE.Y_r-Y0_list[i])
        A_list.append(PE.A)
        lambda1_list.append(PE.lambda1)
        lambda2_list.append(PE.lambda2)


    rou_l1_lambda = 100


    for i in range(2):
        img = image_list[i]
        E_list.append(image_list[i]-Y0_list[i])
        st = datetime.datetime.now()
        PE = PELr.Pattern_Extract(Y=img, lr_l=lr_l, lr_A_E=lr_A,
                                  rou_l1_lambda=rou_l1_lambda,
                                  rou_l1_A=rou_l1_A,
                                  rou_row_sum=rou_row_sum,
                                  dtype=dtype, device=device, max_iteration=max_iteration,epsilon_lambda=3e-4)
        PE.opti_iteration()
        et = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(et - st).seconds, 's', (et - st).microseconds, 'us')

        Yr_list.append(PE.Y_r)
        Er_list.append(PE.Y_r-Y0_list[i])
        A_list.append(PE.A)
        lambda1_list.append(PE.lambda1)
        lambda2_list.append(PE.lambda2)


    # rou_l1_lambda = 1000
    #
    # for i in range(2):
    #     img = image_list[i]
    #     E_list.append(image_list[i]-Y0_list[i])
    #     st = datetime.datetime.now()
    #     PE = PELr.Pattern_Extract(Y=img, lr_l=lr_l, lr_A_E=lr_A,
    #                               rou_l1_lambda=rou_l1_lambda,
    #                               rou_l1_A=rou_l1_A,
    #                               rou_row_sum=rou_row_sum,
    #                               dtype=dtype, device=device, max_iteration=max_iteration,epsilon_lambda=3e-4)
    #     PE.opti_iteration()
    #     et = datetime.datetime.now()
    #     print('No.', i ,' image complete, detection time: ',(et - st).seconds, 's', (et - st).microseconds, 'us')
    #
    #     Yr_list.append(PE.Y_r)
    #     Er_list.append(PE.Y_r-Y0_list[i])
    #     A_list.append(PE.A)
    #     lambda1_list.append(PE.lambda1)
    #     lambda2_list.append(PE.lambda2)

    return image_list,Y0_list,Yr_list,E_list,Er_list,A_list,lambda1_list,lambda2_list

if __name__ == "__main__":
    # image_list,Y0_list,Yr_list,E_list,Er_list,A_list,lambda1_list,lambda2_list = Simulaton()
    # Draw_Lambdas(lambda1_list, lambda2_list)
    # Draw_Yr_E(Yr_list, Er_list)

    st_all = datetime.datetime.now()
    image_list, Y0_list, Yr_list, E_list, Er_list, A_list, lambda1_list, lambda2_list = Simulaton_test_sparsity()
    et_all = datetime.datetime.now()
    print('Running time: ', (et_all - st_all).seconds, 's', (et_all - st_all).microseconds, 'us')
    Draw_Yr_E_Sparse(Yr_list, Er_list)
    Draw_Lambdas_Sparse(lambda1_list, lambda2_list)

    # Draw_Figures(image_list, Y0_list, Yr_list, E_list, Er_list)
    Draw_Images(image_list)

    for i in range(4):
        print(str(i)+" mean_Er"+str(np.mean(Er_list[i].flatten())))
        print(str(i)+" std_Er" + str(np.std(Er_list[i].flatten())))
        print(str(i)+" std_E" + str(np.std(E_list[i].flatten())))
        print(str(i) + " max_Er" + str(np.max(np.abs(Er_list[i].flatten()))))
        print(str(i) + " max_E" + str(np.max(np.abs(E_list[i].flatten()))))

    # save results
    data = {'image_list': image_list,
            'Y0_list': Y0_list,
            'Yr_list': Yr_list,
            'E_list': E_list,
            'Er_list': Er_list,
            'A_list': A_list,
            'lambda1_list': lambda1_list,
            'lambda2_list': lambda2_list}
    output = open("Test_result.pkl", 'wb')
    pickle.dump(data, output)
    output.close()

    plt.imshow(image_list[0], cmap='gray')
