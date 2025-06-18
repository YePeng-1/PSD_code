import copy
import numpy as np
import torch
import datetime
from numpy import fft
from matplotlib import pyplot as plt
import Pattern_Learning as PEoM
from statistics import geometric_mean
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
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
import hnswlib


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
    plt.title(title)
    plt.grid(False)
    plt.show()
    return

def M_Build(size_y):
    dM_dlambda = np.zeros((size_y*size_y,size_y))
    M = np.zeros((size_y * size_y,size_y))
    for i in range(size_y):
        for j in range(i):
            M[i*size_y+j, i-j] = 1.0
            dM_dlambda[i*size_y+j,i-j] = 1.0
        for j in range(i,size_y):
            M[i*size_y+j, j-i] = 1.0
            dM_dlambda[i*size_y+j, j-i] = 1.0
    return M,dM_dlambda
def lambdaSumMtc_Build(size_y):
    lambdaSumMtc = np.zeros((size_y,size_y))
    for i in range(size_y):
        for j in range(i):
            lambdaSumMtc[i, i-j] = lambdaSumMtc[i, i-j]+1.0
        for j in range(i,size_y):
            lambdaSumMtc[i, j-i] = lambdaSumMtc[i, j-i]+1.0
    return lambdaSumMtc

def k_nearest_neighbor(data, k =30):
    # Generating sample data
    num_elements = data.shape[0]
    dim = data.shape[1]
    ids = np.arange(num_elements)
    # Declaring index
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=32)
    # Element insertion (can be called several times):
    p.add_items(data, ids)
    # Controlling the recall by setting ef:
    p.set_ef(30)  # ef should always be > k
    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k)
    return labels, distances

class Resample_Image:
    def __init__(self,Y):
        self.Y = Y
        self.size_Y = self.Y.shape[0]
        return
    def Angle_Compute(self):
        theta_arr = np.arange(0,180.1,0.25)*np.pi/180
        # original img
        Y_spec = fft.fft2(self.Y)
        Y_spec_shift = fft.fftshift(Y_spec)
        self.Y_spec_shift_amplitude1 = np.abs(Y_spec_shift)
        thetaSum = np.zeros((theta_arr.shape[0])).astype(np.float64)
        self.power_specturm = self.Y_spec_shift_amplitude1 ** 2 / self.size_Y
        for theta in range(theta_arr.shape[0]):
            if (theta_arr[theta] > np.pi/4)  and (theta_arr[theta] < 3*np.pi/4) :
                for i in range(round(self.size_Y / 2)):
                    thetaSum[theta] = thetaSum[theta] + \
                                      self.power_specturm[i + round(self.size_Y / 2), -round(i * np.tan(np.pi / 2 - theta_arr[theta] )) + round(self.size_Y / 2)] + \
                                      self.power_specturm[round(self.size_Y / 2) - i, round(i * np.tan(np.pi / 2 - theta_arr[theta])) + round(self.size_Y / 2)]
            else:
                for i in range(round(self.size_Y / 2)):
                    thetaSum[theta] = thetaSum[theta] + \
                                      self.power_specturm[-round(i * np.tan(theta_arr[theta] )) + round(self.size_Y / 2), i + round(self.size_Y / 2)] + \
                                      self.power_specturm[round(i * np.tan(theta_arr[theta] )) + round(self.size_Y / 2), round(self.size_Y / 2) - i]
        self.max_index = thetaSum.argmax()
        self.max_theta_o = theta_arr[self.max_index]/np.pi*180
        self.max_theta = theta_arr[self.max_index]
        if self.max_theta>np.pi/2:
            self.max_theta = self.max_theta-np.pi/2
        if self.max_theta>np.pi/4:
            self.max_theta = self.max_theta-np.pi/2
        self.max_theta = -self.max_theta
        self.Route()
        return

    def Route(self):
        self.R_L = np.zeros((self.size_Y)).astype(int)
        self.dist = np.zeros((self.size_Y))
        for i in range(1,self.size_Y):
            if i+1+np.abs(self.R_L.sum()) == self.size_Y:
                break
            self.dist[i] = int(np.tan(self.max_theta)*i)-int(np.tan(self.max_theta)*(i-1))
            if (self.dist[i])>0.001:
                self.R_L[i] = 1
            if (self.dist[i]) < -0.001:
                self.R_L[i] = -1
            if i+1+np.abs(self.R_L.sum()) == self.size_Y:
                break
        self.RL_sum = self.R_L.sum()
        return


    def Resample(self):
        self.Angle_Compute()
        self.res_width = self.size_Y-np.abs(self.RL_sum)
        self.Y_ReS = np.zeros((self.res_width,self.res_width))
        if self.RL_sum<0:
            headCol = 0
            for i in range(self.res_width):
                headCol = headCol - self.R_L[self.res_width-1-i]
                cutRow = i - self.RL_sum
                for j in range(self.res_width):
                    cutRow = cutRow + self.R_L[j]
                    self.Y_ReS[i,j] = self.Y[cutRow,headCol+j]
        else:
            headCol = self.RL_sum-1
            for i in range(self.res_width):
                headCol = headCol - self.R_L[self.res_width-1-i]
                cutRow = i
                for j in range(self.res_width):
                    cutRow = cutRow + self.R_L[j]
                    self.Y_ReS[i,j] = self.Y[cutRow,headCol+j]
        return

    def Route_Backward(self):
        self.RL_Size_Y = np.zeros((self.size_Y)).astype(int)
        self.dist_Size_Y = np.zeros((self.size_Y))
        for i in range(1,self.size_Y):
            self.dist_Size_Y[i] = int(np.tan(-self.max_theta)*i)-int(np.tan(-self.max_theta)*(i-1))
            if (self.dist_Size_Y[i])>0.001:
                self.RL_Size_Y[i] = 1
            if (self.dist_Size_Y[i]) < -0.001:
                self.RL_Size_Y[i] = -1
        self.RL_Size_Y_sum = self.RL_Size_Y.sum()
        self.num_addrow = round((np.abs(self.RL_Size_Y_sum)+self.size_Y-self.res_width)/2)
        self.num_exp = self.res_width+2*self.num_addrow
        return

    def Reconstruction(self,Y_r,lambda1,lambda2):
        self.Route_Backward()
        num_addrow = self.num_addrow
        lambdaSumMtc = lambdaSumMtc_Build(self.res_width)
        M, dM_dlambda = M_Build(self.res_width)

        M_lambda2 = np.dot(M, lambda2).reshape(self.res_width, self.res_width)
        w2 = np.diag((1 / np.dot(lambdaSumMtc, lambda2)).reshape(self.res_width))

        lambda_head1 = copy.deepcopy(lambda1)
        lambda_head1[0] = 0
        sum_lambda_head1 = np.sum(lambda_head1)
        lambda_tail1 = lambda1[::-1]
        lambda_tail1[self.res_width-1] = 0
        sum_lambda_tail1 = np.sum(lambda_tail1)
        Y_ReS_exp = np.zeros((self.res_width+2*num_addrow,self.res_width+2*num_addrow))
        Y_ReS_exp[num_addrow:self.res_width+num_addrow,num_addrow:self.res_width+num_addrow] = self.Y_ReS
        for i in range(num_addrow):
            Y_cur = np.zeros((self.res_width,self.res_width))
            Y_cur[1:self.res_width,:] = Y_ReS_exp[num_addrow-i:self.res_width-1+num_addrow-i,num_addrow:self.res_width+num_addrow]
            Y_ReS_exp[num_addrow-i-1,num_addrow:self.res_width+num_addrow] = np.dot(np.dot(w2,M_lambda2),(np.dot(np.transpose(Y_cur),lambda_head1)/sum_lambda_head1)).flatten()
        for i in range(num_addrow):
            Y_cur = np.zeros((self.res_width, self.res_width))
            Y_cur[0:self.res_width-1, :] = Y_ReS_exp[num_addrow+i:self.res_width+num_addrow - 1 + i, num_addrow:self.res_width+num_addrow]
            Y_ReS_exp[self.res_width+num_addrow+i, num_addrow:self.res_width+num_addrow] = np.dot(np.dot(w2,M_lambda2),(np.dot(np.transpose(Y_cur), lambda_tail1) / sum_lambda_tail1)).flatten()


        M_lambda1 = np.dot(M, lambda1).reshape(self.res_width, self.res_width)
        w1 = np.diag((1 / np.dot(lambdaSumMtc, lambda1)).reshape(self.res_width))
        lambda_head2 = copy.deepcopy(lambda2)
        lambda_head2[0] = 0
        sum_lambda_head2 = np.sum(lambda_head2)
        lambda_tail2 = lambda2[::-1]
        lambda_tail2[self.res_width - 1] = 0
        sum_lambda_tail2 = np.sum(lambda_tail2)
        for i in range(num_addrow):
            Y_cur = np.zeros((self.res_width+2*num_addrow, self.res_width))
            Y_cur[:,1:self.res_width] = Y_ReS_exp[:,num_addrow - i:self.res_width - 1 + num_addrow - i]
            Y_ReS_exp[:,num_addrow - i - 1] = np.dot(np.dot(w1, M_lambda1), (np.dot(Y_cur, lambda_head2) / sum_lambda_head2)).flatten()
        for i in range(num_addrow):
            Y_cur = np.zeros((self.res_width+2*num_addrow, self.res_width))
            Y_cur[:,0:self.res_width - 1] = Y_ReS_exp[:,num_addrow + i:self.res_width+num_addrow - 1 + i]
            Y_ReS_exp[:,self.res_width+num_addrow + i] = np.dot(np.dot(w1, M_lambda1), (np.dot(Y_cur, lambda_tail2) / sum_lambda_tail2)).flatten()

        self.Y_r_exp = np.zeros((self.res_width+2*num_addrow,self.res_width+2*num_addrow))
        self.Y_r_exp[num_addrow:self.res_width+num_addrow,num_addrow:self.res_width+num_addrow] = Y_r
        self.Y_r_exp[0:num_addrow,:] = Y_ReS_exp[0:num_addrow,:]
        self.Y_r_exp[self.res_width+num_addrow:self.res_width+2*num_addrow,:] = Y_ReS_exp[self.res_width+num_addrow:self.res_width+2*num_addrow,:]
        self.Y_r_exp[:,0:num_addrow] = Y_ReS_exp[:,0:num_addrow]
        self.Y_r_exp[:,self.res_width+num_addrow:self.res_width+2*num_addrow] = Y_ReS_exp[:,self.res_width+num_addrow:self.res_width+2*num_addrow]
        self.Recover()
        return

    def Recover(self):
        self.Y_recon = np.zeros((self.size_Y, self.size_Y))
        center = self.num_exp / 2
        scale = np.square(self.res_width)/(np.square(self.RL_sum)+np.square(self.res_width))
        if self.RL_sum<0:
            headCol = self.RL_Size_Y_sum-1
            for i in range(self.size_Y):
                headCol = headCol - self.RL_Size_Y[self.size_Y-1-i]
                cutRow = i
                for j in range(self.size_Y):
                    cutRow = cutRow + self.RL_Size_Y[j]
                    self.Y_recon[i,j] = self.Y_r_exp[int((cutRow-center)*scale+center),int((headCol+j-center)*scale+center)]
        else:
            headCol = 0
            for i in range(self.size_Y):
                headCol = headCol - self.RL_Size_Y[self.size_Y-1-i]
                cutRow = i - self.RL_Size_Y_sum
                for j in range(self.size_Y):
                    cutRow = cutRow + self.RL_Size_Y[j]
                    self.Y_recon[i,j] = self.Y_r_exp[int((cutRow-center)*scale+center),int((headCol+j-center)*scale+center)]
        return


class Probability_Estimate():

    def Weighted_geometric_mean(self,value_arr,weight_vec):
        sum_arr = value_arr.shape[0]
        sum_weight = weight_vec.sum()
        pow_arr = np.power(value_arr, weight_vec)
        value = pow(geometric_mean(pow_arr),sum_arr/sum_weight)
        return value

    def __init__(self,Y,Y_r):
        self.Y = Y
        self.Y_r = Y_r
        self.size_Y = self.Y.shape[0]
        return

    def DPGMM(self,win_width=15):

        self.pca = PCA()
        self.pca.fit(self.neigh_vec_arr)
        # self.Samp, self.sampY = igmm_full_cov_sampler(self.neigh_vec_arr, cov_type='full', Nsamples=2000)
        # self.igmm = IGMM(self.neigh_vec_arr, .5, NIG, (0., 1., 1., 1.,), seqinit=True)
        # self.bgm = BayesianGaussianMixture(random_state=42,n_components=300)
        # self.bgm = BayesianGaussianMixture(
        #     n_components=10, covariance_type='full', weight_concentration_prior=1e+2,
        #     weight_concentration_prior_type='dirichlet_process',
        #     mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(self.neigh_vec_arr.shape[1]),
        #     init_params="kmeans", max_iter=100, random_state=2)
        #
        # self.bgm.fit(self.neigh_vec_arr)
        return

    def Probabilities_Estimate(self,clu_win_width=21,kernel_std = 5):
        win_width = clu_win_width
        self.kernel_std = kernel_std
        # knn query
        self.clu_win_width = clu_win_width
        win_side_wid = round((clu_win_width-1)/2)
        self.clu_win_side_wid = win_side_wid
        self.neigh_vec_arr = np.zeros(((self.size_Y-2*win_side_wid)**2,win_width**2))
        self.origial_neigh_vec_arr = np.zeros(((self.size_Y-2*win_side_wid)**2,win_width**2))

        for i in range(win_side_wid,self.size_Y-win_side_wid):
            for j in range(win_side_wid,self.size_Y-win_side_wid):
                self.neigh_vec_arr[(i-win_side_wid)*(self.size_Y-2*win_side_wid)+j-win_side_wid,:] = self.Y_r[i-win_side_wid:i+win_side_wid+1,j-win_side_wid:j+win_side_wid+1].flatten()
                self.origial_neigh_vec_arr[(i-win_side_wid)*(self.size_Y-2*win_side_wid)+j-win_side_wid,:] = self.Y[i-win_side_wid:i+win_side_wid+1,j-win_side_wid:j+win_side_wid+1].flatten()
        self.knn_label, self.knn_dist = k_nearest_neighbor(self.neigh_vec_arr)

        clu_width = (self.size_Y - 2 * self.clu_win_side_wid)

        self.mean_image = np.zeros((self.size_Y,self.size_Y))
        self.normalize_Y = np.zeros((self.size_Y,self.size_Y))

        care_ind = np.arange(self.clu_win_side_wid,self.size_Y-self.clu_win_side_wid,self.clu_win_width).astype(int)
        care_ind = np.concatenate([care_ind,[self.size_Y-self.clu_win_side_wid-1]])

        self.win_side_wid = round((clu_win_width-1)/2)
        self.a_score = np.zeros((self.size_Y,self.size_Y))
        self.a_score_pixel = np.zeros((self.size_Y,self.size_Y))
        weight = cv2.getGaussianKernel(clu_win_width, kernel_std)
        weight_Mtx =weight * weight.T
        self.weight_Vec = weight_Mtx.flatten()

        for ind1 in care_ind:
            for ind2 in care_ind:
                index = (ind1-self.clu_win_side_wid)*clu_width+(ind2-self.clu_win_side_wid)
                self.a_score_pixel[ind1 - self.clu_win_side_wid:ind1 + self.clu_win_side_wid + 1,
                ind2 - self.clu_win_side_wid:ind2 + self.clu_win_side_wid + 1] =np.square((self.origial_neigh_vec_arr[index,:]-np.average((self.origial_neigh_vec_arr[self.knn_label[index],:]), axis=0))/np.std(self.origial_neigh_vec_arr[self.knn_label[index],:],axis=0)).reshape(
                    self.clu_win_width, self.clu_win_width)

        self.exp_a_score_pixel = np.ones((self.size_Y+2*self.win_side_wid,self.size_Y+2*self.win_side_wid))
        self.exp_a_score_pixel[self.win_side_wid:self.size_Y+self.win_side_wid,self.win_side_wid:self.size_Y+self.win_side_wid] = self.a_score_pixel
        margin_sign =  np.zeros((self.size_Y+2*self.win_side_wid,self.size_Y+2*self.win_side_wid))
        margin_sign[self.win_side_wid:self.size_Y+self.win_side_wid,self.win_side_wid:self.size_Y+self.win_side_wid] = 1
        for i in range(self.size_Y):
            for j in range(self.size_Y):
                cur_margin_vec = margin_sign[ind1-self.win_side_wid:ind1+self.win_side_wid+1,ind2-self.win_side_wid:ind2+self.win_side_wid+1].flatten()
                cut_prob_arr = self.exp_a_score_pixel[ind1-self.win_side_wid:ind1+self.win_side_wid+1,ind2-self.win_side_wid:ind2+self.win_side_wid+1].flatten()
                cur_weight_Vec = cur_margin_vec*self.weight_Vec
                self.a_score[i, j] = np.dot(cut_prob_arr, cur_weight_Vec)

        # self.anomaly= np.zeros((self.size_Y,self.size_Y))
        # cond = np.where(self.a_score>(1/threshold_prob))
        # self.anomaly[cond[0],cond[1]] = 1
        return

    def knn_clustering(self):
        clu_width = (self.size_Y - 2 * self.clu_win_side_wid)
        self.knn_corre_index = list()
        k = self.knn_label.shape[1]
        self.knn_corre_index_arr = np.zeros((clu_width**2*k,2))
        for i in range(clu_width**2):
            cur_knn_indexs  = np.zeros((k,2))
            w_ind = i//clu_width
            h_ind = i%clu_width
            divmod_result = divmod(self.knn_label[i,:],clu_width)
            cur_knn_indexs[:,0] = divmod_result[0].astype(float) - w_ind
            cur_knn_indexs[:, 1] = divmod_result[1].astype(float) - h_ind
            self.knn_corre_index_arr[i*k:(i+1)*k,:] = cur_knn_indexs
            self.knn_corre_index.append(cur_knn_indexs.tolist())
        params = {"K": 2000, "I": 20, "P": 5, "THETA_M": 30, "THETA_S": 2,
                  "THETA_C": 5, "THETA_O": 0.1}
        self.knn_clu_class, self.knn_clu_dists = isodata.isodata(self.neigh_vec_arr, params)
        self.knn_ind_k = self.knn_clu_class.max()+1
        self.knn_clu_class_list = []
        for i in range(self.knn_ind_k):
            self.knn_clu_class_list.append(list())
        for i in range(self.knn_clu_class.shape[0]):
            self.knn_clu_class_list[self.knn_clu_class[i]].append(i)


if __name__ == "__main__":
    # file_name = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Image/metal_contamination_003.png'
    # label_file = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Label/metal_contamination_003_mask.png'
    # file_name = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Image/c.png'
    # label_file = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Label/bent_001_mask.png'
    file_name = "E:\Grid_Data_Set\Test_Set\Image/bent_000.png"
    label_file = "E:\Grid_Data_Set\Test_Set\Label/bent_000_mask.png"
    # file_name = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Image/broken_008.png'
    # label_file = '/Users/yepeng/Documents/文稿 - 叶芃的MacBook Pro/mvtec_anomaly_detection/grid/Test_Set/Label/broken_008_mask.png'
    # read image
    # file_name = '/Volumes/DATASETS/DataSets/mvtec_anomaly_detection/grid/test/bent/002.png'
    # imgPIL = Image.open(file_name)
    # enh_Brightness = ImageEnhance.Brightness(imgPIL)
    # img_Brightness = enh_Brightness.enhance(1.5)
    # img_array = np.asarray(img_Brightness)/255.0
    # pixel_0 = np.arange(256)*4
    # img = img_array[pixel_0,:]
    # img = img[:, pixel_0]
    lable_src = cv2.imread(label_file, 0)

    src = cv2.imread(file_name, 0)
    dst = cv2.equalizeHist(src)
    # img_mean = cv2.blur(dst, (15, 15))
    pixel_0 = np.arange(256) * 4
    img = src[pixel_0,:]/255
    img = img[:, pixel_0]

    win_width = 11
    win_side_width = round((win_width-1)/2)
    ref_neigh_rec = np.zeros((256,256))
    for i in range(win_side_width, 256-win_side_width):
        for j in range(win_side_width, 256-win_side_width):
            cur_patch = img[i-win_side_width:i+win_side_width+1,j-win_side_width:j+win_side_width+1]
            cur_patch = (cur_patch-cur_patch.mean())/cur_patch.std()
            ref_neigh_rec[i - win_side_width:i + win_side_width + 1, j - win_side_width:j + win_side_width + 1] = cur_patch

    img = ref_neigh_rec



    label = lable_src[pixel_0,:]/255
    label = label[:, pixel_0]
    label_int = np.zeros((label.shape[0],label.shape[0])).astype(int)
    cond = np.where(label>0.5)
    label_int[cond[0],cond[1]] = 1


    starttime = datetime.datetime.now()
    RI = Resample_Image(img)
    RI.Resample()



    Y = RI.Y_ReS
    size_y1 = Y.shape[0]
    size_y2 = Y.shape[1]
    # Grid parameters
    learning_rate = 0.001 # 0.01, 0.001
    # learning_rate_A = 0.002  # 0.01, 0.001
    # sigma_sparse_A = 0.3 # 0.01,0.02,0.05
    sigma_sparse_lambda = 10
    sigma_sumlambda = 100
    dtype = torch.double
    device = torch.device("cpu")

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

    PE = PEoM.Pattern_Extract(Y=Y, learning_rate=learning_rate,
                              rou_l1_lambda=sigma_sparse_lambda,
                              rou_row_sum=sigma_sumlambda,
                              dtype=dtype, device=device, max_iteration=250)
    starttime = datetime.datetime.now()
    PE.opti_iteration()

    norm1 = np.linalg.norm(PE.Y_r)
    norm2 = np.linalg.norm(Y)
    norm3 = np.linalg.norm(Y-PE.Y_r)
    E = (Y-PE.Y_r).reshape(-1,1)


    RI.Reconstruction(PE.Y_r,PE.lambda1,PE.lambda2)

    E2 = img - RI.Y_recon
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

    starttime = datetime.datetime.now()

    ProbE = Probability_Estimate(img,RI.Y_recon)
    ProbE.Probabilities_Estimate(clu_win_width=win_width,kernel_std = 5)

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

    figure = plt.figure()
    ax = Axes3D(figure)
    X = np.arange(0, ProbE.a_score.shape[0],1)
    Y2 = np.arange(0, ProbE.a_score.shape[0],1)
    X, Y2 = np.meshgrid(X, Y2)
    ax.plot_surface(X, Y2, ProbE.a_score, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    # ProbE.class_dists.std()


    # sp1 = RI.power_specturm
    # cond1 = np.where(sp1>1000)
    # sp1[cond1[0],cond1[1]] = 1000
    # show_image_colorbar(sp1)


    show_image_colorbar(ProbE.a_score)
    show_image_colorbar(img)
    show_image_colorbar(RI.Y_ReS)
    show_image_colorbar(RI.Y_recon)

    # show_image(img)
    show_image(img)
    # show power_specturm
    sp = copy.deepcopy(RI.power_specturm)
    cond1 = np.where(sp>1000)
    sp[cond1[0],cond1[1]] = 1000
    show_image_colorbar(sp)
    # show Res
    show_image(RI.Y_ReS)
    # show Y_r
    show_image(PE.Y_r)
    # show lambda
    PEoM.plot_lambda_2(PE.lambda1,PE.lambda2)
    # show Y_r_exp
    show_image(RI.Y_r_exp)
    # show Y_ReCon
    show_image(RI.Y_recon)
    # show pixel prob map
    # show_image_colorbar(ProbE.prob_Img)
    # show a_score map
    show_image_colorbar(ProbE.a_score)

    # show anomaly center
    # ind = np.argmax(ProbE.a_score)
    # ind_i = int(ind/256)
    # ind_j = int(ind - ind_i*256)
    #
    # ind_i_clu = ind_i-ProbE.clu_win_side_wid
    # ind_j_clu = ind_j - ProbE.clu_win_side_wid
    # clu_width = (ProbE.size_Y - 2 * ProbE.clu_win_side_wid)
    # class_ind = ProbE.neigh_class[ind_i_clu*clu_width+ind_j_clu]
    # class_map = ProbE.neigh_class.reshape(clu_width,clu_width)
    # class_map_ind =np.zeros((clu_width,clu_width)).astype(int)
    # cond = np.where(class_map == class_ind)
    # class_map_ind[cond[0],cond[1]] = 1
    # reference_patch = RI.Y_recon[ind_i-ProbE.win_side_wid:ind_i+ProbE.win_side_wid+1,ind_j-ProbE.win_side_wid:ind_j+ProbE.win_side_wid+1]
    # original_patch = img[ind_i-ProbE.win_side_wid:ind_i+ProbE.win_side_wid+1,ind_j-ProbE.win_side_wid:ind_j+ProbE.win_side_wid+1]
    # class_mean_patch = ProbE.class_neigh_distr[class_ind,:,0].reshape(ProbE.clu_win_width,ProbE.clu_win_width)
    # class_str_patch = ProbE.class_neigh_distr[class_ind, :, 1].reshape(ProbE.clu_win_width, ProbE.clu_win_width)

    # show_image_colorbar(class_map)
    # show_image_colorbar(class_map_ind)
    # show_image_colorbar(reference_patch,'reference_patch')
    # show_image_colorbar(original_patch,'original_patch')
    # show_image_colorbar(class_mean_patch,'class_mean_patch')
    # show_image_colorbar(class_str_patch,'class_str_patch')

    anomaly_draw = copy.deepcopy(ProbE.a_score)
    cond = np.where(anomaly_draw>1000)
    anomaly_draw[cond[0],cond[1]] = 1000
    show_image_colorbar(anomaly_draw)

    show_image_colorbar(img)
    show_image_colorbar(RI.Y_recon)

    # draw_list = list()
    # for  i in range(30):
    #     patch = ProbE.origial_neigh_vec_arr[ProbE.class_list[class_ind][i]].reshape(ProbE.clu_win_width,-1)
    #     draw_list.append(patch)
    #
    # Show30Img(draw_list)


    draw1_list = list()
    for i in range(30):
        patch = ProbE.origial_neigh_vec_arr[ProbE.knn_label[100][i],:].reshape(ProbE.clu_win_width,-1)
        draw1_list.append(patch)

    Show30Img(draw1_list)


    auc_value, fpr, tpr, thresholds = Cal_AUC(ProbE.a_score.flatten(), label_int.flatten())
    average_precision = average_precision_score(label_int.flatten(), ProbE.a_score.flatten())
    Draw_Roc(ProbE.a_score.flatten(), label_int.flatten())
    precision, recall, pr_thresholds = precision_recall_curve(label_int.flatten(), ProbE.a_score.flatten())
    plt.figure()
    plt.plot(precision,recall)
    plt.show()

    cv2.imwrite('/Users/yepeng/Desktop/SSIMtestImg/ima.png', img*256)
    cv2.imwrite('/Users/yepeng/Desktop/SSIMtestImg/imb.png', RI.Y_recon*256)

    # error_arr = (ProbE.normalize_Y-ProbE.mean_image).flatten()
    #
    # x = stats.norm.rvs(size=65536)
    # x = x*error_arr.std()+error_arr.mean()
    # # x = x*np.square(error_arr.std())+error_arr.mean()
    # kstest(x,'norm',(error_arr.mean(),error_arr.std()))
    # plt.figure()
    # plt.hist([x,error_arr],bins = 100)
    # plt.show()
    #
    # kstestResult = kstest((error_arr-error_arr.mean())/error_arr.std(),'norm')
    # kstest(error_arr ,'norm',(error_arr.mean(),error_arr.std()))
    #
    # result = stats.normaltest(error_arr)
    # print(result)
    # show_image_colorbar(ProbE.normalize_Y-ProbE.mean_image)
    #
    # stats.probplot(error_arr,dist = "norm",plot=plt)
    # plt.show()

    # starttime = datetime.datetime.now()
    # ProbE.DPGMM()
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)
    # # Y_pred = ProbE.bgm.predict(ProbE.neigh_vec_arr)
    # plt.figure()
    # plt.plot(ProbE.pca.explained_variance_ratio_)
    # plt.show()

    # origial_neigh_rec = np.zeros((256,256))
    # for i in range(5, 251):
    #     for j in range(5, 251):
    #         origial_neigh_rec[i-5:i+5+1,j-5:j+5+1] = ProbE.origial_neigh_vec_arr[(i-5)*(256-2*5)+j-5,:].reshape(11,11)
    #         # self.origial_neigh_vec_arr[(i-win_side_wid)*(self.size_Y-2*win_side_wid)+j-win_side_wid,:] = self.Y[i-win_side_wid:i+win_side_wid+1,j-win_side_wid:j+win_side_wid+1].flatten()
    #
    # show_image_colorbar(origial_neigh_rec)
    #
    # ref_neigh_rec = np.zeros((256,256))
    # for i in range(5, 251):
    #     for j in range(5, 251):
    #         ref_neigh_rec[i-5:i+5+1,j-5:j+5+1] = ProbE.neigh_vec_arr[(i-5)*(256-2*5)+j-5,:].reshape(11,11)
    #         # self.origial_neigh_vec_arr[(i-win_side_wid)*(self.size_Y-2*win_side_wid)+j-win_side_wid,:] = self.Y[i-win_side_wid:i+win_side_wid+1,j-win_side_wid:j+win_side_wid+1].flatten()
    #
    # show_image_colorbar(ref_neigh_rec)

    # starttime = datetime.datetime.now()
    # ProbE.knn_clustering()
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)
    #
    #
    # plt.figure()
    # plt.scatter(ProbE.knn_corre_index_arr[:,0],ProbE.knn_corre_index_arr[:,1])
    # plt.show()
    #
    # show_image(img)


