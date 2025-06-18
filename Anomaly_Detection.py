
import copy
import matplotlib.pyplot as plt
from numpy import fft
import cv2
import hnswlib
from numba import jit
from Pattern_Learning import *
import argparse


def M_Build(size_y):
    M = np.zeros((size_y * size_y, size_y))
    for i in range(size_y):
        for j in range(i):
            M[i * size_y + j, i - j] = 1.0
        for j in range(i, size_y):
            M[i * size_y + j, j - i] = 1.0
    return M


def lambdaSumMtc_Build(size_y):
    lambdaSumMtc = np.zeros((size_y, size_y))
    for i in range(size_y):
        for j in range(i):
            lambdaSumMtc[i, i - j] = lambdaSumMtc[i, i - j] + 1.0
        for j in range(i, size_y):
            lambdaSumMtc[i, j - i] = lambdaSumMtc[i, j - i] + 1.0
    return lambdaSumMtc


@jit(nopython=True)
def loop_in_resample(RL_sum, res_width, R_L, Y_, Y_ReS):
    if RL_sum < 0:
        headCol = 0
        for i in range(res_width):
            headCol = headCol - R_L[res_width - 1 - i]
            cutRow = i - RL_sum
            for j in range(res_width):
                cutRow = cutRow + R_L[j]
                Y_ReS[i, j] = Y_[cutRow, headCol + j]
    else:
        headCol = RL_sum - 1
        for i in range(res_width):
            headCol = headCol - R_L[res_width - 1 - i]
            cutRow = i
            for j in range(res_width):
                cutRow = cutRow + R_L[j]
                Y_ReS[i, j] = Y_[cutRow, headCol + j]
    return Y_ReS


@jit(nopython=True)
def loop_in_recover(Y_recon, Y_r_exp, num_exp, res_width, RL_sum, RL_Size_Y_sum, size_Y, RL_Size_Y):
    center = round(num_exp / 2)
    scale = np.square(res_width) / (np.square(RL_sum) + np.square(res_width))
    if RL_sum < 0:
        headCol = RL_Size_Y_sum - 1
        for i in range(size_Y):
            headCol = headCol - RL_Size_Y[size_Y - 1 - i]
            cutRow = i
            for j in range(size_Y):
                cutRow = cutRow + RL_Size_Y[j]
                Y_recon[i, j] = Y_r_exp[
                    int((cutRow - center) * scale + center), int((headCol + j - center) * scale + center)]
    else:
        headCol = 0
        for i in range(size_Y):
            headCol = headCol - RL_Size_Y[size_Y - 1 - i]
            cutRow = i - RL_Size_Y_sum
            for j in range(size_Y):
                cutRow = cutRow + RL_Size_Y[j]
                Y_recon[i, j] = Y_r_exp[
                    int((cutRow - center) * scale + center), int((headCol + j - center) * scale + center)]
    return Y_recon


@jit(nopython=True)
def loop_in_Reconstruction(Y_ReS_exp, num_addrow, res_width, w2, M_lambda2, lambda_head1, sum_lambda_head1,
                           lambda_tail1, sum_lambda_tail1, w1, M_lambda1, lambda_head2, sum_lambda_head2, lambda_tail2,
                           sum_lambda_tail2):
    for i in range(num_addrow):
        Y_cur = np.zeros((res_width, res_width))
        Y_cur[1:res_width, :] = Y_ReS_exp[num_addrow - i:res_width - 1 + num_addrow - i,
                                num_addrow:res_width + num_addrow]
        Y_ReS_exp[num_addrow - i - 1, num_addrow:res_width + num_addrow] = np.dot(np.dot(w2, M_lambda2), (
                np.dot(np.transpose(Y_cur), lambda_head1) / sum_lambda_head1)).flatten()
        Y_cur = np.zeros((res_width, res_width))
        Y_cur[0:res_width - 1, :] = Y_ReS_exp[num_addrow + i:res_width + num_addrow - 1 + i,
                                    num_addrow:res_width + num_addrow]
        Y_ReS_exp[res_width + num_addrow + i, num_addrow:res_width + num_addrow] = np.dot(
            np.dot(w2, M_lambda2),
            (np.dot(np.transpose(Y_cur), lambda_tail1) / sum_lambda_tail1)).flatten()

        Y_cur = np.zeros((res_width, res_width))
        Y_cur[:, 1:res_width] = Y_ReS_exp[num_addrow:res_width + num_addrow,
                                num_addrow - i:res_width - 1 + num_addrow - i]
        Y_ReS_exp[num_addrow:res_width + num_addrow, num_addrow - i - 1] = np.dot(np.dot(w1, M_lambda1), (
                np.dot(Y_cur, lambda_head2) / sum_lambda_head2)).flatten()
        Y_cur = np.zeros((res_width, res_width))
        Y_cur[:, 0:res_width - 1] = Y_ReS_exp[num_addrow:res_width + num_addrow,
                                    num_addrow + i:res_width + num_addrow - 1 + i]
        Y_ReS_exp[num_addrow:res_width + num_addrow, res_width + num_addrow + i] = np.dot(np.dot(w1, M_lambda1), (
                np.dot(Y_cur, lambda_tail2) / sum_lambda_tail2)).flatten()
    for i in range(num_addrow):
        # EDGE
        Y_cur = np.zeros((num_addrow, res_width))
        Y_cur[:, 1:res_width] = Y_ReS_exp[0:num_addrow,
                                num_addrow - i:res_width - 1 + num_addrow - i]
        Y_ReS_exp[0:num_addrow, num_addrow - i - 1] = (np.dot(Y_cur, lambda_head2) / sum_lambda_head2).flatten()
        Y_cur = np.zeros((num_addrow, res_width))
        Y_cur[:, 0:res_width - 1] = Y_ReS_exp[0:num_addrow, num_addrow + i:res_width + num_addrow - 1 + i]
        Y_ReS_exp[0:num_addrow, res_width + num_addrow + i] = (np.dot(Y_cur, lambda_tail2) / sum_lambda_tail2).flatten()

        Y_cur = np.zeros((num_addrow, res_width))
        Y_cur[:, 1:res_width] = Y_ReS_exp[res_width + num_addrow:res_width + 2 * num_addrow,
                                num_addrow - i:res_width - 1 + num_addrow - i]
        Y_ReS_exp[res_width + num_addrow:res_width + 2 * num_addrow, num_addrow - i - 1] = (
                np.dot(Y_cur, lambda_head2) / sum_lambda_head2).flatten()
        Y_cur = np.zeros((num_addrow, res_width))
        Y_cur[:, 0:res_width - 1] = Y_ReS_exp[res_width + num_addrow:res_width + 2 * num_addrow,
                                    num_addrow + i:res_width + num_addrow - 1 + i]
        Y_ReS_exp[res_width + num_addrow:res_width + 2 * num_addrow, res_width + num_addrow + i] = (
                np.dot(Y_cur, lambda_tail2) / sum_lambda_tail2).flatten()

    return Y_ReS_exp


@jit(nopython=True)
def loop_in_Anomaly_score_map_getCareVec(cur_patch_vec, cur_original_patch_vec, Y_r, Y_, care_ind, patch_side_width):
    care_ind_num = care_ind.shape[0]
    for i in range(care_ind_num):
        for j in range(care_ind_num):
            ind1 = care_ind[i]
            ind2 = care_ind[j]
            cur_patch_vec[i * care_ind_num + j, :] = Y_r[(ind1 - patch_side_width):(ind1 + patch_side_width + 1),
                                                     (ind2 - patch_side_width):(ind2 + patch_side_width + 1)].flatten()
            cur_original_patch_vec[i * care_ind_num + j, :] = Y_[(ind1 - patch_side_width):(ind1 + patch_side_width + 1),
                                                              (ind2 - patch_side_width):(
                                                                      ind2 + patch_side_width + 1)].flatten()
    return cur_patch_vec, cur_original_patch_vec


@jit(nopython=True)
def loop_in_Anomaly_score_map_calPixelAScore(a_score_pixel, patch_side_width, cur_original_patch_vec, knn_average,
                                             knn_std, patch_width_, care_ind):
    care_ind_num = care_ind.shape[0]
    for i in range(care_ind_num):
        ind1 = care_ind[i]
        for j in range(care_ind_num):
            ind2 = care_ind[j]
            index = i * care_ind_num + j
            a_score_pixel[(ind1 - patch_side_width):(ind1 + patch_side_width + 1),
            (ind2 - patch_side_width):(ind2 + patch_side_width + 1)] = np.square(
                (cur_original_patch_vec[index] - knn_average[index]) / (knn_std[index] + 1e-6)).reshape(patch_width_,
                                                                                                        patch_width_)
    return a_score_pixel


@jit(nopython=True)
def loop_in_Anomaly_score_map_wgt_ave(a_score_, exp_a_score_pixel, size_Y, patch_side_width, margin_sign, weight_Vec):
    for i in range(size_Y):
        ind1 = i + patch_side_width
        for j in range(size_Y):
            ind2 = j + patch_side_width
            a_score_[i, j] = np.dot(exp_a_score_pixel[(ind1 - patch_side_width):(ind1 + patch_side_width + 1),
                                   (ind2 - patch_side_width):(ind2 + patch_side_width + 1)].flatten(),
                                   margin_sign[(ind1 - patch_side_width):(ind1 + patch_side_width + 1),
                                   (ind2 - patch_side_width):(
                                           ind2 + patch_side_width + 1)].flatten() * weight_Vec)
    return a_score_


class Resample_Image:
    def __init__(self, init_Y):
        self.init_Y = init_Y
        self.size_Y = self.init_Y.shape[0]
        self.Y_recon = None
        self.Y_r_exp = None
        self.Y_ReS = None
        self.Y = None
        self.sum_lambda_tail2 = None
        self.lambda_tail2 = None
        self.sum_lambda_head2 = None
        self.lambda_head2 = None
        self.w1 = None
        self.M_lambda1 = None
        self.sum_lambda_tail1 = None
        self.lambda_tail1 = None
        self.sum_lambda_head1 = None
        self.lambda_head1 = None
        self.w2 = None
        self.M_lambda2 = None
        self.lambda2 = None
        self.lambda1 = None
        self.num_exp = None
        self.num_addrow = None
        self.RL_Size_Y_sum = None
        self.dist_Size_Y = None
        self.RL_Size_Y = None
        self.res_width = None
        self.RL_sum = None
        self.dist = None
        self.R_L = None
        self.max_theta = None
        self.max_theta_o = None
        self.max_index = None
        self.Y_spec_shift_amplitude1 = None
        self.Y_sub_A = None
        self.power_spectrum = None
        self.Angle_Compute()
        return

    def Angle_Compute(self):
        theta_arr = np.arange(0, 180.1, 0.25) * np.pi / 180
        # original img
        Y_spec = fft.fft2(self.init_Y)
        Y_spec_shift = fft.fftshift(Y_spec)
        self.Y_spec_shift_amplitude1 = np.abs(Y_spec_shift)
        thetaSum = np.zeros((theta_arr.shape[0])).astype(np.float64)
        self.power_spectrum = self.Y_spec_shift_amplitude1 ** 2 / self.size_Y
        for theta in range(theta_arr.shape[0]):
            if (theta_arr[theta] > np.pi / 4) and (theta_arr[theta] < 3 * np.pi / 4):
                for i in range(round(self.size_Y / 2)):
                    thetaSum[theta] = thetaSum[theta] + \
                                      self.power_spectrum[i + round(self.size_Y / 2), -round(
                                          i * np.tan(np.pi / 2 - theta_arr[theta])) + round(self.size_Y / 2)] + \
                                      self.power_spectrum[round(self.size_Y / 2) - i, round(
                                          i * np.tan(np.pi / 2 - theta_arr[theta])) + round(self.size_Y / 2)]
            else:
                for i in range(round(self.size_Y / 2)):
                    thetaSum[theta] = thetaSum[theta] + \
                                      self.power_spectrum[
                                          -round(i * np.tan(theta_arr[theta])) + round(self.size_Y / 2), i + round(
                                              self.size_Y / 2)] + \
                                      self.power_spectrum[
                                          round(i * np.tan(theta_arr[theta])) + round(self.size_Y / 2), round(
                                              self.size_Y / 2) - i]
        self.max_index = thetaSum.argmax()
        self.max_theta_o = theta_arr[self.max_index] / np.pi * 180
        self.max_theta = theta_arr[self.max_index]
        if self.max_theta > np.pi / 2:
            self.max_theta = self.max_theta - np.pi / 2
        if self.max_theta > np.pi / 4:
            self.max_theta = self.max_theta - np.pi / 2
        self.max_theta = -self.max_theta
        self.Route()
        self.Route_Backward()
        return

    def Route(self):
        self.R_L = np.zeros(self.size_Y).astype(int)
        self.dist = np.zeros(self.size_Y)
        for i in range(1, self.size_Y):
            if i + 1 + np.abs(self.R_L.sum()) == self.size_Y:
                break
            self.dist[i] = int(np.tan(self.max_theta) * i) - int(np.tan(self.max_theta) * (i - 1))
            if (self.dist[i]) > 0.001:
                self.R_L[i] = 1
            if (self.dist[i]) < -0.001:
                self.R_L[i] = -1
            if i + 1 + np.abs(self.R_L.sum()) == self.size_Y:
                break
        self.RL_sum = self.R_L.sum()
        self.res_width = self.size_Y - np.abs(self.RL_sum)
        return

    def Route_Backward(self):
        self.RL_Size_Y = np.zeros(self.size_Y).astype(int)
        self.dist_Size_Y = np.zeros(self.size_Y)
        for i in range(1, self.size_Y):
            self.dist_Size_Y[i] = int(np.tan(-self.max_theta) * i) - int(np.tan(-self.max_theta) * (i - 1))
            if (self.dist_Size_Y[i]) > 0.001:
                self.RL_Size_Y[i] = 1
            if (self.dist_Size_Y[i]) < -0.001:
                self.RL_Size_Y[i] = -1
        self.RL_Size_Y_sum = self.RL_Size_Y.sum()
        self.num_addrow = round((np.abs(self.RL_Size_Y_sum) + self.size_Y - self.res_width) / 2)
        self.num_exp = self.res_width + 2 * self.num_addrow
        return

    def Reconstruction_params(self, lambda1, lambda2, Y_sub_A):
        self.Y_sub_A = Y_sub_A
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        M = M_Build(self.res_width)
        lambdaSumMtc = lambdaSumMtc_Build(self.res_width)
        self.M_lambda2 = np.dot(M, lambda2).reshape(self.res_width, self.res_width)
        self.w2 = np.diag((1 / np.dot(lambdaSumMtc, lambda2)).reshape(self.res_width))
        self.lambda_head1 = copy.deepcopy(lambda1)
        self.lambda_head1[0] = 0
        self.sum_lambda_head1 = np.sum(self.lambda_head1)
        self.lambda_tail1 = lambda1[::-1]
        self.lambda_tail1[self.res_width - 1] = 0
        self.sum_lambda_tail1 = np.sum(self.lambda_tail1)

        self.M_lambda1 = np.dot(M, lambda1).reshape(self.res_width, self.res_width)
        self.w1 = np.diag((1 / np.dot(lambdaSumMtc, lambda1)).reshape(self.res_width))
        self.lambda_head2 = copy.deepcopy(lambda2)
        self.lambda_head2[0] = 0
        self.sum_lambda_head2 = np.sum(self.lambda_head2)
        self.lambda_tail2 = lambda2[::-1]
        self.lambda_tail2[self.res_width - 1] = 0
        self.sum_lambda_tail2 = np.sum(self.lambda_tail2)
        return

    def Resample(self, Y_):
        self.Y = Y_
        self.Y_ReS = np.zeros((self.res_width, self.res_width))
        self.Y_ReS = loop_in_resample(self.RL_sum, self.res_width, self.R_L, Y_, self.Y_ReS)
        return

    def Reconstruction(self,Y_r = None):
        # Y_r = np.transpose(
        #     np.dot(np.dot(self.w2, self.M_lambda2),
        #            np.transpose(np.dot(np.dot(self.w1, self.M_lambda1), self.Y_sub_A))))
        if Y_r is None:
            Y_r = self.Y_sub_A
        num_addrow = self.num_addrow
        Y_ReS_exp = np.zeros((self.res_width + 2 * num_addrow, self.res_width + 2 * num_addrow))
        Y_ReS_exp[num_addrow:self.res_width + num_addrow, num_addrow:self.res_width + num_addrow] = Y_r
        Y_ReS_exp = loop_in_Reconstruction(Y_ReS_exp, num_addrow, self.res_width, self.w2, self.M_lambda2,
                                           self.lambda_head1, self.sum_lambda_head1,
                                           self.lambda_tail1, self.sum_lambda_tail1, self.w1, self.M_lambda1,
                                           self.lambda_head2, self.sum_lambda_head2,
                                           self.lambda_tail2, self.sum_lambda_tail2)

        self.Y_r_exp = np.zeros((self.res_width + 2 * num_addrow, self.res_width + 2 * num_addrow))
        self.Y_r_exp[num_addrow:self.res_width + num_addrow, num_addrow:self.res_width + num_addrow] = Y_r
        self.Y_r_exp[0:num_addrow, :] = Y_ReS_exp[0:num_addrow, :]
        self.Y_r_exp[self.res_width + num_addrow:self.res_width + 2 * num_addrow, :] = Y_ReS_exp[
                                                                                       self.res_width + num_addrow:self.res_width + 2 * num_addrow,
                                                                                       :]
        self.Y_r_exp[:, 0:num_addrow] = Y_ReS_exp[:, 0:num_addrow]
        self.Y_r_exp[:, self.res_width + num_addrow:self.res_width + 2 * num_addrow] = Y_ReS_exp[:,
                                                                                       self.res_width + num_addrow:self.res_width + 2 * num_addrow]
        self.Recover()
        return

    def Recover(self):
        self.Y_recon = np.zeros((self.size_Y, self.size_Y))
        self.Y_recon = loop_in_recover(self.Y_recon, self.Y_r_exp, self.num_exp, self.res_width, self.RL_sum,
                                       self.RL_Size_Y_sum, self.size_Y, self.RL_Size_Y)
        return


class A_Score_Estimate:
    def __init__(self, init_Y_r, init_Y, patch_width_=17, kernel_std_=5):
        self.p = None
        self.original_neigh_vec_arr = None
        self.a_score_pixel = None
        self.init_Y_r = init_Y_r
        self.init_Y = init_Y
        self.size_Y = init_Y_r.shape[0]
        self.patch_width = patch_width_
        self.patch_side_width = round((patch_width_ - 1) / 2)
        self.sqr_patch_num = (self.size_Y - 2 * self.patch_side_width)
        self.kernel_std = kernel_std_
        weight = cv2.getGaussianKernel(self.patch_width, self.kernel_std)
        self.weight_Vec = (weight * weight.T).flatten()
        self.care_ind = np.arange(self.patch_side_width, self.size_Y - self.patch_side_width, self.patch_width).astype(
            int)
        self.care_ind = np.concatenate([self.care_ind, [self.size_Y - self.patch_side_width - 1]])
        self.margin_sign = np.zeros((self.size_Y + 2 * self.patch_side_width, self.size_Y + 2 * self.patch_side_width))
        self.margin_sign[self.patch_side_width:self.size_Y + self.patch_side_width,
        self.patch_side_width:self.size_Y + self.patch_side_width] = 1
        self.knn_num_elements = self.sqr_patch_num ** 2
        self.knn_ids = np.arange(self.knn_num_elements)
        self.knn_dim = patch_width_ ** 2
        self.hnsw_build()
        return

    def hnsw_build(self):
        neigh_vec_arr = np.zeros(((self.size_Y - 2 * self.patch_side_width) ** 2, self.patch_width ** 2))
        self.original_neigh_vec_arr = np.zeros(((self.size_Y - 2 * self.patch_side_width) ** 2, self.patch_width ** 2))

        for i in range(self.patch_side_width, self.size_Y - self.patch_side_width):
            for j in range(self.patch_side_width, self.size_Y - self.patch_side_width):
                neigh_vec_arr[
                (i - self.patch_side_width) * (self.size_Y - 2 * self.patch_side_width) + j - self.patch_side_width,
                :] = self.init_Y_r[
                     (i - self.patch_side_width):(i + self.patch_side_width + 1),
                     (j - self.patch_side_width):(j + self.patch_side_width + 1)].flatten()
                self.original_neigh_vec_arr[
                (i - self.patch_side_width) * (self.size_Y - 2 * self.patch_side_width) + j - self.patch_side_width,
                :] = self.init_Y[
                     (i - self.patch_side_width):(i + self.patch_side_width + 1),
                     (j - self.patch_side_width):(j + self.patch_side_width + 1)].flatten()
        # Declaring index
        # st = datetime.datetime.now()
        self.p = hnswlib.Index(space='l2', dim=self.knn_dim)  # possible options are l2, cosine or ip
        # Initializing index - the maximum number of elements should be known beforehand
        self.p.init_index(max_elements=self.knn_num_elements, ef_construction=200, M=32)
        # Element insertion (can be called several times):
        self.p.add_items(neigh_vec_arr, self.knn_ids)
        # Controlling the recall by setting ef:
        self.p.set_ef(31)  # ef should always be > k

        return

    def Anomaly_score_map(self, Y_, Y_r, k=30):
        a_score_ = np.zeros((self.size_Y, self.size_Y))
        a_score_pixel = np.zeros((self.size_Y, self.size_Y))
        for ind1 in self.care_ind:
            for ind2 in self.care_ind:
                cur_patch_vec = Y_r[(ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                                (ind2 - self.patch_side_width):(ind2 + self.patch_side_width + 1)].flatten()
                cur_original_patch_vec = Y_[(ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                                         (ind2 - self.patch_side_width):(ind2 + self.patch_side_width + 1)].flatten()
                knn_label, knn_dist = self.p.knn_query(cur_patch_vec, k=k)
                a_score_pixel[(ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                (ind2 - self.patch_side_width):(ind2 + self.patch_side_width + 1)] = np.square(
                    (cur_original_patch_vec - np.average((self.original_neigh_vec_arr[
                                                          knn_label[0, :], :]), axis=0)) / (
                            np.std(self.original_neigh_vec_arr[knn_label[0, :], :], axis=0) + 1e-2)).reshape(
                    self.patch_width, self.patch_width)

        exp_a_score_pixel = 0.0001 * np.ones(
            (self.size_Y + 2 * self.patch_side_width, self.size_Y + 2 * self.patch_side_width))
        exp_a_score_pixel[self.patch_side_width:(self.size_Y + self.patch_side_width),
        self.patch_side_width:(self.size_Y + self.patch_side_width)] = a_score_pixel

        self.a_score_pixel = a_score_pixel
        for i in range(self.size_Y):
            ind1 = i + self.patch_side_width
            for j in range(self.size_Y):
                ind2 = j + self.patch_side_width
                margin_sign_weight_Vec = self.margin_sign[
                                         (ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                                         (ind2 - self.patch_side_width):(
                                                 ind2 + self.patch_side_width + 1)].flatten() * self.weight_Vec / (
                                                 self.margin_sign[
                                                 (ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                                                 (ind2 - self.patch_side_width):(
                                                         ind2 + self.patch_side_width + 1)].flatten() * self.weight_Vec).sum()
                a_score_[i, j] = np.dot(
                    exp_a_score_pixel[(ind1 - self.patch_side_width):(ind1 + self.patch_side_width + 1),
                    (ind2 - self.patch_side_width):(ind2 + self.patch_side_width + 1)].flatten(),
                    margin_sign_weight_Vec)
        return a_score_


def Draw_Figures(image_, Y_Res_, Y_r_, Y_ReCon_, a_score_):
    plt.figure()
    # draw img
    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Test Image')
    plt.imshow(image_, cmap='gray', vmin=0, vmax=1)
    # draw Y_Res
    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Resampled Image')
    plt.imshow(Y_Res_, cmap='gray', vmin=0, vmax=1)
    # draw Y_r
    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Resampled Reference Image')
    plt.imshow(Y_r_, cmap='gray', vmin=0, vmax=1)
    # draw Y_ReCon
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Reference Image')
    plt.imshow(Y_ReCon_, cmap='gray', vmin=0, vmax=1)
    # draw a_score
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(a_score_, cmap='jet', vmin=0, vmax=10)
    cb = plt.colorbar()
    plt.title('Anomaly Score')
    cb.ax.tick_params(labelsize=8)
    plt.show()
    return

def Draw_Pattern(lambda_1, lambda_2):
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('lambda_1')
    plt.plot(lambda_1)

    plt.subplot(2, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('lambda_2')
    plt.plot(lambda_2)

    plt.show()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Anomaly Detection for An Image. Image Size: 256x256')
    parser.add_argument('--image_path', default='test_grid_image.png', help='Path to the image')
    parser.add_argument('--lr_l', type=float, default=0.001,
                        help='Learning rate for adam optimizer for periodic pattern vectors lambda_1 and lambda_2')
    parser.add_argument('--lr_A_E', type=float, default=0.005,
                        help='learning rate for adam optimizer for Anomaly and Noise components')
    parser.add_argument('--rou_l1_lambda', type=float, default=1,
                        help='Tuning parameter to control the sparsity of pattern vectors lambda_1 and lambda_2')
    parser.add_argument('--rou_l1_A', type=float, default=0.02,
                        help='Tuning parameter to control the sparsity of Anomaly components')
    parser.add_argument('--rou_l2_E', type=float, default=0.1,
                        help='Tuning parameter to control the L2 norm of Noise components')
    parser.add_argument('--rou_row_sum', type=float, default=5,
                        help='Tuning parameter to control the sum of each row of W matrix')
    parser.add_argument('--epsilon_lambda', type=float, default=2e-4,
                        help='Tuning parameter to control the sum of each row of W matrix')
    parser.add_argument('--max_iteration', type=int, default=200,
                        help='max iteration time')
    parser.add_argument('--patch_width', type=int, default=11,
                        help='patch width')
    parser.add_argument('--kernel_std', type=float, default=5,
                        help='standard deviation for gaussian kernel average')
    parser.add_argument('--device', default='cuda',
                        help='torch device, cuda(default) or cpu')

    args = parser.parse_args()
    image_path = args.image_path
    lr_l = args.lr_l
    lr_A_E = args.lr_A_E
    rou_l1_lambda = args.rou_l1_lambda
    rou_l1_A = args.rou_l1_A
    rou_l2_E = args.rou_l2_E
    rou_row_sum = args.rou_row_sum
    dtype = torch.double
    patch_width = args.patch_width
    device = torch.device(args.device)
    max_iteration = args.max_iteration
    kernel_std = args.kernel_std
    epsilon_lambda = args.epsilon_lambda

    img = cv2.imread(image_path, 0)
    img = img / 255
    label = None
    # resample image
    RI = Resample_Image(img)
    RI.Resample(img)
    # periodic pattern extract
    Y = RI.Y_ReS
    size_y1 = Y.shape[0]
    size_y2 = Y.shape[1]
    PE = Pattern_Extract(Y=Y, lr_l=lr_l, lr_A_E=lr_A_E,
                          rou_l1_lambda=rou_l1_lambda,
                          rou_l1_A=rou_l1_A,
                          rou_l2_E=rou_l2_E,
                          rou_row_sum=rou_row_sum,
                          dtype=dtype, device=device, max_iteration=max_iteration, epsilon_lambda=epsilon_lambda)
    PE.opti_iteration()
    # image reconstruction
    RI.Reconstruction_params(PE.lambda1, PE.lambda2, PE.Y_A_E)
    RI.Reconstruction()
    # anomaly scoring
    ASE = A_Score_Estimate(RI.Y_recon, img, patch_width_=patch_width, kernel_std_=kernel_std)
    a_score = ASE.Anomaly_score_map(img, RI.Y_recon)

    Draw_Figures(Y, PE.Y_A_E, np.abs(PE.A),  np.abs(PE.E), a_score)
    Draw_Figures(Y, Y[:,0].reshape(-1,1), Y[0,:].reshape(1,-1), np.abs(PE.E), a_score)

    Draw_Pattern(PE.lambda1, PE.lambda2)


# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.imshow(Y[0,:].reshape(1,-1),cmap='gray', vmin=0, vmax=1)
# plt.show()


