import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.measurements import label
from generic_util import trapezoid
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import argparse
import scipy.io as scio


def Draw_Lambdas(lambda1_list,lambda2_list):
    num = len(lambda1_list)
    plt.figure()
    for i in range(num):

        plt.subplot(5, 3, i%3+1+(i//3)*6)
        plt.grid(False)
        # plt.imshow(images_list[i])
        plt.plot(lambda1_list[i])
        # plt.xlabel(r'$ \lambda_1 $',fontsize=6)


        plt.subplot(5, 3, i%3+4+(i//3)*6)
        plt.grid(False)
        plt.plot(lambda2_list[i])
        # plt.xlabel(r'$ \lambda_2 $',fontsize=6)

    plt.show()
    return

def ShowcBoxPlot(data1,data2, data3, data4, labels):
    plt.figure()
    flierprops = dict(marker='o', markersize=16)
    plt.boxplot([data1,data2, data3, data4], labels=labels, flierprops=flierprops)
    plt.show()

def Cal_AUC(a_score, label_):
    y_true = np.zeros((label_.shape[0])).astype(int)
    cond = np.where(label_ > 0.5)
    y_true[cond[0]] = 1
    fpr_, tpr_, thresholds_ = roc_curve(y_true, a_score, drop_intermediate=False)
    auc_value_ = auc(fpr_, tpr_)
    return auc_value_, fpr_, tpr_, thresholds_


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


def Draw_Figures(images_list_, label_list_, Y_Res_list_, Y_r_list_, Y_ReCon_list_, a_score_list_, a_score_threshold_,
                 ind_arr_):
    plt.figure()
    num = ind_arr_.shape[0]
    for ind in range(num):
        a_score = a_score_list_[ind_arr_[ind]]
        # cond0 = np.where(a_score>1000)
        # a_score[cond0[0],cond0[1]] = 1000
        anomaly = np.zeros((a_score.shape[0], a_score.shape[1]))
        cond = np.where(a_score > a_score_threshold_)
        anomaly[cond[0], cond[1]] = 1
        # draw img
        plt.subplot(num, 7, ind * 7 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_list_[ind_arr_[ind]], cmap='gray')
        # draw Y_Res
        plt.subplot(num, 7, ind * 7 + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Y_Res_list_[ind_arr_[ind]], cmap='gray')
        # draw Y_r
        plt.subplot(num, 7, ind * 7 + 3)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Y_Res_list_[ind_arr_[ind]]-Y_r_list_[ind_arr_[ind]], cmap='gray')
        # draw Y_ReCon
        plt.subplot(num, 7, ind * 7 + 4)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Y_ReCon_list_[ind_arr_[ind]], cmap='gray')
        # draw a_score
        plt.subplot(num, 7, ind * 7 + 5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(a_score, cmap='jet', vmin=0, vmax=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=8)
        # draw anomaly
        plt.subplot(num, 7, ind * 7 + 6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        plt.imshow(anomaly, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # draw true label
        plt.subplot(num, 7, ind * 7 + 7)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(label_list_[ind_arr_[ind]], cmap='gray')
    plt.show()
    return


def compute_classification_roc(
        anomaly_maps,
        scoring_function,
        ground_truth_labels_,
        draw_fig = False):
    """Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain
          a real-valued anomaly score at each pixel.
        scoring_function: Function that turns anomaly maps into a single
          real valued anomaly score.

        ground_truth_labels_: List of integers that indicate the ground truth
          class for each input image. 0 corresponds to an anomaly-free sample
          while a value != 0 indicates an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    assert len(anomaly_maps) == len(ground_truth_labels_)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = list(map(scoring_function, anomaly_maps))
    class_auc_ = Cal_AUC(anomaly_scores, ground_truth_labels_)
    title_ = 'Classification_Roc, au_ROC = ' + str(class_auc_[0])
    if draw_fig:
        Draw_Roc(anomaly_scores, ground_truth_labels_, title_)

    return class_auc_


def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    print("Compute PRO curve...")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs_ = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros_ = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs_ = fprs_[keep_mask]
    pros_ = pros_[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs_, a_min=None, a_max=1., out=fprs_)
    np.clip(pros_, a_min=None, a_max=1., out=pros_)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs_, one)), np.concatenate((zero, pros_, one))


def compute_confusion_threshold(ascoreMap, gt, threshold):
    anomaly = np.zeros((ascoreMap.shape[0], ascoreMap.shape[1])).astype(int)
    cond = np.where(ascoreMap > threshold)
    anomaly[cond[0], cond[1]] = 1
    tn_, fp_, fn_, tp_ = confusion_matrix(np.around(gt.flatten()), anomaly.flatten()).ravel()
    return tn_, fp_, fn_, tp_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Read Saved Results of Experiment for the Grid Dataset from MVTec AD dataset')
    parser.add_argument('--result_path', default='Grid_Result_final_0729.pkl', help='Path to the result file, ,pkl')
    args = parser.parse_args()

    result_file_path = args.result_path
    # Read result file
    pkl_file = open(result_file_path, 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()
    images_list = data1['images_list']
    label_list = data1['label_list']
    Y_ReS_list = data1['Y_ReS_list']
    Y_r_exp_list = data1['Y_r_exp_list']
    Y_ReCon_list = data1['Y_ReCon_list']
    ascore_list = data1['ascore_list']
    images_path = data1['image_filepaths']
    Y_r_list = data1['Y_r_list']
    gt_filepaths = data1['gt_filepaths']
    A_list = data1['A_list']
    lambda1_list = data1['lambda1_list']
    lambda2_list = data1['lambda2_list']

    a_score_summary = np.asarray(ascore_list).flatten()
    label_summary = np.asarray(label_list).flatten()

    # pixel-level classification result
    # AU-ROC
    auc_value, fpr, tpr, thresholds = Cal_AUC(a_score_summary, label_summary)
    title = 'Pixel-level ROC, au_ROC = ' + str(auc_value)
    Draw_Roc(a_score_summary, label_summary, title)
    au_pr = average_precision_score(label_summary, a_score_summary)
    precision, recall, pr_thresholds = precision_recall_curve(label_summary, a_score_summary)
    plt.figure()
    plt.title('PR-Curve, au_PR =' + str(au_pr))
    plt.plot(precision, recall, label="Anomaly PR")
    plt.axis("square")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    # AU-PRO
    fprs, pros = compute_pro(ascore_list, label_list)
    au_pro = trapezoid(fprs, pros, x_max=1.0)
    plt.figure()
    plt.title('PRO-Curve, au_PRO =' + str(au_pro))
    plt.ylabel('PRO Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(fprs, pros, label="Anomaly PRO")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AU_PR = 0.5)")
    plt.show()
    # Limit AU-ROC & AU-PRO
    integration_limit = 0.3
    au_ROC_limit = trapezoid(fpr, tpr, x_max=integration_limit)
    au_ROC_limit /= integration_limit
    print(f"AU-ROC (FPR limit: {integration_limit}): {au_ROC_limit}")
    au_pro_limit = trapezoid(fprs, pros, x_max=integration_limit)
    au_pro_limit /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro_limit}")

    print(f"AU-PR : {au_pr}")
    print(f"AU-ROC(Pixel-level) : {auc_value}")

    # image-level classification result
    auc_list = list()
    ground_truth_labels = np.ones((len(gt_filepaths))).astype(int)
    for i in range(len(gt_filepaths)):
        if gt_filepaths[i] is None:
            ground_truth_labels[i] = 0
    for i in range(9900,10001):
        f = lambda x: np.percentile(x, i*0.01)
        class_auc = compute_classification_roc(ascore_list, f, ground_truth_labels)
        auc_list.append(class_auc[0])

    plt.figure()
    plt.plot(np.array(auc_list))
    plt.show()
    best_auc = np.max(np.array(auc_list))
    best_percentile = (np.argmax(np.array(auc_list))+9900)*0.01
    print(f"AU-ROC(Image-level) : : {best_auc}")

    best_auc = np.max(np.array(auc_list))
    best_percentile = (np.argmax(np.array(auc_list)) + 9900) * 0.01
    print(f"AU-ROC(Image-level) : : {best_auc}")

    # Draw Figures
    ind_arr = np.asarray([37, 1, 13, 25, 57, 69]).astype(int)
    Draw_Figures(images_list, label_list, Y_ReS_list, A_list, Y_ReCon_list, ascore_list, 2.5,
                 ind_arr)

    for i in range(12):
        Draw_Figures(images_list, label_list, Y_ReS_list, A_list, Y_ReCon_list, ascore_list, 2,
                     np.array(range(i*7,(i+1)*7)))

    # for i in range(13):
    #     Draw_Lambdas(lambda1_list[i*6:(i+1)*6],lambda2_list[i*6:(i+1)*6])

    # indices compare with untrained methods
    defect_type = ["bent", "broken", "glue", "metal_contamination", "thread"]
    defect_img_list = list([list(), list(), list(), list(), list()])

    N = len(images_list)
    for i in range(N):
        if images_path[i].find(defect_type[0]) > -1:
            defect_img_list[0].append(i)
            continue
        if images_path[i].find(defect_type[1]) > -1:
            defect_img_list[1].append(i)
            continue
        if images_path[i].find(defect_type[2]) > -1:
            defect_img_list[2].append(i)
            continue
        if images_path[i].find(defect_type[3]) > -1:
            defect_img_list[3].append(i)
            continue
        if images_path[i].find(defect_type[4]) > -1:
            defect_img_list[4].append(i)
            continue

    ACC_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                     np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                     np.zeros((len(defect_img_list[4], )))])
    TPR_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                     np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                     np.zeros((len(defect_img_list[4], )))])
    FPR_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                     np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                     np.zeros((len(defect_img_list[4], )))])
    PPV_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                     np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                     np.zeros((len(defect_img_list[4], )))])
    NPV_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                     np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                     np.zeros((len(defect_img_list[4], )))])
    F_measure_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                           np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                           np.zeros((len(defect_img_list[4], )))])
    FOR_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                           np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                           np.zeros((len(defect_img_list[4], )))])
    FNR_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                           np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                           np.zeros((len(defect_img_list[4], )))])
    BA_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                           np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                           np.zeros((len(defect_img_list[4], )))])
    DICE_list = list([np.zeros((len(defect_img_list[0], ))), np.zeros((len(defect_img_list[1], ))),
                           np.zeros((len(defect_img_list[2], ))), np.zeros((len(defect_img_list[3], ))),
                           np.zeros((len(defect_img_list[4], )))])

    indices_type = np.zeros((len(defect_type), 10))
    for i in range(len(defect_type)):
        for j in range(len(defect_img_list[i])):
            tn, fp, fn, tp = compute_confusion_threshold(ascore_list[defect_img_list[i][j]],
                                                         label_list[defect_img_list[i][j]], 2.0)
            ACC_list[i][j] = (tp + tn) / (tp + fp + fn + tn)
            TPR_list[i][j] = tp / (tp + fn + 1e-9)
            FPR_list[i][j] = fp / (fp + tn + 1e-9)
            PPV_list[i][j] = tp / (tp + fp + 1e-9)
            NPV_list[i][j] = tn / (tn + fn + 1e-9)
            F_measure_list[i][j] = 2 * PPV_list[i][j] * TPR_list[i][j] / (PPV_list[i][j] + TPR_list[i][j] + 1e-9)
            FOR_list[i][j] = fp/(fp+tp+ 1e-9)
            FNR_list[i][j] = fn/(fn+tp+ 1e-9)
            BA_list[i][j] = tn/(2*(tn+fp)+ 1e-9)+tp/(2*tp+fn+ 1e-9)
            DICE_list[i][j] = 2*tp/(2*tp+fn+fp+1e-9)
        indices_type[i, 0] = np.mean(ACC_list[i])
        indices_type[i, 1] = np.mean(TPR_list[i])
        indices_type[i, 2] = np.mean(FPR_list[i])
        indices_type[i, 3] = np.mean(PPV_list[i])
        indices_type[i, 4] = np.mean(NPV_list[i])
        indices_type[i, 5] = np.mean(F_measure_list[i])
        indices_type[i, 6] = np.mean(FOR_list[i])
        indices_type[i, 7] = np.mean(FNR_list[i])
        indices_type[i, 8] = np.mean(BA_list[i])
        indices_type[i, 9] = np.mean(DICE_list[i])
    indices_all = np.mean(indices_type, axis=0)

    num_bent = len(defect_img_list[0])
    num_broken = len(defect_img_list[1])
    num_glue = len(defect_img_list[2])
    num_mc = len(defect_img_list[3])
    num_thread = len(defect_img_list[4])
    num_defect = num_bent+num_broken+num_glue+num_mc+num_thread
    # Comparison results

    dataFile = 'Comparison_Results/PLSR.mat'
    PLSR_data = scio.loadmat(dataFile)
    PLSR_indecs_list = PLSR_data['indecs_list']
    PLSR_FOR_array = np.zeros((num_defect,))
    PLSR_FNR_array = np.zeros((num_defect,))
    PLSR_BA_array = np.zeros((num_defect,))
    PLSR_DICE_array = np.zeros((num_defect,))
    ind = 0
    for i in range(5):
        PLSR_FOR_array[ind:ind+len(defect_img_list[i])] = np.asarray(PLSR_indecs_list[i][0])[:,6]
        PLSR_FNR_array[ind:ind + len(defect_img_list[i])] = np.asarray(PLSR_indecs_list[i][0])[:, 7]
        PLSR_BA_array[ind:ind + len(defect_img_list[i])] = np.asarray(PLSR_indecs_list[i][0])[:, 8]
        PLSR_DICE_array[ind:ind + len(defect_img_list[i])] = np.asarray(PLSR_indecs_list[i][0])[:, 9]
        ind+=len(defect_img_list[i])

    dataFile = 'Comparison_Results/Spec_Res.mat'
    Spec_Res_data = scio.loadmat(dataFile)
    Spec_Res_indecs_list = Spec_Res_data['indecs_list']
    Spec_Res_FOR_array = np.zeros((num_defect,))
    Spec_Res_FNR_array = np.zeros((num_defect,))
    Spec_Res_BA_array = np.zeros((num_defect,))
    Spec_Res_DICE_array = np.zeros((num_defect,))
    ind = 0
    for i in range(5):
        Spec_Res_FOR_array[ind:ind+len(defect_img_list[i])] = np.asarray(Spec_Res_indecs_list[i][0])[:,6]
        Spec_Res_FNR_array[ind:ind + len(defect_img_list[i])] = np.asarray(Spec_Res_indecs_list[i][0])[:, 7]
        Spec_Res_BA_array[ind:ind + len(defect_img_list[i])] = np.asarray(Spec_Res_indecs_list[i][0])[:, 8]
        Spec_Res_DICE_array[ind:ind + len(defect_img_list[i])] = np.asarray(Spec_Res_indecs_list[i][0])[:, 9]
        ind+=len(defect_img_list[i])

    dataFile = 'Comparison_Results/PNLR.mat'
    PNLR_data = scio.loadmat(dataFile)
    PNLR_indecs_list = PNLR_data['indecs_list']
    PNLR_FOR_array = np.zeros((num_defect,))
    PNLR_FNR_array = np.zeros((num_defect,))
    PNLR_BA_array = np.zeros((num_defect,))
    PNLR_DICE_array = np.zeros((num_defect,))
    ind = 0
    for i in range(5):
        PNLR_FOR_array[ind:ind+len(defect_img_list[i])] = np.asarray(PNLR_indecs_list[i][0])[:,6]
        PNLR_FNR_array[ind:ind + len(defect_img_list[i])] = np.asarray(PNLR_indecs_list[i][0])[:, 7]
        PNLR_BA_array[ind:ind + len(defect_img_list[i])] = np.asarray(PNLR_indecs_list[i][0])[:, 8]
        PNLR_DICE_array[ind:ind + len(defect_img_list[i])] = np.asarray(PNLR_indecs_list[i][0])[:, 9]
        ind+=len(defect_img_list[i])

    PSD_FOR_array = np.zeros((num_defect,))
    PSD_FNR_array = np.zeros((num_defect,))
    PSD_BA_array = np.zeros((num_defect,))
    PSD_DICE_array = np.zeros((num_defect,))
    ind = 0
    for i in range(5):
        PSD_FOR_array[ind:ind+len(defect_img_list[i])] = FOR_list[i]
        PSD_FNR_array[ind:ind + len(defect_img_list[i])] = FNR_list[i]
        PSD_BA_array[ind:ind + len(defect_img_list[i])] = BA_list[i]
        PSD_DICE_array[ind:ind + len(defect_img_list[i])] = DICE_list[i]
        ind+=len(defect_img_list[i])

    labels = 'PG-LSR', 'Spec-Res', 'P-NLR', 'Ours'
    ShowcBoxPlot(PLSR_FOR_array,Spec_Res_FOR_array, PNLR_FOR_array, PSD_FOR_array, labels)
    ShowcBoxPlot(PLSR_FNR_array, Spec_Res_FNR_array, PNLR_FNR_array, PSD_FNR_array, labels)
    ShowcBoxPlot(PLSR_BA_array, Spec_Res_BA_array, PNLR_BA_array, PSD_BA_array, labels)
    ShowcBoxPlot(PLSR_DICE_array, Spec_Res_DICE_array, PNLR_DICE_array, PSD_DICE_array, labels)


