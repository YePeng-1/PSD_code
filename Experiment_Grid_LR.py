# This code integrated the anomaly scoring module into the P-NLR method
import torch
import Anomaly_Detection
from os import listdir, path
import cv2
import Pattern_Learning as PELr
import datetime
from read_results import *

def Draw_Figures_LR(images_list_, label_list_, Y_ReCon_list_, a_score_list_, a_score_threshold_,
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
        plt.subplot(num, 5, ind * 5 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_list_[ind_arr_[ind]], cmap='gray')
        # draw Y_ReCon
        plt.subplot(num, 5, ind * 5 + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Y_ReCon_list_[ind_arr_[ind]], cmap='gray')
        # draw a_score
        plt.subplot(num, 5, ind * 5 + 3)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(a_score, cmap='jet', vmin=0, vmax=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=8)
        # draw anomaly
        plt.subplot(num, 5, ind * 5 + 4)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(anomaly, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # draw true label
        plt.subplot(num, 5, ind * 5 + 5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(label_list_[ind_arr_[ind]], cmap='gray')
    plt.show()
    return


def paths_dataset_files(dataset_dir_input,result_dir_input):
    image_filepaths_ = []
    gt_filepaths_ = []
    results_filepaths = []

    images_dir = path.join(dataset_dir_input, 'test')
    gt_dir = path.join(dataset_dir_input, 'ground_truth')

    for subdir in listdir(str(images_dir)):
        if not subdir.replace('_', '').isalpha():
            continue
        test_images = [path.splitext(file)[0]
                       for file
                       in listdir(path.join(images_dir, subdir))
                       if path.splitext(file)[1] == '.png']
        # If subdir is not 'good', derive corresponding GT names.
        if subdir != 'good':
            gt_filepaths_.extend(
                [path.join(gt_dir, subdir, file + '_mask.png')
                 for file in test_images])
            results_filepaths.extend(
                [path.join(result_dir_input, subdir, file + '.png')
                 for file in test_images])
            image_filepaths_.extend(
                [path.join(images_dir, subdir, file + '.png')
                 for file in test_images])

    print(f"Parsed {len(gt_filepaths_)} ground truth image files.")
    return image_filepaths_, gt_filepaths_,results_filepaths


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Anomaly Detection Experiment for the Grid Dataset from MVTec AD dataset')
    parser.add_argument('--dataset_path', help='Path to the Grid dataset')
    parser.add_argument('--result_output_path', default='Grid_Result_LR_AS.pkl', help='Path to the save results, .pkl')
    parser.add_argument('--lr_l', type=float, default=0.001,
                        help='Learning rate for adam optimizer for periodic pattern vectors lambda_1 and lambda_2')
    parser.add_argument('--lr_A_E', type=float, default=0.005,
                        help='learning rate for adam optimizer for Anomaly and Noise components')
    parser.add_argument('--rou_l1_lambda', type=float, default=1,
                        help='Tuning parameter to control the sparsity of pattern vectors lambda_1 and lambda_2')
    parser.add_argument('--rou_l1_A', type=float, default=0.02,
                        help='Tuning parameter to control the sparsity of Anomaly components')
    parser.add_argument('--rou_l2_E', type=float, default=1,
                        help='Tuning parameter to control the L2 norm of Noise components')
    parser.add_argument('--rou_row_sum', type=float, default=10,
                        help='Tuning parameter to control the sum of each row of W matrix')
    parser.add_argument('--epsilon_lambda', type=float, default=3e-4,
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

    # Grid parameters
    result_output_path = args.result_output_path

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
    results_dir = 'E:\LR_Grid'
    dataset_dir = 'E:\Grid_Data_Set\grid'
    # dataset_dir = args.dataset_path
    image_filepaths, gt_filepaths,result_filepaths = paths_dataset_files(dataset_dir,results_dir)

    # output data
    # dataset
    images_list = list()  # image list
    label_list = list()  # label list

    # anomaly score data
    ascore_list = list()
    Y_ReCon_list = list()

    # Read Dataset, Convert to 256 x 256
    pixel_0 = np.arange(256) * 4
    for i in range(len(image_filepaths)):
        image_path = image_filepaths[i]
        gt_path = gt_filepaths[i]
        result_path = result_filepaths[i]
        src = cv2.imread(image_path, 0)
        img = src[pixel_0, :]
        img = img[:, pixel_0]
        img = img / 255
        label = None
        if gt_path is not None:
            label = cv2.imread(gt_path, 0)
            label = label[pixel_0, :]
            label = label[:, pixel_0]
            label = label / 255
        else:
            label = np.zeros(img.shape)
        images_list.append(img)
        label_list.append(label)
        res_img = cv2.imread(result_path, 0)
        res_img = res_img / 255
        Y_ReCon_list.append(res_img)
    # images number
    N = len(images_list)

    st = datetime.datetime.now()
    # anomaly detection
    for i in range(N):
        sti = datetime.datetime.now()
        img = images_list[i]
        # resample image

        ASE = Anomaly_Detection.A_Score_Estimate(Y_ReCon_list[i], img, patch_width_=patch_width, kernel_std_=kernel_std)
        a_score = ASE.Anomaly_score_map(img, Y_ReCon_list[i])
        ascore_list.append(a_score)
        eti = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(eti - sti).seconds, 's', (eti - sti).microseconds, 'us')

    et = datetime.datetime.now()
    print('detection time', (et - st).seconds, 's', (et - st).microseconds, 'us')



    # print results
    a_score_summary = np.asarray(ascore_list).flatten()
    label_summary = np.asarray(label_list).flatten()

    auc_value, fpr, tpr, thresholds = Cal_AUC(a_score_summary, label_summary)
    fprs, pros = compute_pro(ascore_list, label_list)

    au_pr = average_precision_score(label_summary, a_score_summary)
    precision, recall, pr_thresholds = precision_recall_curve(label_summary, a_score_summary)
    ground_truth_labels = np.ones((len(gt_filepaths))).astype(int)

    integration_limit = 0.3
    au_ROC_limit = trapezoid(fpr, tpr, x_max=integration_limit)
    au_ROC_limit /= integration_limit
    print(f"AU-ROC (FPR limit: {integration_limit}): {au_ROC_limit}")

    au_pro_limit = trapezoid(fprs, pros, x_max=integration_limit)
    au_pro_limit /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro_limit}")
    print(f"AU-PR : {au_pr}")
    print(f"AU-ROC(Pixel-level) : {auc_value}")

    for i in range(len(gt_filepaths)):
        if gt_filepaths[i] is None:
            ground_truth_labels[i] = 0
    f = lambda x: np.percentile(x, 99.99)
    class_auc = compute_classification_roc(ascore_list, f, ground_truth_labels)

    print(f"AU-ROC(Image-level) : : {class_auc[0]}")

    # indices compare with untrained methods
    defect_type = ["bent", "broken", "glue", "metal_contamination", "thread"]
    defect_img_list = list([list(), list(), list(), list(), list()])

    N = len(images_list)
    for i in range(N):
        if image_filepaths[i].find(defect_type[0]) > -1:
            defect_img_list[0].append(i)
            continue
        if image_filepaths[i].find(defect_type[1]) > -1:
            defect_img_list[1].append(i)
            continue
        if image_filepaths[i].find(defect_type[2]) > -1:
            defect_img_list[2].append(i)
            continue
        if image_filepaths[i].find(defect_type[3]) > -1:
            defect_img_list[3].append(i)
            continue
        if image_filepaths[i].find(defect_type[4]) > -1:
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
            FOR_list[i][j] = fp / (fp + tp + 1e-9)
            FNR_list[i][j] = fn / (fn + tp + 1e-9)
            BA_list[i][j] = tn / (2 * (tn + fp) + 1e-9) + tp / (2 * tp + fn + 1e-9)
            DICE_list[i][j] = 2 * tp / (2 * tp + fn + fp + 1e-9)
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
    num_defect = num_bent + num_broken + num_glue + num_mc + num_thread

    for i in range(12):
        Draw_Figures_LR(images_list, label_list, Y_ReCon_list, ascore_list, 2,
                     np.array(range(i*5,(i+1)*5)))

    ind_arr = np.asarray([1, 13, 25, 36, 48]).astype(int)
    Draw_Figures_LR(images_list, label_list, Y_ReCon_list, ascore_list, 2,
                    ind_arr)


    # # save results
    data = {'image_filepaths': image_filepaths,
            'gt_filepaths': gt_filepaths,
            'images_list': images_list,
            'label_list': label_list,
            'Y_ReCon_list': Y_ReCon_list,
            'ascore_list': ascore_list}
    output = open(result_output_path, 'wb')
    pickle.dump(data, output)
    output.close()


    image_filepaths = data['image_filepaths']
    gt_filepaths = data['gt_filepaths']
    images_list = data['images_list']
    Y_ReCon_list = data['Y_ReCon_list']
    label_list = data['label_list']
    ascore_list = data['ascore_list']





