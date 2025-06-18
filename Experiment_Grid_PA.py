import torch
import Anomaly_Detection
from os import listdir, path
import cv2
import Pattern_Learning_PA as PELr
import datetime
from read_results import *


def paths_dataset_files(dataset_dir_input):
    image_filepaths_ = []
    gt_filepaths_ = []

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
        else:
            # No ground truth maps exist for anomaly-free images.
            gt_filepaths_.extend([None] * len(test_images))
        image_filepaths_.extend(
            [path.join(images_dir, subdir, file + '.png')
             for file in test_images])

    print(f"Parsed {len(gt_filepaths_)} ground truth image files.")
    return image_filepaths_, gt_filepaths_


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Anomaly Detection Experiment for the Grid Dataset from MVTec AD dataset')
    parser.add_argument('--dataset_path', help='Path to the Grid dataset')
    parser.add_argument('--result_output_path', default='Grid_Result_PA.pkl', help='Path to the save results, .pkl')
    parser.add_argument('--lr_l', type=float, default=0.01,
                        help='Learning rate for adam optimizer for periodic pattern vectors lambda_1 and lambda_2')
    parser.add_argument('--lr_A_E', type=float, default=0.005,
                        help='learning rate for adam optimizer for Anomaly and Noise components')
    parser.add_argument('--rou_l1_lambda', type=float, default=0,
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
    dataset_dir = 'E:\Grid_Data_Set\grid'
    # dataset_dir = args.dataset_path
    image_filepaths, gt_filepaths = paths_dataset_files(dataset_dir)

    # output data
    # dataset
    images_list = list()  # image list
    label_list = list()  # label list

    # anomaly score data
    ascore_list = list()
    # observe data
    Y_ReS_list = list()
    Y_r_list = list()
    Y_r_exp_list = list()
    Y_ReCon_list = list()
    A_list = list()
    E_list = list()
    lambda1_list = list()
    lambda2_list = list()

    # Read Dataset, Convert to 256 x 256
    pixel_0 = np.arange(256) * 4
    for i in range(len(image_filepaths)):
        image_path = image_filepaths[i]
        gt_path = gt_filepaths[i]
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
    # images number
    N = len(images_list)

    st = datetime.datetime.now()
    # anomaly detection
    for i in range(N):
        sti = datetime.datetime.now()
        img = images_list[i]
        # resample image
        RI = Anomaly_Detection.Resample_Image(img)
        RI.Resample(img)
        Y_ReS_list.append(RI.Y_ReS)
        # periodic pattern extract
        Y = RI.Y_ReS
        size_y1 = Y.shape[0]
        size_y2 = Y.shape[1]
        PE = PELr.Pattern_Extract(Y=Y, lr_l=lr_l, lr_A_E=lr_A_E,
                                  rou_l1_lambda=rou_l1_lambda,
                                  rou_l1_A=rou_l1_A,
                                  rou_l2_E=rou_l2_E,
                                  rou_row_sum=rou_row_sum,
                                  dtype=dtype, device=device, max_iteration=max_iteration, epsilon_lambda=epsilon_lambda)
        PE.opti_iteration()
        lambda1_list.append(PE.lambda1)
        lambda2_list.append(PE.lambda2)
        Y_r_list.append(PE.Y_r)
        A_list.append(PE.A)
        E_list.append(Y-PE.E)
        # image reconstruction
        RI.Reconstruction_params(PE.lambda1, PE.lambda2, PE.Y_A_E)
        RI.Reconstruction()
        Y_r_exp_list.append(RI.Y_r_exp)
        Y_ReCon_list.append(RI.Y_recon)
        ASE = Anomaly_Detection.A_Score_Estimate(RI.Y_recon, img, patch_width_=patch_width, kernel_std_=kernel_std)
        a_score = ASE.Anomaly_score_map(img, RI.Y_recon)
        ascore_list.append(a_score)
        eti = datetime.datetime.now()
        print('No.', i ,' image complete, detection time: ',(eti - sti).seconds, 's', (eti - sti).microseconds, 'us')

    et = datetime.datetime.now()
    print('detection time', (et - st).seconds, 's', (et - st).microseconds, 'us')

    # save results
    data = {'image_filepaths': image_filepaths,
            'gt_filepaths': gt_filepaths,
            'images_list': images_list,
            'label_list': label_list,
            'Y_ReS_list': Y_ReS_list,
            'Y_r_list': Y_r_list,
            'A_list': A_list,
            'E_list':E_list,
            'Y_r_exp_list': Y_r_exp_list,
            'Y_ReCon_list': Y_ReCon_list,
            'ascore_list': ascore_list,
            'lambda1_list':lambda1_list,
            'lambda2_list':lambda2_list}
    output = open(result_output_path, 'wb')
    pickle.dump(data, output)
    output.close()

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




