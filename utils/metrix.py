from tqdm import tqdm
import numpy as np
import os
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_list(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

                subname = path.split('/')
                images.append(os.path.join(subname[-2], subname[-1]))
    return images

def jac(predict, target):

    predict = np.atleast_1d(predict.astype(np.bool_))
    target = np.atleast_1d(target.astype(np.bool_))

    intersection = np.count_nonzero(predict & target)
    union = np.count_nonzero(predict | target)

    jac = float(intersection) / float(union)

    return jac



def computeBER_mth(gt_path, pred_path):
    print(gt_path, pred_path)

    gt_list = get_image_list(gt_path)
    nim = len(gt_list)

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
        im = gt_list[i]
        GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
        posPoints = GTim > 0.5
        negPoints = GTim <= 0.5
        countPos = np.sum(posPoints.astype('uint8'))
        countNeg = np.sum(negPoints.astype('uint8'))
        sz = GTim.shape
        GTim = GTim > 0.5

        Predim = np.asarray(
            Image.open(os.path.join(pred_path, im)).convert('L').resize((sz[1], sz[0]),
                                                                                                Image.NEAREST))

        # BER
        tp = (Predim > 127) & posPoints
        tn = (Predim <= 127) & negPoints
        countTP = np.sum(tp)
        countTN = np.sum(tn)
        stats[i, :] = [countTP, countTN, countPos, countNeg]

        # IoU
        pred_iou = (Predim > 127)
        stats_jaccard[i] = jac(pred_iou, posPoints)

        # MAE
        pred_mae = (Predim > 12)
        mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))
        stats_mae[i] = mae_value

        # Precision and Recall for FMeasure
        eps = 1e-4
        for jj in range(0, 256):
            real_tp = np.sum((Predim > jj) & posPoints)
            real_t = countPos
            real_p = np.sum((Predim > jj).astype('uint8'))

            precision_value = (real_tp + eps) / (real_p + eps)
            recall_value = (real_tp + eps) / (real_t + eps)
            stats_fscore[jj, i, :] = [precision_value, recall_value]

    # Print BER
    posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
    negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
    pA = 100 - 100 * posAcc
    nA = 100 - 100 * negAcc
    BER = 0.5 * (2 - posAcc - negAcc) * 100
    print('BER, S-BER, N-BER:')
    print(BER, pA, nA)

    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)

    # Print Fmeasure
    precision_threshold_list = np.mean(stats_fscore[:, :, 0], axis=1).tolist()
    recall_threshold_list = np.mean(stats_fscore[:, :, 1], axis=1).tolist()
    fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
    print('Fmeasure:', fmeasure)

    return {"BER": BER, "S-BER": pA, "N-BER": nA, "IoU": jaccard_value, "MAE": mean_mae_value, "Fmeasure": fmeasure}


def computeIoU(gt_path, pred_path):
    print(gt_path, pred_path)

    gt_list = get_image_list(gt_path)
    nim = len(gt_list)

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
        im = gt_list[i]
        GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
        posPoints = GTim > 0.5
        negPoints = GTim <= 0.5
        sz = GTim.shape
        GTim = GTim > 0.5

        Predim = np.asarray(
            Image.open(os.path.join(pred_path, im)).convert('L').resize((sz[1], sz[0]),Image.NEAREST))

        # BER

        # IoU
        pred_iou = (Predim > 102)
        stats_jaccard[i] = jac(pred_iou, posPoints)

        # MAE
        pred_mae = (Predim > 12)
        mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))
        stats_mae[i] = mae_value


    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)



    return {"IoU": jaccard_value, "MAE": mean_mae_value}

def computeIOU_MAE_BER(gt_path, pred_path):
    print(gt_path, pred_path)

    gt_list = get_image_list(gt_path)
    nim = len(gt_list)

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
        im = gt_list[i]
        GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
        posPoints = GTim > 0.5
        negPoints = GTim <= 0.5
        countPos = np.sum(posPoints.astype('uint8'))
        countNeg = np.sum(negPoints.astype('uint8'))
        sz = GTim.shape
        GTim = GTim > 0.5

        Predim = np.asarray(
            Image.open(os.path.join(pred_path, im)).convert('L').resize((sz[1], sz[0]),
                                                                                                Image.NEAREST))

        # BER
        tp = (Predim > 102) & posPoints
        tn = (Predim <= 102) & negPoints
        countTP = np.sum(tp)
        countTN = np.sum(tn)
        stats[i, :] = [countTP, countTN, countPos, countNeg]

        # IoU
        pred_iou = (Predim > 102)
        stats_jaccard[i] = jac(pred_iou, posPoints)

        # MAE
        pred_mae = (Predim > 12)
        mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))
        stats_mae[i] = mae_value

        # Precision and Recall for FMeasure


    # Print BER
    posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
    negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
    pA = 100 - 100 * posAcc
    nA = 100 - 100 * negAcc
    BER = 0.5 * (2 - posAcc - negAcc) * 100
    print('BER, S-BER, N-BER:')
    print(BER, pA, nA)

    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)

    return { "BER": BER, "S-BER": pA, "N-BER": nA,"IoU": jaccard_value, "MAE": mean_mae_value}

from joblib import Parallel, delayed

def compute_metrics(im, gt_path, pred_path):
    GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
    posPoints = GTim > 0.5
    negPoints = GTim <= 0.5
    countPos = np.sum(posPoints.astype('uint8'))
    countNeg = np.sum(negPoints.astype('uint8'))
    sz = GTim.shape
    GTim = GTim > 0.5

    Predim = np.asarray(
        Image.open(os.path.join(pred_path, im)).convert('L').resize((sz[1], sz[0]), Image.Resampling.NEAREST))

    # BER
    tp = (Predim > 102) & posPoints
    tn = (Predim <= 102) & negPoints
    countTP = np.sum(tp)
    countTN = np.sum(tn)
    stats = [countTP, countTN, countPos, countNeg]

    # IoU
    pred_iou = (Predim > 102)
    jaccard_value = jac(pred_iou, posPoints)

    # MAE
    pred_mae = (Predim > 12)
    mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))

    # Precision and Recall for FMeasure
    eps = 1e-4
    stats_fscore = []
    for jj in range(0, 256):
        real_tp = np.sum((Predim > jj) & posPoints)
        real_t = countPos
        real_p = np.sum((Predim > jj).astype('uint8'))

        precision_value = (real_tp + eps) / (real_p + eps)
        recall_value = (real_tp + eps) / (real_t + eps)
        stats_fscore.append([precision_value, recall_value])

    return stats, jaccard_value, mae_value, stats_fscore

def computeALL(gt_path, pred_path):
    gt_list = get_image_list(gt_path)
    nim = len(gt_list)

    results = Parallel(n_jobs=64)(
        delayed(compute_metrics)(im, gt_path, pred_path) for im in tqdm(gt_list, desc="Calculating Metrics:")
    )

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    for i, (metric_stats, jaccard_value, mae_value, fscore_values) in enumerate(results):
        stats[i, :] = metric_stats
        stats_jaccard[i] = jaccard_value
        stats_mae[i] = mae_value
        stats_fscore[:, i, :] = fscore_values

    # Print BER
    posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
    negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
    pA = 100 - 100 * posAcc
    nA = 100 - 100 * negAcc
    BER = 0.5 * (2 - posAcc - negAcc) * 100
    print('BER, S-BER, N-BER:')
    print(BER, pA, nA)

    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)

    # Print Fmeasure
    precision_threshold_list = np.mean(stats_fscore[:, :, 0], axis=1).tolist()
    recall_threshold_list = np.mean(stats_fscore[:, :, 1], axis=1).tolist()
    fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
    print('Fmeasure:', fmeasure)

    return {"BER": BER, "S-BER": pA, "N-BER": nA, "IoU": jaccard_value, "MAE": mean_mae_value, "Fmeasure": fmeasure}

if __name__ == '__main__':
    gt_path = "/home/haipeng/Code/Data/ViSha/test/labels"
    pred = "your_pred_path"
    measure = computeALL(gt_path,pred)
    print(measure)