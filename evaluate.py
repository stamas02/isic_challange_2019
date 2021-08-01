from src import utils
from src.dataset import ImageData
import pandas as pd
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

DIR_TEST_DATA = "ISIC-2017_Test_v2_Data"
FILE_TEST_LABELS = "ISIC-2017_Test_v2_Part3_GroundTruth.csv"


def evaluate(test_file):
    log_name = os.path.basename(test_file).split("-")[0]
    log_dir = os.path.dirname(test_file)
    classes = "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"

    df = pd.read_csv(test_file)
    results = [[np.array(df[c + "_true"], dtype=bool),np.array(df[c]),  c] for c in classes]

    # ROC
    roc_file_path = os.path.join(log_dir, log_name)
    fig = plt.figure()
    axes = None
    for gt, p, name in results:
        fpr, tpr, thresholds = metrics.roc_curve(y_true=gt, y_score=p)
        df_roc = pd.DataFrame(data={"Fpr": fpr, "Tpr": tpr, "Thresholds": thresholds})
        df_roc.to_csv(f"{roc_file_path}-{name}-roc.csv", index=False, header=True)
        plt.plot(fpr, tpr, label=name)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(f"{roc_file_path}roc.pdf", format="pdf", bbox_inches="tight")

    # Integral Metrics
    auc = [metrics.roc_auc_score(gt, p) for gt, p, _ in results]
    auc_80 = [metrics.roc_auc_score(gt, p, max_fpr=(1 - 0.8)) for gt, p, _ in results]
    avg_precision = [metrics.average_precision_score(gt, p) for gt, p, _ in results]

    # Threshold Metrics
    threshold = 0.5
    cn_matrices = np.array([metrics.confusion_matrix(gt, p >= 0.5).ravel() for gt, p, _ in results])
    tn, fp, fn, tp = cn_matrices[:, 0], cn_matrices[:, 1], cn_matrices[:, 2], cn_matrices[:, 3]

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (tpr + tnr) / 2
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    dice = [metrics.f1_score(gt, p >= 0.5) for gt, p, _ in results]
    jaccard = [metrics.jaccard_score(gt, p >= 0.5) for gt, p, _ in results]

    df_performance = pd.DataFrame(data={"Metrics": [name for _, _, name in results],
                                        "AUC": auc,
                                        "AUC, Sens > 80%": auc_80,
                                        "Average Precision": avg_precision,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": balanced_accuracy,
                                        "Sensitivity": tpr,
                                        "Specificity": tnr,
                                        "Dice Coefficient": dice,
                                        "Jaccard Index": jaccard,
                                        "PPV": ppv,
                                        "NPV": npv})

    df_performance.to_csv(f"{roc_file_path}-performance.csv", index=False, header=True)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate model.')
    # Dataset Arguments
    parser.add_argument("--result", "-r",
                        type=str,
                        help='String Value - path to the *test_result.csv file obtained by running train.py',
                        )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir)
    evaluate(**args.__dict__)
