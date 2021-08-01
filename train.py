from src import utils
from src.dataset import ImageData
from src.model import EfficientNetClassifier
from evaluate import evaluate
import pandas as pd
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
from tqdm import tqdm
import time

DIR_TRAINING_DATA = "ISIC_2019_Training_Input"
FILE_TRAINING_LABELS = "ISIC_2019_Training_GroundTruth.csv"
NUM_CLASSES = 8


def read_datasets(dataset_files):
    df = pd.DataFrame()
    for dataset_file in dataset_files:
        _df = pd.read_csv(dataset_file)
        df = pd.concat([df, _df], ignore_index=True)
    return df


def train(dataset_dir, image_x, image_y, lr, lr_decay, lr_step, batch_size, epoch, log_dir, log_name, split,
          seed, num_workers):
    df = pd.read_csv(os.path.join(dataset_dir, FILE_TRAINING_LABELS))
    files = [os.path.join(dataset_dir, DIR_TRAINING_DATA, f + ".jpg") for f in df.image]
    labels = np.array([df.MEL, df.NV, df.BCC, df.AK, df.BKL, df.DF, df.VASC, df.SCC, df.UNK], dtype=int)
    labels = np.argmax(labels, axis=0)

    train_x, _x, train_y, _y = train_test_split(files, labels, test_size=split, random_state=seed)
    test_x, val_x, test_y, val_y = train_test_split(_x, _y, test_size=split, random_state=seed)

    train_dataset = ImageData(train_x, train_y, transform=utils.get_train_transform((image_x, image_y)))
    val_dataset = ImageData(val_x, val_y, transform=utils.get_test_transform((image_x, image_y)))
    test_dataset = ImageData(test_x, test_y, transform=utils.get_test_transform((image_x, image_y)))

    print("Training set: {0}".format(len(train_x)))
    print("Validation set: {0}".format(len(val_x)))
    print("Test set: {0}".format(len(test_x)))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)

    device = torch.device("cuda")
    model = EfficientNetClassifier(b=0, num_classes=NUM_CLASSES).to(device)
    optimizer = SGD(params=model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    df_train_log = pd.DataFrame(columns=['epoch', 'train-loss', 'val-loss'])

    # TRAIN
    for _epoch in range(epoch):
        model.train()
        train_loss = 0
        p_bar = tqdm(train_data_loader, desc=f"Training epoch {_epoch}")
        for i, (images, labels, _) in enumerate(p_bar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, dropout=True)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            train_loss = train_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))
            p_bar.set_postfix({'loss': train_loss})
            optimizer.step()

        scheduler.step()
        model.eval()
        val_loss = 0
        p_bar = tqdm(val_data_loader, desc=f"Validation epoch {_epoch}")
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(p_bar):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images, dropout=False)
                loss = F.cross_entropy(logits, labels)
                val_loss = val_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))

        df_train_log = df_train_log.append({'epoch': _epoch,
                                            'train-loss': train_loss,
                                            'val-loss': val_loss}, ignore_index=True)

    df_train_log.to_csv(os.path.join(log_dir, log_name + "-train_log.csv"), index=False, header=True)
    torch.save(model, os.path.join(log_dir, log_name + "-model.pt"))

    # TEST
    test_predictions = []
    test_files = []
    test_labels = []
    with torch.no_grad():
        for images, _labels, _files in tqdm(test_data_loader, desc="Predicting on test set"):
            images = images.to(device)
            logits = model(images, dropout=False)
            test_predictions += torch.softmax(logits, 1).detach().cpu().numpy().tolist()
            test_files += _files
            test_labels += _labels.detach().cpu().numpy().tolist()

    test_predictions = np.array(test_predictions)
    # Converting test lables to hot vector representation
    _test_labels = np.zeros((len(test_labels), NUM_CLASSES))
    _test_labels[list(range(len(test_labels))), test_labels] = 1
    test_labels = _test_labels

    df_test_log = pd.DataFrame(data=np.concatenate([test_predictions, test_labels], axis=1),
                               columns=["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC",
                                        "MEL_true", "NV_true", "BCC_true", "AK_true", "BKL_true",
                                        "DF_true", "VASC_true", "SCC_true"])
    df_test_log["file"] = test_files

    df_test_log.to_csv(os.path.join(log_dir, log_name + "-test_result.csv"), index=False, header=True)

    evaluate(os.path.join(log_dir, log_name + "-test_result.csv"))


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate different thresholds')
    # Dataset Arguments
    parser.add_argument("--dataset-dir", "-d",
                        type=str,
                        help='String Value - The folder where the dataset is downloaded using get_dataset.py',
                        )
    parser.add_argument("--image_x", type=int,
                        default=300,
                        help="Integer Value - Width of the image that should be resized to.")
    parser.add_argument("--image_y", type=int,
                        default=225,
                        help="Integer Value - Height of the image that should be resized to.")
    parser.add_argument("--split", type=float,
                        default=0,
                        help="Floating Point Value - Ratio with which the dataset split to train/test/val subsets."
                             "E.g the value 0.2 would result in a split 0.8/0.16/0.04")

    # Training Arguments
    parser.add_argument("--lr", type=float,
                        default=0.1,
                        help="Floating Point Value - Starting learning rate.")
    parser.add_argument("--lr_decay", type=float,
                        default=0.5,
                        help="Floating Point Value - Learning rate decay.")
    parser.add_argument("--lr_step", type=float,
                        default=15,
                        help="Integer Value - Decay learning rate after stepsize.")
    parser.add_argument("--batch_size", type=int,
                        default=2,
                        help="Integer Value - The sizes of the batches during training.")
    parser.add_argument("--epoch", type=int,
                        default=0,
                        help="Integer Value - Number of epoch.")

    # Logging Arguments
    parser.add_argument("--log-dir", type=str,
                        help="String Value - Path to the folder the log is to be saved.")
    parser.add_argument("--log-name", type=str,
                        default=225,
                        help="String Value - This is a descriptive name of the method. "
                             "Will be used in legends e.g. ROC curve")

    parser.add_argument("--seed", type=float,
                        default=7,
                        help="Floating Point Value - Random Seed")

    parser.add_argument("--num-workers", type=float,
                        default=2,
                        help="Integer Value - Number of workers used for parallel data loading.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir)
    train(**args.__dict__)
