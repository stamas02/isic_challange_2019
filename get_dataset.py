import argparse
import os
import zipfile

from src import web_helper

LINK_TRAINING_DATA = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
LINK_TRAINING_LABELS = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
LINK_TEST_DATA = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip"


def main(output):
    output_file = os.path.join(output, "data.zip")

    print("Downloading data. Might take a while.")
    for url in [LINK_TRAINING_DATA, LINK_TEST_DATA]:
        web_helper.download_url(url, output_file)
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            print("Extracting zip file...")
            zip_ref.extractall(output)
        os.remove(output_file)

    print("Downloading ground truth.")
    for url in [LINK_TRAINING_LABELS]:
        web_helper.download_url(url, os.path.join(output, os.path.basename(url)))

    pass


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Downloads the ISIC 2017 Challange data.')
    # Dataset Arguments
    parser.add_argument("--output", "-o",
                        type=str,
                        help='String Value - Destination path',
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
