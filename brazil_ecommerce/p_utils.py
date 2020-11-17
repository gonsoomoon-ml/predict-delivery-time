import boto3, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


#########################
# 전처리
#########################

def upload_s3(bucket, file_path, prefix):
    '''
    bucket = sagemaker.Session().default_bucket()
    prefix = 'comprehend'
    train_file_name = 'test/train/train.csv'
    s3_train_path = upload_s3(bucket, train_file_name, prefix)
    '''
    
    prefix_path = os.path.join(prefix, file_path)
    # prefix_test_path = os.path.join(prefix, 'infer/test.csv')

    boto3.Session().resource('s3').Bucket(bucket).Object(prefix_path).upload_file(file_path)
    s3_path = "s3://{}/{}".format(bucket, prefix_path)
    # print("s3_path: ", s3_path)

    return s3_path

#########################
# 평가
#########################
import itertools


def plot_conf_mat(cm, classes, title, cmap = plt.cm.Greens):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="black" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

