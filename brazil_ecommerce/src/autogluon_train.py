
import argparse
import logging
import os
import json
import boto3

import subprocess
import sys

from urllib.parse import urlparse

#os.system('pip install autogluon')

from autogluon import TabularPrediction as task

import pandas as pd # this should come after the pip install. 

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case an AutoGluon network)
    """
    net = task.load(model_dir)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The AutoGluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    data = json.loads(data)
    # the input request payload has to be deserialized twice since it has a discrete header
    data = json.loads(data)
    df_parsed = pd.DataFrame(data)

    prediction = net.predict(df_parsed)

    response_body = json.dumps(prediction.tolist())

    return response_body, output_content_type

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(args):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.

    model_dir = args.model_dir
    target = args.label_column

    train_file_path = get_file_path(args.train, args.train_filename)

    train_data = task.Dataset(file_path= train_file_path )
    subsample_size = int(args.train_rows)  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)


    predictor = task.fit(train_data = train_data, label=target, output_directory=model_dir)

    return predictor



# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train_filename', type=str, default='train.csv')
    parser.add_argument('--test_filename', type=str, default='train.csv')
    parser.add_argument('--s3_output', type=str, default=os.environ['SM_HP_S3_OUTPUT'])    
    parser.add_argument('--train_job_name', type=str, default='autogluon')    
    parser.add_argument('--label_column', type=str, default='target')        
    parser.add_argument('--train_rows', type=str, default=50)            


    parser.add_argument('--target', type=str, default='target')


    return parser.parse_args()

# ------------------------------------------------------------ #
# Util Functions                                         
# ------------------------------------------------------------ #

def get_file_path(folder, file_name):
    file_path = folder + '/' + file_name  
    print("file_path: ", file_path)
    print(subprocess.check_output(["ls", file_path]))

    return file_path

def display_args(args):
    '''
    # 모든 파라미터를 보여주기    
    '''
    for arg in vars(args):
        print (f'{arg}: {getattr(args, arg)}')


def get_bucket_prefix(args):
    '''
    bucket, prefix 가져오기
    '''
    u = urlparse(args.s3_output, allow_fragments=False)
    bucket = u.netloc
    print("bucket: ", bucket)
    prefix = u.path.strip('/')
    print("prefix: ", prefix)

    return bucket, prefix

def inference(test_data, predictor):
    s3 = boto3.client('s3')

    try:
        y_test = test_data[args.label_column]  # values to predict
        test_data_nolab = test_data.drop(labels=[args.label_column], axis=1) # delete label column to prove we're not cheating

        y_pred = predictor.predict(test_data_nolab)
        y_pred_df = pd.DataFrame.from_dict({'True': y_test, 'Predicted': y_pred})
        pred_file = f'test_predictions.csv'
        y_pred_df.to_csv(pred_file, index=False, header=True)

        leaderboard = predictor.leaderboard()
        lead_file = f'leaderboard.csv'
        leaderboard.to_csv(lead_file)

        files_to_upload = [pred_file, lead_file]

    except Exception as ex:
        print("Error occured")
        print(ex)


    for file in files_to_upload:
        s3.upload_file(file, bucket, os.path.join(prefix, args.train_job_name.replace('mxnet-training', 'autogluon', 1), file))


if __name__ == '__main__':
    # 파라미터 받기    
    args = parse_args()
    # 파라미터 프린트
    display_args(args)
    # 버킷, 프리픽스 가져오기
    bucket, prefix = get_bucket_prefix(args)

    # 훈련    
    predictor = train(args)

    # 테스트 파일 경로 받기
    test_file_path = get_file_path(args.test, args.test_filename)
    test_data = task.Dataset(file_path= test_file_path)

    # 추론
    inference(test_data, predictor)



