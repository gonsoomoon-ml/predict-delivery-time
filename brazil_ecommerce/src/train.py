import ast
import argparse
import logging
import os
import json
import boto3
import pickle
from prettytable import PrettyTable

import subprocess
import sys

from urllib.parse import urlparse

#os.system('pip install autogluon')

from autogluon import TabularPrediction as task

import pandas as pd # this should come after the pip install. 

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

def __load_input_data(path: str):
    """
    Load training data as dataframe
    :param path:
    :return: DataFrame
    """
    input_data_files = os.listdir(path)
    try:
        input_dfs = [pd.read_csv(f'{path}/{data_file}') for data_file in input_data_files]
        return task.Dataset(df=pd.concat(input_dfs))
    except:
        print(f'No csv data in {path}!')
        return None

def format_for_print(df):
    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
        table.add_row(row[1:])
    return str(table)

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')



# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(args):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.

    model_dir = args.model_dir
    target = args.label
    presets = args.presets
    # Load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    train_data = __load_input_data(args.train)

    # train_file_path = get_file_path(args.train, args.train_filename)
    # train_data = task.Dataset(file_path= train_file_path )
    columns = train_data.columns.tolist()
    column_dict = {"columns":columns}
    with open('columns.pkl', 'wb') as f:
        pickle.dump(column_dict, f)
    
    subsample_size = int(args.train_rows)  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)


    predictor = task.fit(train_data = train_data, label=target, 
                         output_directory=model_dir,
                         presets = presets)

    # Results summary
    predictor.fit_summary(verbosity=1)

    # Optional test data
    if args.test:
        print(f'Test files: {os.listdir(args.test)}')
        test_data = __load_input_data(args.test)
        # Test data must be labeled for scoring

        # Leaderboard on test data
        print('Running model on test data and getting Leaderboard...')
        leaderboard = predictor.leaderboard(dataset=test_data, silent=True)
        print(format_for_print(leaderboard), end='\n\n')

        # Feature importance on test data
        # Note: Feature importance must be calculated on held-out (test) data.
        # If calculated on training data it will be biased due to overfitting.
        if args.feature_importance:      
            print('Feature importance:')
            # Increase rows to print feature importance                
            pd.set_option('display.max_rows', 500)
            print(predictor.feature_importance(test_data))

    # Files summary
    print(f'Model export summary:')
    print(f"/opt/ml/model/: {os.listdir('/opt/ml/model/')}")
    models_contents = os.listdir('/opt/ml/model/models')
    print(f"/opt/ml/model/models: {models_contents}")
    print(f"/opt/ml/model directory size: {du('/opt/ml/model/')}\n")

    
    return predictor



# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
#    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.register('type','bool', lambda v: v.lower() in ('yes', 'true', 't', '1'))    

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
#     parser.add_argument('--train_filename', type=str, default='train.csv')
#     parser.add_argument('--test_filename', type=str, default='train.csv')
#     parser.add_argument('--s3_output', type=str, default=os.environ['SM_HP_S3_OUTPUT'])    
#     parser.add_argument('--train_job_name', type=str, default='autogluon')    
    parser.add_argument('--label', type=str, default='target')        
    parser.add_argument('--train_rows', type=str, default=50)            
    parser.add_argument('--target', type=str, default='target')

    # Arguments to be passed to task.fit()
    # Additional options
    parser.add_argument('--presets', type=str, default='optimize_for_deployment')    
    parser.add_argument('--feature_importance', type='bool', default=True)

    return parser.parse_args()

    

# ------------------------------------------------------------ #
# Util Functions                                         
# ------------------------------------------------------------ #

# def get_file_path(folder, file_name):
#     file_path = folder + '/' + file_name  
#     print("file_path: ", file_path)
#     print(subprocess.check_output(["ls", file_path]))

#     return file_path

def display_args(args):
    '''
    # 모든 파라미터를 보여주기    
    '''
    print("######## Display Arguments #########")
    for arg in vars(args):
        print (f'{arg}: {getattr(args, arg)}')


# def get_bucket_prefix(args):
#     '''
#     bucket, prefix 가져오기
#     '''
#     u = urlparse(args.s3_output, allow_fragments=False)
#     bucket = u.netloc
#     print("bucket: ", bucket)
#     prefix = u.path.strip('/')
#     print("prefix: ", prefix)

#     return bucket, prefix

# def inference(test_data, predictor):
#     s3 = boto3.client('s3')

#     try:
#         y_test = test_data[args.label]  # values to predict
#         test_data_nolab = test_data.drop(labels=[args.label], axis=1) # delete label column to prove we're not cheating

#         y_pred = predictor.predict(test_data_nolab)
#         y_pred_df = pd.DataFrame.from_dict({'True': y_test, 'Predicted': y_pred})
#         pred_file = f'test_predictions.csv'
#         y_pred_df.to_csv(pred_file, index=False, header=True)

#         leaderboard = predictor.leaderboard()
#         lead_file = f'leaderboard.csv'
#         leaderboard.to_csv(lead_file)

#         files_to_upload = [pred_file, lead_file]

#     except Exception as ex:
#         print("Error occured")
#         print(ex)


#     for file in files_to_upload:
#         s3.upload_file(file, bucket, os.path.join(prefix, args.train_job_name.replace('mxnet-training', 'autogluon', 1), file))


if __name__ == '__main__':
    # 파라미터 받기    
    args = parse_args()
    # 파라미터 프린트
    display_args(args)
    # 버킷, 프리픽스 가져오기
    # bucket, prefix = get_bucket_prefix(args)
    
    # Verify label is included
    if 'label' == args.label:
        raise ValueError('"label" is a required parameter!')


    # 훈련 및 추론   
    train(args)
    
    # Package inference code with model export
    print("#### Package Inference Codes #### ")
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    subprocess.call('cp columns.pkl /opt/ml/model/code/'.split())

    




