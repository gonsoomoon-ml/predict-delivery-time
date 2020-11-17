import ast
import argparse
import warnings
import logging
import os
import json
import boto3
import pickle
# from prettytable import PrettyTable

import subprocess
import sys

from urllib.parse import urlparse

#os.system('pip install autogluon')

# from autogluon import TabularPrediction as task

import pandas as pd # this should come after the pip install. 

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', category=DeprecationWarning)
#     from prettytable import PrettyTable
#     import autogluon as ag
#     from autogluon import TabularPrediction as task
#     from autogluon.task.tabular_prediction import TabularDataset

from prettytable import PrettyTable
import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction import TabularDataset



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
#     target = args.label
#     presets = args.presets
    # Load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    train_data = __load_input_data(args.train)

    columns = train_data.columns.tolist()
    column_dict = {"columns":columns}
    with open('columns.pkl', 'wb') as f:
        pickle.dump(column_dict, f)
    
    subsample_size = int(args.train_rows)  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)


#     predictor = task.fit(train_data = train_data, label=target, 
#                          output_directory=model_dir,
#                          presets = presets)
    # Train models
    predictor = task.fit(
        train_data=train_data,
        output_directory= model_dir,
        **args.fit_args,
    )

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
    parser.add_argument('--label', type=str, default='target')        
    parser.add_argument('--train_rows', type=str, default=50)            
    parser.add_argument('--target', type=str, default='target')

    # Arguments to be passed to task.fit()
    # Additional options
    # parser.add_argument('--presets', type=str, default='optimize_for_deployment')    
    # Arguments to be passed to task.fit()
    parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
                        default="{'presets': ['optimize_for_deployment']}",
                        help='https://autogluon.mxnet.io/api/autogluon.task.html#tabularprediction')    
    parser.add_argument('--feature_importance', type='bool', default=True)

    return parser.parse_args()

    

# ------------------------------------------------------------ #
# Util Functions                                         
# ------------------------------------------------------------ #


def display_args(args):
    '''
    # 모든 파라미터를 보여주기    
    '''
    print("######## Display Arguments #########")
    for arg in vars(args):
        print (f'{arg}: {getattr(args, arg)}')



if __name__ == '__main__':
    # 파라미터 받기    
    args = parse_args()
    # 파라미터 프린트
    # display_args(args)
    
    # Verify label is included
    if 'label' not in args.fit_args:
        raise ValueError('"label" is a required parameter of "fit_args"!')
        
    # Convert optional fit call hyperparameters from strings
    if 'hyperparameters' in args.fit_args:
        for model_type,options in args.fit_args['hyperparameters'].items():
            assert isinstance(options, dict)
            for k,v in options.items():
                args.fit_args['hyperparameters'][model_type][k] = eval(v) 
 
    # Print SageMaker args
    print('fit_args:')
    for k,v in args.fit_args.items():
        print(f'{k},  type: {type(v)},  value: {v}')

        

    # 훈련 및 추론   
    train(args)
    
    # Package inference code with model export
    print("#### Package Inference Codes #### ")
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    subprocess.call('cp columns.pkl /opt/ml/model/code/'.split())

    




