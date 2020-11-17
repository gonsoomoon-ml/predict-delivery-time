import argparse
import ast
import os

def parse_args():
#    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.register('type','bool', lambda v: v.lower() in ('yes', 'true', 't', '1'))    

    parser.add_argument('--feature_importance', type='bool', default=True)       
#     # Arguments to be passed to task.fit()
    parser.add_argument('--fit_args', type=str,
                        default="test")


#     parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
#                         default="{'presets': ['optimize_for_deployment']}",
#                         help='https://autogluon.mxnet.io/api/autogluon.task.html#tabularprediction')

    return parser.parse_args()

def display_args(args):
    '''
    # 모든 파라미터를 보여주기    
    '''
    for arg in vars(args):
        print (f'{arg}: {getattr(args, arg)}')



if __name__ == '__main__':
    # 파라미터 받기    
    args = parse_args()
    display_args(args)



