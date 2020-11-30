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

def filter_df(raw_df, cols):
    '''
    cols = ['label','userno', 'ipaddr','try_cnt','paytool_cnt','total_amount','EVENT_DATE']    
    df = filter_df(df, cols)
    '''
    df = raw_df.copy()
    df = df[cols]
    return df


def split_data_date(raw_df, sort_col,data1_end, data2_start):
    '''
    train, test 데이터 분리
    train_end = '2020-01-31'
    test_start = '2020-02-01'
    train_df, test_df = split_data_date(df, sort_col='EVENT_DATE',
                                        data1_end = train_end, 
                                    data2_start = test_start)

    '''
    df = raw_df.copy()
    
    df = df.sort_values(by= sort_col) # 시간 순으로 정렬
    # One-Hot-Encoding
    data1 = df[df[sort_col] <= data1_end]
    data2 = df[df[sort_col] >= data2_start]    
        
    print(f"data1, data2 shape: {data1.shape},{data2.shape}")
    print(f"data1 min, max date: {data1[sort_col].min()}, {data1[sort_col].max()}")
    print(f"data2 min, max date: {data2[sort_col].min()}, {data2[sort_col].max()}")        
    
    return data1, data2

def convert_date_type(raw_df, col1, dtype='str'):
    '''
    train_pd = convert_date_type(train_pd, col1='customer_zip_code_prefix')
    '''
    df = raw_df.copy()
    
    if df.columns.isin([col1]).any():
        df[col1] = df[col1].astype(dtype)
        print(df[col1].dtypes)
    else:
        pass
    return df


#########################
# 레이블 인코더
#########################


# from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
class LabelEncoderExt(object):
    '''
    Source:
    # https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    '''
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)
    
def make_test_label_encoding(raw_train_df, raw_test_df,cols):
    '''
    label_cols = ['customer_city','customer_state','customer_zip_code_prefix']
    train_pd_lb, test_pd_lb = make_test_label_encoding(train_pd, test_pd, label_cols)    
    '''
    train_df = raw_train_df.copy()
    test_df = raw_test_df.copy()
    
    for lb_col in cols:
        print("Starting: ", lb_col)
        le = LabelEncoderExt()
        le = le.fit(train_df[lb_col])
        
        train_en = le.transform(train_df[lb_col])
        test_en = le.transform(test_df[lb_col])        
        lb_col_name = 'lb_' + lb_col
        print("new col name: ", lb_col_name)
        train_df[lb_col_name] = train_en
        train_df[lb_col_name] = train_df[lb_col_name].astype('str')     
        test_df[lb_col_name] = test_en        
        test_df[lb_col_name] = test_df[lb_col_name].astype('str')                
    
    return train_df, test_df




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

