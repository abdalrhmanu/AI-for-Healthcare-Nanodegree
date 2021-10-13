import pandas as pd
import numpy as np
import os
import tensorflow as tf

import functools


####### STUDENTS FILL THIS OUT ######
# Question 3


def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = pd.merge(df, ndc_df[['Non-proprietary Name', 'NDC_Code']],
                  left_on='ndc_code',
                  right_on='NDC_Code')
    df['generic_drug_name'] = df['Non-proprietary Name']
    return df


# Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    return df.sort_values(['encounter_id'], ascending=True).groupby('patient_nbr').head(1)



# Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    dataset_len = len(df)
    
    permuted_df = df.iloc[np.random.permutation(dataset_len)]
    unique_keys = permuted_df[patient_key].unique()

    n_total = round(len(unique_keys))
    n_train = round(n_total * .6)
    n_val = round(n_total * .8)

    train = df[df[patient_key].isin(unique_keys[:n_train])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_keys[n_train:n_val])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_keys[n_val:])].reset_index(drop=True)

    return train, validation, test


# Question 7


def create_tf_categorical_feature_cols(categorical_col_list,
                                       vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''

    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")

        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=0)

        if c == 'primary_diagnosis_code':
            dimensions = 10
            cat_col = tf.feature_column.embedding_column(
                tf_categorical_feature_column, dimension=dimensions)
        else:
            cat_col = tf.feature_column.indicator_column(
                tf_categorical_feature_column)
        output_tf_list.append(cat_col)


    return output_tf_list

# Question 8


def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(
        normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,
                                                        default_value=0,
                                                        normalizer_fn=normalizer, dtype=tf.float64)

    return tf_numeric_feature

# Question 9


def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10


def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(
        lambda x: 1 if x >= 5 else 0).values
    print(
        f'### Transformed to numpy: {type(student_binary_prediction)}, shape: {student_binary_prediction.shape}')
    return student_binary_prediction
