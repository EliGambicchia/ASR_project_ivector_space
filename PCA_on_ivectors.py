# Author: Elisa Gambicchia (2021)
# Implementation of Principal Component Analysis (PCA) for the visualisation of the i-vector space according to language variants
# PCA code taken from https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from kaldi_io_dir.kaldi_io import kaldi_io
import glob
import os
import re
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

pd.set_option('display.width', 1000)

# Reading directory to get all the ivector paths (.scp)
def collect_ivec_files(path_ivectors):
    os.chdir(path_ivectors)
    ivectors_list = [file for file in glob.glob("*.scp")]
    return ivectors_list

def from_file_to_dataframe(ivectors_list):
    '''
    :param ivectors_list: list of ivector matrices with scp extension
    :return: a dataframe with all ivectors (indexed by key/utterance)
    '''
    for ivec in ivectors_list:
        for key, mat in kaldi_io.read_mat_scp(ivec):
            df_ivector_matrix = pd.DataFrame(mat.mean(0)).T # mean of all the frames-ivectors into one ivec per utterance
            df_ivector_matrix['utterance'] = key

            try:
                df_all_ivectors = pd.concat([df_all_ivectors, df_ivector_matrix], axis=0)
            except NameError:
                df_all_ivectors = df_ivector_matrix

    return df_all_ivectors

def attach_accent(df_ivectors, tsv_file):
    '''
    :param df_ivectors: ivectors dataframe: each row has utterance info + 100 dimensions of ivectors
    :param tsv_file: tsv file with all the training data and its metadata (including client_id and accent)
    :return: the extended dataframe: ivectors dataframe augmented with accent info
    '''

    utterances = df_ivectors['utterance']

    clients = []
    for utterance in utterances:

        # take only speaker_id from utterance
        mo = re.compile(r"(sp[0-9].[0-9]-)?(?P<speaker_id>.+)-common_voice_en")  # sp is for speed perturbation
        speaker_id = mo.search(utterance)
        client_id = speaker_id['speaker_id']
        clients.append(client_id)

    # adding client_id to ivectors dataframe
    df_ivectors['client_id'] = clients

    # loading the dataset with accent info + creating subset with only accent and client_id info
    df_info = pd.read_csv(tsv_file, sep='\t')
    df_duo = df_info[['client_id', 'accent']]
    df_duo.set_index('client_id', inplace=True)
    accent_dictionary = df_duo.to_dict()['accent']

    # merging datasets
    df_ivectors["accent"] = df_ivectors["client_id"].map(accent_dictionary)
    df_ivectors.reset_index(inplace=True, drop=True)

    return df_ivectors

def PCA_code(df):
    '''
    Getting a dataframe and perform PCA and make a visualisation
    :param df: daatframe with ivector dimensions + accent
    :return: visualisation
    '''

    X_array = df.iloc[:, 0:100].to_numpy()
    print(type(X_array))

    # scaling
    arr_scaled = StandardScaler().fit_transform(X_array)

    features = arr_scaled.T  # transposed
    cov_matrix = np.cov(features)

    values, vectors = np.linalg.eig(cov_matrix)

    # percentage of explained variance per component
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
    print(f"Total variance: {np.sum(explained_variances)}'\n'Per component:{explained_variances}")

    # visualisation - dot product
    projected_1 = arr_scaled.dot(vectors.T[0])  # first
    projected_2 = arr_scaled.dot(vectors.T[1])

    res = pd.DataFrame(projected_1, columns=['PC1'])
    res['PC2'] = projected_2
    res['accent'] = df['accent']

    # plt.style.available
    plt.style.use(['seaborn-darkgrid'])
    plt.figure(figsize=(20, 10))
    sns.scatterplot(res['PC1'], res['PC2'], hue=res['accent'], s=100)
    # plt.show()
    plt.savefig("/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/PCA_visual_big.pdf")

def main():

    # collecting list of ivectors
    ivec_list = collect_ivec_files("/group/project/cstr1/mscslp/2020-21/s2065084_ElisaGambicchia/recipe_mozilla_big_dataset/exp/nnet3/ivectors_train_sp_hires")

    # reading file
    df_ivectors = from_file_to_dataframe(ivec_list)

    # adding accent column to dataframe
    extended_dataframe = attach_accent(df_ivectors, tsv_file="/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/bigger_dataset/train.tsv")

    # PCA code
    PCA_code(extended_dataframe)



if __name__ == '__main__':
    main()