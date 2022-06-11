# Author: Elisa Gambicchia (2021)
# Implementation of Principal Component Analysis (PCA), also combined with other dimensionality reduction techniques (LDA and TSNE) 
# for the visualisation of the MFCCs space according to language variants and gender of speakers 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from dir_kaldi_io.kaldi_io import kaldi_io
import glob
import os
import re
import argparse

from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pd.set_option('display.width', 1000)

# Reading directory to get all the ivector paths (.scp)
def collect_mfcc_files(mfcc_folder, file_type):
    os.chdir(mfcc_folder)
    mfcc_list = [file for file in glob.glob(f"{file_type}")] # here
    print(mfcc_list)
    return mfcc_list

def from_file_to_dataframe(mfcc_list):
    '''
    :param ivectors_list: list of ivector matrices with scp extension
    :return: a dataframe with all ivectors (indexed by key/utterance)
    '''
    for m in mfcc_list:
        for key, mat in kaldi_io.read_mat_scp(m):

            df_matrix = pd.DataFrame(mat)

            # filtering to log energy more than 0
            #df_matrix.drop(df_matrix[df_matrix[12] < 2].index, inplace=True) # elisa: here to modify filter of log energy
            mfcc_matrix = pd.DataFrame(df_matrix.mean(0)).T

            mfcc_matrix['utterance'] = key

            try:
                df_all_mfccs = pd.concat([df_all_mfccs, mfcc_matrix], axis=0, ignore_index=True)
                print(f"This is the shape of the df_all_mfccs: {df_all_mfccs.shape}\n")
            
            except NameError:
                df_all_mfccs = mfcc_matrix

    print(f"This is the shape of the df_all_mfccs: {df_all_mfccs.shape}")
    
    df_all_mfccs.dropna(inplace=True) 

    print(f"This is the shape of the df_all_mfccs, after DROPPING NA: {df_all_mfccs.shape}")

    df_all_mfccs.to_csv(f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/df_mfcc{args.mfcc_dim}_{args.dataset}.tsv', index=False, sep='\t') # here

    return df_all_mfccs

def from_csv_to_df(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    return df

def attach_accent(df_mfccs, tsv_file):
    '''
    :param df_ivectors: ivectors dataframe: each row has utterance info + 100 dimensions of ivectors
    :param tsv_file: tsv file with all the training data and its metadata (including client_id and accent)
    :return: the extended dataframe: ivectors dataframe augmented with accent info
    '''

    utterances = df_mfccs['utterance']

    clients = []
    for utterance in utterances:

        # take only speaker_id from utterance
        mo = re.compile(r"(sp[0-9].[0-9]-)?(?P<speaker_id>.+)-common_voice_en")  # sp is for speed perturbation
        speaker_id = mo.search(utterance)
        client_id = speaker_id['speaker_id']
        clients.append(client_id)

    # adding client_id to ivectors dataframe
    df_mfccs['client_id'] = clients

    # loading the dataset with accent info + creating subset with only accent and client_id info
    df_info = pd.read_csv(tsv_file, sep='\t')

    df_duo = df_info[['client_id', 'accent']]
    df_duo.set_index('client_id', inplace=True)
    accent_dictionary = df_duo.to_dict()['accent']

    # creating mapping for gender
    df_gender = df_info[['client_id', 'gender']]
    df_gender.set_index('client_id', inplace=True)
    gender_dictionary = df_gender.to_dict()['gender']

    # merging datasets
    df_mfccs["accent"] = df_mfccs["client_id"].map(accent_dictionary)
    df_mfccs["gender"] = df_mfccs["client_id"].map(gender_dictionary)

    df_mfccs.reset_index(inplace=True, drop=True)

    print(f"After attaching accent: {df_mfccs.shape}")

    return df_mfccs, accent_dictionary, gender_dictionary


def aggregate_by_speaker(df_mfcc, accent_dictionary, gender_dictionary):

    speakers = set(df_mfcc['client_id'])
    print(f"This is the number of different speakers: {len(speakers)}")
    for speaker in speakers:

        df_single_speaker = df_mfcc[df_mfcc['client_id'] == speaker]

        # reindexing from the start
        df_single_speaker.reset_index(inplace=True)
        del df_single_speaker['index']

        df_mean_single_speaker = pd.DataFrame(df_single_speaker.iloc[:,:40].mean(0)).T # here

        # add column for client_id, accent
        df_mean_single_speaker['client_id'] = speaker

        try:
            df_all_speakers = pd.concat([df_all_speakers, df_mean_single_speaker], axis=0, ignore_index=True)
        except NameError:
            df_all_speakers = df_mean_single_speaker
    
    df_all_speakers["accent"] = df_all_speakers["client_id"].map(accent_dictionary)
    df_all_speakers["gender"] = df_all_speakers["client_id"].map(gender_dictionary)
    
    print(f"After aggregating per speaker: {df_all_speakers.shape}")

    # replacing names of accents
    df_all_speakers["accent"].replace({"england":"ENG", "us": "US", "indian": "IND", "african": "AFR", "australia":"AUS", "canada":"CAN", "scotland":"SCO", "ireland":"IRE", "newzealand":"NZL", "philippines":"PHI"}, inplace=True)
    print(f"Accents: {set(df_all_speakers['accent'])}")

    return df_all_speakers


def four_accent_filter(df, acc1, acc2, acc3, acc4):
    
    df = df[(df['accent'] == acc1) | (df['accent'] == acc2) | (df['accent'] == acc3) | (df['accent'] == acc4)]
    print(f"After filtering for accents, shape of the dataframe: {df.shape}")
    return df

def three_accent_filter(df, acc1, acc2, acc3):
    
    df = df[(df['accent'] == acc1) | (df['accent'] == acc2) | (df['accent'] == acc3)]
    print(f"After filtering for accents, shape of the dataframe: {df.shape}")
    return df

def PCA_version(df, mfcc_dim, target, filename_to_save_fig):
    '''
    PCA code using sklearn
    code from https://github.com/msminhas93/embeddings-visualization/blob/main/Visualization.ipynb

    '''

    print("------------ PCA ----------")
    X_array = df.iloc[:, 0:int(mfcc_dim)].to_numpy()

    # doing the PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_array)

    # creating dataframe
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)

    # explained how much variance ratio
    print(f"Total variance: {pca.explained_variance_ratio_}")
    print(f"Shape of finalDf when doing PCA: {finalDf.shape}")

    # plt.style.available
    scatterplot_settings(df=finalDf, x="PC1", y="PC2", target=target, filename_to_save_fig=filename_to_save_fig)


def tsne_code(df, mfcc_dim, target, filename_to_save_fig):

    # from df to array
    X_array = df.iloc[:, 0:int(mfcc_dim)].to_numpy()

    tsne = TSNE(2, verbose=1,n_iter=300)
    tsne_proj = tsne.fit_transform(X_array)

    principalDf = pd.DataFrame(data = tsne_proj, columns = ['TSNE1', 'TSNE2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)
    print(f"Shape of finalDf when doing TSNE: {finalDf.shape}")

    scatterplot_settings(df=finalDf, x="TSNE1", y="TSNE2", target=target, filename_to_save_fig=filename_to_save_fig)



def pca_and_tsne(df, mfcc_dim, target, filename_to_save_fig):
    # from df to array
    
    print("------------ PCA + TSNE ----------")

    X_array = df.iloc[:, 0:int(mfcc_dim)].to_numpy()

    # PCA
    pca_50 = PCA(n_components=10)
    pca_result_50 = pca_50.fit_transform(X_array)

    # TSNE
    tsne = TSNE(n_components=2, verbose=1, n_iter=300, learning_rate=50.0, perplexity=40.0)
    tsne_proj = tsne.fit_transform(pca_result_50)

    principalDf = pd.DataFrame(data = tsne_proj, columns = ['TSNE1', 'TSNE2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)
    print(f"Shape of finalDf when doing TSNE: {finalDf.shape}")

    scatterplot_settings(df=finalDf, x="TSNE1", y="TSNE2", target=target, filename_to_save_fig=filename_to_save_fig)



def pca_and_lda(df, mfcc_dim, target, filename_to_save_fig):
    # from df to array
    print("------------ PCA + LDA ----------")

    X_array = df.iloc[:, 0:int(mfcc_dim)].to_numpy()
    y = df[['accent']]

    # PCA
    pca_50 = PCA(n_components=10)
    pca_result_50 = pca_50.fit_transform(X_array)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_proj = lda.fit(pca_result_50, y.values.ravel()).transform(pca_result_50)

    principalDf = pd.DataFrame(data = lda_proj, columns = ['LDA1', 'LDA2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)
    print(f"Shape of finalDf when doing lda: {finalDf.shape}")

    scatterplot_settings(df=finalDf, x="LDA1", y="LDA2", target=target, filename_to_save_fig=filename_to_save_fig)


def scatterplot_settings(df, x, y, target, filename_to_save_fig):
    
    plt.figure(figsize=(16,10))

    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    colours = sns.color_palette("tab10")

    if target == 'accent':
        palette = {"US":colours[0], "CAN":colours[8], "ENG":colours[3], "IND":colours[1], "AUS":colours[4], "AFR":colours[9] , "NZL":colours[6], "PHI":colours[7], "IRE":colours[2], "SCO":colours[5]}
        title = 'Accent'

    if target == 'gender':
        palette = {"male":colours[0], "female":colours[1], "other":colours[3]}
        title = "Gender"

    sns.scatterplot(
        x=x, y=y, 
        hue=target,
        data=df,
        legend="full",  
        palette=palette, 
        style=df[target], 
        linewidth=0
        )
    plt.legend(title=title)
    plt.savefig(filename_to_save_fig)


def process_commandline():
    parser = argparse.ArgumentParser(
        description='PCA visualisation for mfcc')

    # Arguments for extension tasks
    parser.add_argument('--filter', '-f', default=None, choices=['yes', 'no'],
                        help="Do I want to filter?")

    parser.add_argument('--acc1', '-a1', default=None, choices=['us', 'england', 'indian', 'australia', 'canada', 'african', 'newzealand' 'ireland', 'scotland', 'philippines'],
                        help="How to filter PCA based on chosen langs")
    
    parser.add_argument('--acc2', '-a2', default=None, choices=['us', 'england', 'indian', 'australia', 'canada', 'african', 'newzealand' 'ireland', 'scotland', 'philippines'],
                        help="How to filter PCA based on chosen langs")

    parser.add_argument('--acc3', '-a3', default=None, choices=['us', 'england', 'indian', 'australia', 'canada', 'african', 'newzealand' 'ireland', 'scotland', 'philippines'],
                        help="How to filter PCA based on chosen langs")

    parser.add_argument('--acc4', '-a4', default=None, choices=['us', 'england', 'indian', 'australia', 'canada', 'african', 'newzealand' 'ireland', 'scotland', 'philippines'],
                        help="How to filter PCA based on chosen langs")

    
    parser.add_argument('--mfcc_dim', default=None, choices=['13', '40'])

    parser.add_argument('--target', '-t', default='accent', choices=['accent', 'gender'])

    parser.add_argument('--dataset', choices=['en_diversity', 'en_utterances', 'aa', 'eh', 'iy'])

    parser.add_argument('--create_df', default=None, choices=['yes', 'no'])

    args = parser.parse_args()
    return args




if __name__ == "__main__":

    args = process_commandline()
    

    if args.mfcc_dim == '13':
        file_type = 'raw_mfcc_train.*.scp'
        mfcc_exp_folder = 'mfcc'

    if args.mfcc_dim == '40':
        file_type = 'raw_mfcc_train_hires.*.scp'
        mfcc_exp_folder = 'mfcc_hires'
    
    mfcc_folder = f"/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/kaldi-accents1/exp_{args.dataset}/{mfcc_exp_folder}"

    if args.create_df:
        mfcc_list = collect_mfcc_files(mfcc_folder=mfcc_folder, file_type=file_type)
        df_mfcc = from_file_to_dataframe(mfcc_list=mfcc_list)

    df_mfcc = from_csv_to_df(tsv_file=f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/df_mfcc{args.mfcc_dim}_{args.dataset}.tsv') # here


    # adding accent column to dataframe
    extended_dataframe, accent_dictionary, gender_dictionary = attach_accent(df_mfcc, tsv_file=f"/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/{args.dataset}/train.tsv") # here

    # aggregated by speakers
    df_speakers = aggregate_by_speaker(df_mfcc=extended_dataframe, accent_dictionary=accent_dictionary, gender_dictionary=gender_dictionary)

    # filter them
    if args.filter == 'yes':
        #df_speakers = four_accent_filter(df_speakers, args.acc1, args.acc2, args.acc3, args.acc4)
        df_speakers = three_accent_filter(df_speakers, args.acc1, args.acc2, args.acc3)
    



    destination_folder = "/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/"

    # PCA
    PCA_version(df_speakers, mfcc_dim=int(args.mfcc_dim), target=f'{args.target}', filename_to_save_fig=f"{destination_folder}/pca_mfcc_{args.mfcc_dim}.pdf")

    # PCA + TSNE
    pca_and_tsne(df_speakers, mfcc_dim=int(args.mfcc_dim), target=f'{args.target}', filename_to_save_fig=f"{destination_folder}/pca+tsne_mfcc_{args.mfcc_dim}.pdf")

    # PCA + LDA
    pca_and_lda(df_speakers, mfcc_dim=int(args.mfcc_dim), target=f'{args.target}', filename_to_save_fig=f"{destination_folder}/pca+lda_mfcc_{args.mfcc_dim}.pdf")




