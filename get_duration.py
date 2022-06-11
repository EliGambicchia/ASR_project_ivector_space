# Author: Elisa Gambicchia (2021)
# Script takes accent info from external file (tsv) and computes duration of audio files per accent

import pandas as pd
import re

pd.set_option('display.max_colwidth', None)

def read_df(file):
    df = pd.read_csv(file, sep=' ', names=["Utt_id", "Duration", "nan"])
    df = df[["Utt_id", "Duration"]]
    return df


def replace_path(df):
    search = []
    for values in df['Utt_id']:
        search.append(re.search(r'(.*)-', values).group(1))

    df['client_id'] = search
    print(df['client_id'])

    return df

def attach_accent(df_duration, tsv_file_metadata):
    '''
    :param df_ivectors: ivectors dataframe: each row has utterance info + 100 dimensions of ivectors
    :param tsv_file_metadata: tsv file with all the training data and its metadata (including client_id and accent)
    :return: the extended dataframe: ivectors dataframe augmented with accent info
    '''

    # loading the dataset with accent info + creating subset with only accent and client_id info
    df_info = pd.read_csv(tsv_file_metadata, sep='\t')

    # creating mapping for accent
    df_duo = df_info[['client_id', 'accent']]
    df_duo.set_index('client_id', inplace=True)
    accent_dictionary = df_duo.to_dict()['accent']

    # merging datasets
    df_duration["accent"] = df_duration["client_id"].map(accent_dictionary)
    df_duration.reset_index(inplace=True, drop=True)

    return df_duration

def sum_duration(df_duration):

    accents = set(df['accent'])
    for accent in accents:
        df_accent = df[df['accent'] == accent]
        print(accent)
        print(df_accent['Duration'].sum())


def main():
    df = read_df('./kaldi-accents1/data/test/utt2dur')

    df_duration = replace_path(df)

    df_with_accent = attach_accent(df_duration, tsv_file_metadata='./kaldi-accents1/data/test.tsv')

    sum_duration(df_with_accent)



if __name__ == '__main__':
    main()


