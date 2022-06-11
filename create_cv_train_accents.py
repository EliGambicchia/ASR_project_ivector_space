# Author: Elisa Gambicchia (2021)

import os
import pandas as pd
import re


def read_df(file):
    df = pd.read_csv(file, sep='\t')
    full_df = df[df['accent'].notnull()]
    restricted_df = full_df[['path', 'accent']]
    return restricted_df

# combining dataframes
def combine_dataframe(list_dataframes):
    total_df = pd.concat(list_dataframes, axis=0)
    print(total_df['accent'].value_counts())
    return total_df

def map_accents(df):
    mapping_dictionary = {'newzealand': 1, 'ireland': 2, 'indian': 3, 'us': 4, 'scotland': 5, 'canada': 6,
                          'england': 7, 'african': 8, 'australia': 9, 'philippines': 10}

    df['accent_n'] = df['accent'].map(mapping_dictionary)
    return df[['path', 'accent_n']]

def replace_path(df):
    search = []
    for values in df['path']:
        search.append(re.search(r'common_voice_en_\d+', values).group())

    df['label'] = search
    return df[['label', 'accent_n']]

def main():
    train_df = read_df("/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/bigger_dataset/train.tsv")
    test_df = read_df("/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/bigger_dataset/test.tsv")
    dev_df = read_df("/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/bigger_dataset/dev.tsv")

    all_df = combine_dataframe([train_df, dev_df, test_df])

    df_with_accent_number = map_accents(all_df)

    df_with_labels = replace_path(df_with_accent_number)

    df_with_labels.to_csv('bigger_dataset/cv-train-accents.tsv', index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()