# Author: Elisa Gambicchia (2021)
# Script filter the entire dataset for language variants and creates train, dev and test set
# making sure that there is enough variety per speakers' pool.

import os
import pandas as pd

def read_df(file):
    df = pd.read_csv(file, sep='\t')
    full_df = df[df['accent'].notnull()]
    return full_df

# filter accents
def filter(df):
    print('\n__________ALL ACCENTS__________')
    print(df['accent'].value_counts())

    # filtering out very limited amount of data or "other"
    df = df[(df['accent'] != 'wales') & (df['accent'] != 'malaysia') & (df['accent'] != 'bermuda')
            & (df['accent'] != 'southatlandtic') & (df['accent'] != 'other')
            & (df['accent'] != 'singapore') & (df['accent'] != 'hongkong')]

    # no duplicates in text prompts
    df = df.sort_values(by='accent', ascending=False)
    filtered_df = df.drop_duplicates(subset='sentence', keep='last')
    print(f"How many utterances after no text duplicates: {len(filtered_df.index)}")  # 338,980
    print(filtered_df['accent'].value_counts())
    return filtered_df


def filter_speakers(df):
    # empty dictionaries to be populated with accent:dataframe pair
    df_dict = {}
    df_train_dict = {}
    df_dev_dict = {}
    df_test_dict = {}

    non_dup_accents = set(df['accent'])
    for accent in non_dup_accents:
        if accent in ['us', 'england', 'indian', 'australia', 'canada', 'african', 'ireland']:
            threshold = 150
        if accent in ['philippines', 'newzealand','scotland']:
            threshold = 400

        print("Filtering speakers:")
        print(f'__________ ACCENT --> {accent}')
        df_accent = df[df['accent'] == accent]  # working on one accent at the time
        df_accent = df_accent.sort_values(by='client_id', ascending=False)  # most common speakers on top

        # for each client id, see if there are too many sentences -- we want variety
        for client, n_sentences in df_accent['client_id'].value_counts().items():
            if n_sentences > threshold:
                accent_client = df_accent[df_accent['client_id'] == client]  # take that client
                accent_sampling = accent_client.sample(n=threshold, replace=False, random_state=1)  # allowing only some threshold per speaker
                try:
                    accent_100 = pd.concat([accent_100, accent_sampling], axis=0)
                except NameError:
                    accent_100 = accent_sampling
            else:
                try:
                    accent_client = df_accent[df_accent['client_id'] == client]
                    accent_100 = pd.concat([accent_100, accent_client], axis=0)
                except NameError:
                    accent_100 = df_accent[df_accent['client_id'] == client]

        # splitting the dataframe
        accent_train = accent_100.head(1800)
        accent_left = accent_100[~accent_100['client_id'].isin(accent_train['client_id'])]  # not in train set

        accent_dev = accent_left.head(200)
        accent_test = accent_left[~accent_left['client_id'].isin(accent_dev['client_id'])].sample(200)

        # populating dictionaries
        df_train_dict[accent] = accent_train
        df_dev_dict[accent] = accent_dev
        df_test_dict[accent] = accent_test

        # aggregating all the dataset per accent together into a dict
        accent_total = pd.concat([accent_train, accent_dev, accent_test], axis=0)
        df_dict[accent] = accent_total

        accent_100 = pd.DataFrame()

        assert len(accent_train.index) + len(accent_dev.index) + len(accent_test.index) == 2200

    return df_dict, df_train_dict, df_dev_dict, df_test_dict

# paths to go to files-to-extract.txt
def write_files_to_extract(dict_dataframes, file_to_write):
    paths = [dataframe['path'] for accent, dataframe in dict_dataframes.items()]

    # write open file with all the files to extract
    file = open(file_to_write, "w")  # write mode
    for accent in paths:
        for item in accent:
            file.write(f"cv-corpus-6.1-2020-12-11/en/clips/{item}\n")
    file.close()

# combining dataframes
def combine_dataframe(dictionary):
    total_df = pd.concat(dictionary.values(), axis=0)
    print(total_df['accent'].value_counts())
    return total_df

def main():
    # reading file as dataframe
    full_df = read_df("/group/project/cstr1/mscslp/2020-21/s2065084_ElisaGambicchia/cv-corpus-6.1-2020-12-11/en/validated_restricted.tsv")

    # filter dataframe for accents + utterances
    filtered_df = filter(full_df)

    # dictionary for dataframes
    df_dict, df_train_dict, df_dev_dict, df_test_dict = filter_speakers(filtered_df)

    # combining dataframes
    df_train = combine_dataframe(df_train_dict)
    df_dev = combine_dataframe(df_dev_dict)
    df_test = combine_dataframe(df_test_dict)
    df_all = combine_dataframe(df_dict)
    print(f"number of clips summing train, test, dev: {df_all.shape}")

    # final results: 3 dataframes train, dev and test
    # write data to tsv

    df_train.to_csv('bigger_dataset/train.tsv', index=False, sep='\t')
    df_dev.to_csv('bigger_dataset/dev.tsv', index=False, sep='\t')
    df_test.to_csv('bigger_dataset/test.tsv', index=False, sep='\t')
    df_all.to_csv('bigger_dataset/all.tsv', index=False, sep='\t')



if __name__ == '__main__':
    main()
