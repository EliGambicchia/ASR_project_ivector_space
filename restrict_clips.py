# Author: Elisa Gambicchia (2021)


import os
import pandas as pd

def read_df(file):
    df = pd.read_csv(file, sep='\t')
    print(f"Size of dataframe before dropping: {df.shape}")
    return df

def open_clips_file(txt_file):
    with open(txt_file, 'r') as f:
        text = f.read()
        clips = text.split('\n')
    print(f"Type of clips: {type(clips)}")
    print(f"Number of clips: {len(clips)}")
    return clips

def dropping_404_clips(clips, df):
    df = df[df['path'].isin(clips)]
    print(f"Size of dataframe after dropping: {df.shape}")
    return df

def write_tsv_file(df, path):
    df.to_csv(path, index=False, sep='\t')


def main():
    # reading file as dataframe
    full_df = read_df("/afs/inf.ed.ac.uk/user/s20/s2065084/PycharmProjects/mozilla_data/bigger_dataset/all.tsv")

    # clips
    clips = open_clips_file("/group/project/cstr1/mscslp/2020-21/s2065084_ElisaGambicchia/cv-corpus-6.1-2020-12-11/en/count_clips.txt")

    # dropping clips (paths) that are not in the list of clips extracted
    restricted_df = dropping_404_clips(clips=clips, df=full_df)

    # write new validated file
    #write_tsv_file(df=restricted_df, path="/group/project/cstr1/mscslp/2020-21/s2065084_ElisaGambicchia/cv-corpus-6.1-2020-12-11/en/validated_restricted.tsv")



if __name__ == '__main__':
    main()