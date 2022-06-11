# Author: Elisa Gambicchia (2021)
# Collecting some info about datasets depending on the language and other filtering constraint

import os
import pandas as pd
import argparse

def read_df(file):
    df = pd.read_csv(file, sep='\t')
    return df

def generic_info(df):
    print(f"Accent information: \n {df['accent'].value_counts()}\n")
    print(f"Gender information: \n {df['gender'].value_counts()}\n")
    print(f"Age information: \n {df['age'].value_counts()}\n")

def count_speakers(df):
    print(f"Number of speakers: {len(df['client_id'].value_counts())}\n")

def number_words(df):
    sentences = df['sentence']
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)

    print(f"Number of words: {len(words)}")


def process_commandline():
    parser = argparse.ArgumentParser(
        description='A basic text-to-speech app that synthesises speech using diphone concatenation.')

    # Arguments for extension tasks
    parser.add_argument('--language', '-l', default=None, choices=['english', 'spanish', 'french', 'irish'],
                        help="Language of the dataset")
    
    parser.add_argument('--filter', '-f', default=None, choices=['baseline', 'more_diversity', 'more_utterances'],
                        help="How to filter utterances for training set")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = process_commandline()

    if args.language == 'irish':
        lang = 'ga-IE'
    if args.language == 'spanish':
        lang = 'es'
    if args.language == 'english':
        lang = 'en'
    if args.language == 'french':
        lang = 'fr'

    sets = ['train', 'dev', 'test']
    for n in sets:
        print(f'___________________SET: {n}_____________________')

        df = read_df(file=f"/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/{lang}/{n}_{args.filter}.tsv")

        accents = set(df['accent'])
        for accent in accents:
            print(f'___________________ACCENT: {accent}_____________________')
            df_accent = df[df['accent'] == accent]
            
            generic_info(df_accent)
            
            count_speakers(df_accent)

            number_words(df_accent)

