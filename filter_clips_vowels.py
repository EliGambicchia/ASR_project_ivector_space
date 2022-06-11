# Author: Elisa Gambicchia (2021)
# Work on specific vowels

import pandas as pd
import re
import argparse

pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 3)

def open_clip_list(filename):

    with open(filename, 'r') as f:
        text = f.read()
        clips_list = text.split('\n')
        del clips_list[-1]

    return clips_list

def open_frames_file(frames_file):
    df = pd.read_csv(frames_file, sep=' ', names=['utterance_id', 'frames', 'nan_column'])
    path = []
    for x in df['utterance_id']:
        pattern = re.compile(r'common_voice_en_\d+')
        utterance = pattern.findall(x)
        clip = str(utterance[0]) + '.mp3'
        path.append(clip)

    df['path'] = path

    df = df[df.frames != 0]

    frames_list = list(df['path'])

    return frames_list


def filter_df(tsv_file, type_vowel, clips_list, frames_list):

    df = pd.read_csv(tsv_file, sep='\t')
    print(f"Shape of df before filtering: {df.shape}")

    filtered_df = df[df['path'].isin(clips_list)]
    print(f"Shape of df after filtering aa clips: {filtered_df.shape}")

    filtered_df = df[df['path'].isin(frames_list)]
    print(f"Shape of df after filtering 0-frame clips: {filtered_df.shape}")

    filtered_df.to_csv(f'/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/{type_vowel}/train.tsv', index=False, sep='\t')


def process_commandline():

    parser = argparse.ArgumentParser(description='Filtering clips based on frame lists')

    parser.add_argument('--vowel', default=None, choices=['aa', 'eh', 'iy'])
    parser.add_argument('--utt2frames_dir', default=None) # data/train/utt2num_frames

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = process_commandline()

    frames_list = open_frames_file(f"/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/kaldi-accents1/{args.utt2frames_dir}")
    clips = open_clip_list(f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/multitask_folders/clips_{args.vowel}.txt')

    filter_df(tsv_file='/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/train.tsv',
              type_vowel=f'{args.vowel}',
              clips_list=clips,
              frames_list=frames_list)




