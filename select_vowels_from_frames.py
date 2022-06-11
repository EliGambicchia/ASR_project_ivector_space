# Author: Elisa Gambicchia (2021)
# This script selects specific audio frames containing vowels of interests
# and turns frames into timestamps for later processing 

import re
import pandas as pd
from itertools import groupby
from operator import itemgetter
import statistics

def open_pdf_file(filename):
    with open(filename, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')

    pdf_list = []
    for line in lines:
        m = r.match(line)
        if m:
            pdf_identity = m.group('pdf')
            pdf_list.append(pdf_identity)

    pdf_list = [int(e) for e in pdf_list]
    print(f"Matching pdfs: {len(pdf_list)}")
    return pdf_list

def ali_to_dict(file):
    with open(file, 'r') as f:
        text = f.read()
        text = text.split('\n')

        dict_frames = {}

        for line in text:
            elements = line.split(' ')

            indices = []
            for i, x in enumerate(elements):
                try:
                    if int(x) in vowel_indices:
                        indices.append(i)
                # will throw an error for the first element that is a string (utterance id)
                except ValueError:
                    continue

            if len(indices) > 2:
                dict_frames[elements[0]] = indices

    print(dict_frames)
    print(len(dict_frames))

    return dict_frames

def count_frames(dict_frames):

    total_frames = [len(indices) for utterance, indices in dict_frames.items()]
    short_frames = [(utterance, len(indices)) for utterance, indices in dict_frames.items() if len(indices) < 2]

    print(f"Mean: {statistics.mean(total_frames)}")
    print(f"Standard deviation: {statistics.stdev(total_frames)}")

    file = open(f"utt2frames_{vowel}.txt", "w")
    for utterance, indices in dict_frames.items():
        line = f"{utterance} {len(indices)}\n"
        file.write(line)
    file.close()


def longest_vowel(dict_frames):

    utt_vowel_dict = {}
    for utterance, indices in dict_frames.items():

        list_clusters = []
        # consecutive elements
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            clusters = list(map(itemgetter(1), g))
            list_clusters.append(clusters)

        # selecting the longest vowel
        longest_vowel = max(list_clusters, key=len)

        utt_vowel_dict[utterance] = longest_vowel

    return utt_vowel_dict

def frames_to_timestamps(dictionary):

    file = open(f"trimming_files_{vowel}.sh", "w")
    file.write("#!/bin/bash\n")
    for k, v in dictionary.items():

        # HH: MM:SS.MILLISECONDS
        start = v[0]/100 # i want this in milliseconds
        finish = v[-1]/100+0.025

        start_timestamp = '00:00:{:06.3f}'.format(start)
        finish_timestamp = '00:00:{:06.3f}'.format(finish)

        pattern = re.compile(r'common_voice_en_\d+')
        utterance = pattern.findall(k)

        path = f'/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/en/clips/{utterance[0]}.mp3'

        path_new_file = f'/exports/eddie/scratch/s2065084/cv-corpus-6.1-2020-12-11/{vowel}/clips/{utterance[0]}.mp3'

        file.write(f"ffmpeg -i {path} -ss {start_timestamp} -to {finish_timestamp} -c copy {path_new_file}\n")

    file.close()

def make_list_clips(dictionary):
    file = open(f"clips_{vowel}.txt", "w")
    for k, v in dictionary.items():
        pattern = re.compile(r'common_voice_en_\d+')
        utterance = pattern.findall(k)
        clip = f'{utterance[0]}.mp3'

        file.write(f"{clip}\n")
    
    file.close()




def process_commandline():

    parser = argparse.ArgumentParser(description='Filtering clips based on vowel of interest')

    parser.add_argument('--vowel', default=None, choices=['aa', 'eh', 'iy'])
    parser.add_argument('--utt2frames_dir', default=None) # data/train/utt2num_frames

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = process_commandline()
    vowel = args.vowel
    upper_vowel = vowel.upper()

    r = re.compile(r'.*(?P<vowel>%s).*pdf\s=\s(?P<pdf>\d+)' % upper_vowel) 

    vowel_indices = open_pdf_file('/Users/macbookpro/Desktop/UNIVERSITY OF EDINBURGH/DISSERTATION/multitask_folders/phones_pdf_correspondence.txt')
    print(vowel_indices)

    dict_frames = ali_to_dict(file='./tri4_ali.txt')

    # count
    count_frames(dict_frames)

    vowel_dictionary = longest_vowel(dict_frames)
    frames_to_timestamps(vowel_dictionary)
    make_list_clips(vowel_dictionary)