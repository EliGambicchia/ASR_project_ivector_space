import numpy as np
import pandas as pd


mfcc_mat1 = pd.DataFrame(np.random.randn(50, 13))
mfcc_mat2 = pd.DataFrame(np.random.randn(50, 13))


# for index, row in mfcc_mat1.iterrows():
#     if row[12] > 0:
print(mfcc_mat1.shape)
mfcc_mat1.drop(mfcc_mat1[mfcc_mat1[12] < 0].index, inplace=True)
print(mfcc_mat1.shape)


# print(mfcc_mat1.shape)
# print(mfcc_mat2.shape)

mfccs_vector1 = pd.DataFrame(mfcc_mat1.mean(0)).T
mfccs_vector2 = pd.DataFrame(mfcc_mat2.mean(0)).T

print(mfccs_vector1.shape)
print(mfccs_vector2.shape)

# mfccs_vector1['utterance'] = 'sent1'
# mfccs_vector2['utterance'] = 'sent2'
#
# print(mfccs_vector1.shape)
# print(mfccs_vector2.shape)

#
df_all_mfccs = pd.concat([mfccs_vector1, mfccs_vector2], axis=0)
print(df_all_mfccs)
print(df_all_mfccs.shape)

