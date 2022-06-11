# ASR project: I-Vector Extraction and Visualisation Analysis for Multi-Accent Speech Recognition

### From Abstract of Thesis Project:

One of the challenges faced by the automatic speech recognition (ASR) community is to build accent-robust systems that can successfully handle multiple accents. Using the open-source Mozilla Common Voice corpus (v. 6.1), we explored the effects of a popular speaker adaptation technique (i-vectors) on multi-accent speech recognition. An initial exploratory analysis showed gains in the system’s performance (5.3% absolute word error rate improvement) when the acoustic model was trained with a time-delay neural network (TDNN) model, compared to architectures solely based on Gaussian mixture models (GMMs). I-vectors were used to augment the acoustic input and the effects of i) speaker diversity and ii) size of the training set on the i-vector ex- tractor training were investigated. Training the i-vector extractor with a more diverse pool of speakers, as well as more training data, resulted in a further 4.2% in absolute error reduction. Finally, principal component analysis (PCA) was used on both the acoustic feature space and the i-vector space to see how the accents related to one another acoustically, displaying more significant clusters when PCA was combined with linear discriminant analysis (LDA).


### Filtering process

The dataset comprised clips of the recorded utterances and a metadata file with infor- mation such as utterance ID, speaker ID, accent, gender, etc. As expected, the dataset was not balanced in terms of accent coverage: 52% was American English, 17% British English and the residual 32% covered the other 14 accents. Therefore, the first steps aimed at filtering the data and selecting good proportions of speech for each accent.


### PCA

In this study, Principal Component Analysis (PCA) was performed on the i-vectors to confirm that these additional feature vectors do carry accent-related information as claimed in the literature. In exploratory data analysis, PCA is used to reduce the dimensions of the dataset for interpretability by retaining most of the information. It works by finding a linear combination that maximises the variance of new uncorrelated data points. PCA is often combined with other techniques (LDA and TSNE).
