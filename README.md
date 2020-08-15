# Eigenfaces-for-Face-Recognition
LFW dataset is used in this problem. The data set has two parts- data and images. Total 1194 data points and 1194 images exist. Each image is of resolution 62x47 which accounts for 2914 features for each data point. Firstly, PCA is applied and features are reduced to 100. Both 2D and 3D plots have been obtained. Red , green and blue colours denote Colin Powell, George W Bush and Tony Blair respectively. In 2D, red and green data points are more closely packed amongst themselves. Three classification reports have been attatched here. K-neighbours classifier is used here for classification. The parameter 'n_neighbours' is set to 1. Fig.(c) corresponds to case 1 when PCA is applied before splitting the data into train and test sets. Fig.(d) is the case 2 when PCA is applied to test and train sets separately. Case 3 is shown in fig.(e), where PCA is not applied. The best results are obtained in case 1. Case 3 is slightly behind than case 1, but it is way ahead than case 2. Case 1 leading over case 3 is obvious as the dimensional reduction is better in case 1. But case 2 leading over case 3 is interesting. Fig.(f) contains first 20 eigen faces. The importance of eigen faces decreases downwards and rightwards. Also, it has been computed that around 80.12 % variance is explained by first 32 principal components. Classification report for that case is in fig.(g). Performance drops only a bit compared to the case which had all the 100 principal components.


(a) (b) (c)
(d) (e) (f)
(g)
Figure 1: Problem 1
