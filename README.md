# Eigenfaces-for-Face-Recognition
LFW(Labeled faces in the wild) dataset is used in this problem. The data set has two parts- data and images. Total 1194 data points and 1194 images exist. Each image is of resolution 62x47 which accounts for 2914 features for each data point. Firstly, PCA is applied and features are reduced to 100. Both 2D and 3D plots of the images have been obtained after applying t-SNE. Red , green and blue colours denote Colin Powell, George W Bush and Tony Blair respectively. In 2D, red and green data points are more closely packed amongst themselves.
K-neighbours classifier is used here for classification. The parameter 'n_neighbours' is set to 1. The attached image contains first 20 eigen faces. The importance of eigen faces decreases downwards and rightwards. Also, it has been computed that around 80.12 % variance is explained by first 32 principal components.

![I1](https://github.com/gurpreet-singh-5000/Eigenfaces-for-Face-Recognition/blob/master/Images/Eigenfaces.png)
  

