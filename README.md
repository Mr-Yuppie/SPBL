# SPBL
The matlab code for the IJCAI-16 paper "Self-Paced Boost Learning for Classification"

1. Store the data as "Data/fea.mat" (in the "Data/" directory), which has two matrices:
   fea: n-by-d matrix, where each row is the feature of a sample;
   gnd: n-by-1 vector, where each element is the class label index of a sample.

2. Run "Gen_Split.m" to generate the split of traning/validation/test set of the data.
   Set the "train_ratio" and "vali_ratio" variables for the proportions of the training and the validation samples, respectively.

3. (Optional) Run "Gen_Noise" to generate label noise in the training set.
   Set the "n_ratio" variable for the proportion of the noisily labeled samples.

4. Run "SPBLmain.m" to train the SPBL model with the training set and test the learned model on the test set.
