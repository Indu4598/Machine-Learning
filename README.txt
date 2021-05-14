Please install the following libraries:
 word2num


After installing the packages run the following commands to generate csv files which are required for margin perceptron and neural network.

1. python3 glove.py->  Produces glove_train.csv, glove_test.csv, and gloe_eval.csv.
2. python3 temp1.py -> Produces misc_train_preprocess.csv. Contains like feautres like defandants_age, number of victims, defedants' gender, offence Category, offence Sub-Category.
3. python3 temp2.py -> Produces misc_test_preprocess.csv. Contains like feautres like defandants_age, number of victims, defedants' gender, offence Category, offence Sub-Category.
4. python3 temp3.py -> Produces misc_eval_preprocess.csv. Contains like feautres like defandants_age, number of victims, defedants' gender, offence Category, offence Sub-Category.



Submission1:

      python3 avg.py
      Average Perceptron: Produces "predict.csv" . Requires no special libraries on CADE Machine.


Submission2:

       python3 margin.py
       Margin Perceptron : On glove data and misc attributes. 


       Neural Network: using Keras:For this submission please open the .ipynb extension.



Submission3:

      python3 svm.py -> produces predict_svm.csv files.
      python3 logistic.py 
      python3 Ensemble.py


 Note: Commented cross validation part in svm ang logistic, and directly used the hyper-parameters produced by cross validation. Reason: kernel kills the process.



All the data files should be in this project directory. The ipynb should be launched from the project directory.

Data required: bow, glove and miscellaneous

 
