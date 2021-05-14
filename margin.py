import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import keras
from sklearn.preprocessing import  LabelEncoder



glove_train = pd.read_csv('glove_train.csv')
glove_test = pd.read_csv('glove_test.csv')
glove_eval = pd.read_csv('glove_eval.csv')
train_misc= pd.read_csv('misc_train.csv')
test_misc = pd.read_csv('misc_train.csv')
eval_misc = pd.read_csv('misc_train.csv')

number = LabelEncoder()
train_misc['offence_category'] = number.fit_transform(train_misc['offence_category'].astype('str'))
test_misc['offence_category'] = number.fit_transform(test_misc['offence_category'].astype('str'))
eval_misc['offence_category'] = number.fit_transform(eval_misc['offence_category'].astype('str'))
train_misc['offence_subcategory'] = number.fit_transform(train_misc['offence_subcategory'].astype('str'))
test_misc['offence_subcategory'] = number.fit_transform(test_misc['offence_subcategory'].astype('str'))
eval_misc['offence_subcategory'] = number.fit_transform(eval_misc['offence_subcategory'].astype('str'))
train_misc['defendant_gender'] = number.fit_transform(train_misc['defendant_gender'].astype('str'))
test_misc['defendant_gender'] = number.fit_transform(test_misc['defendant_gender'].astype('str'))
eval_misc['defendant_gender'] = number.fit_transform(eval_misc['defendant_gender'].astype('str'))




glove_train = glove_train.replace(to_replace={'0': {1: 1,0:-1}})
glove_test = glove_test.replace(to_replace={'0': {1: 1,0:-1}})
glove_eval = glove_eval.replace(to_replace={'0': {1: 1,0:-1}})

# print(glove_train.head())
# print(glove_test.head())
# print(glove_eval.head())
# #
# print(misc_train.head())
# print(misc_test.head())
# print(misc_eval.head())
predictors = list(train_misc.columns)
#print(predictors)
train_misc[predictors] = train_misc[predictors]/train_misc[predictors].max()
test_misc[predictors] = test_misc[predictors]/test_misc[predictors].max()
eval_misc[predictors] = eval_misc[predictors]/eval_misc[predictors].max()


train  = glove_train
test = glove_test
eval = glove_eval


train['defendant_age'] = train_misc['defendant_age']
train['defendant_gender'] = train_misc['defendant_gender']
train['num_victims'] = train_misc['num_victims']
train['offence_category'] = train_misc['offence_category']
train['offence_subcategory'] = train_misc['offence_subcategory']


test['defendant_age'] = test_misc['defendant_age']
test['defendant_gender'] = test_misc['defendant_gender']
test['num_victims'] = test_misc['num_victims']
test['offence_category'] = test_misc['offence_category']
test['offence_subcategory'] = test_misc['offence_subcategory']

eval['defendant_age'] = eval_misc['defendant_age']
eval['defendant_gender'] = eval_misc['defendant_gender']
eval['num_victims'] = eval_misc['num_victims']
eval['offence_category'] = eval_misc['offence_category']
eval['offence_subcategory'] = eval_misc['offence_subcategory']



print(train.head())

# train  = glove_train.append(misc_train)


# train['defendant_age'] = misc_train['defendant_age']
# train['defendant_gender'] = misc_train['defendant_gender']
# train['num_victims'] = misc_train['num_victims']
# train['offence_category'] = misc_train['offence_category']
# train['offence_subcategory'] = misc_train['offence_subcategory']
#
# print(train.head())
# print(len(train.columns))





print("****Margin Perceptron*******")

def simple_perceptron(data,lr,mu,e):

    np.random.seed(7)
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    b = np.random.uniform(-0.01, 0.01)
    epoch_dict={}
    update=0
    for i in range(0, e):
        acc = 0

        np.random.shuffle(data)
        # print(len(data))
        for j in range(len(data)):

            if data[j,0]*(np.dot(np.transpose(w), data[j, 1:]) + b) <= mu:
                w = w + lr * data[j, 0] * data[j, 1:]
                b = b + lr * data[j, 0]
                update+=1
            else:
                acc+=1


        lr=lr/(i+1)


        epoch_list=[]
        epoch_list.append(w.tolist())
        epoch_list.append(b)
        epoch_list.append(update)
        epoch_list.append(acc/len(data)*100)
        epoch_dict[i]=epoch_list


    return epoch_dict





def max_vals(epoch_accuracies):
    max_list=[]
    max_acc = 0
    max_w = []
    max_b = 0
    for key, value in epoch_accuracies.items():
        if max_acc < epoch_accuracies[key][-1]:
            max_acc = epoch_accuracies[key][-1]
            max_w = epoch_accuracies[key][0]
            max_b = epoch_accuracies[key][1]
            max_u=epoch_accuracies[key][2]

    max_list.append(max_w)
    max_list.append(max_b)
    max_list.append(max_u)
    max_list.append(max_acc)
    return max_list

predictors = list(train_misc.columns)
#print(predictors)






def accuracy(data,w,b):
    acc=0
    for j in range(len(data)):



        if np.dot(np.transpose(w), data[j, 1:]) + b <= 0:
            y_pred = -1
        else:
            y_pred = 1
        if y_pred == data[j, 0]:
            acc+=1

    return acc/len(data)*100






# Training the data













# train1 = f1.append([f2,f3,f4],ignore_index=True)
#
# max_dict=simple_perceptron(train1.to_numpy(),0.1)
# max_values=max_vals(max_dict)
# t1_acc = accuracy(f5.to_numpy(),max_values[0],max_values[1])
# print(t1_acc)
#
# train1 = f1.append([f2,f5,f4],ignore_index=True)
#
# max_dict=simple_perceptron(train1.to_numpy(),0.1)
# max_values=max_vals(max_dict)
# t1_acc = accuracy(f3.to_numpy(),max_values[0],max_values[1])
# print(t1_acc)



# learning_rate=[1,0.1,0.01]
# mu=[1,0.1,0.01]
# acc_list=[]
# #
# dict_val={}
# max_cv_acc=0
# lr_l=[]
# for i in learning_rate:
#     mu_l=[]
#     for j in mu:
#
#         train1 = f1.append([f2, f3, f4], ignore_index=True)
#         max_dict = simple_perceptron(train1.to_numpy(), i, j,10)
#         max_values = max_vals(max_dict)
#         t1_acc = accuracy(f5.to_numpy(), max_values[0], max_values[1])
#
#         train2 = f1.append([f3, f4, f5], ignore_index=True)
#         max_dict = simple_perceptron(train2.to_numpy(), i, j,10)
#         max_values = max_vals(max_dict)
#         t2_acc = accuracy(f2.to_numpy(), max_values[0], max_values[1])
#
#         train3 = f2.append([f3, f4, f5], ignore_index=True)
#         max_dict = simple_perceptron(train3.to_numpy(), i,j, 10)
#         max_values = max_vals(max_dict)
#         t3_acc = accuracy(f1.to_numpy(), max_values[0], max_values[1])
#
#         train4 = f2.append([f3, f1, f5], ignore_index=True)
#         max_dict = simple_perceptron(train4.to_numpy(), i, j,10)
#         max_values = max_vals(max_dict)
#         t4_acc = accuracy(f4.to_numpy(), max_values[0], max_values[1])
#
#         train5 = f2.append([f4, f1, f5], ignore_index=True)
#         max_dict = simple_perceptron(train5.to_numpy(), i,j, 10)
#         max_values = max_vals(max_dict)
#         t5_acc = accuracy(f3.to_numpy(), max_values[0], max_values[1])
#
#         # print(t1_acc, t2_acc, t3_acc, t4_acc, t5_acc)
#         avg_acc = (t1_acc + t2_acc + t3_acc + t4_acc + t5_acc) / 5
#
#
#         dict_val[(i, j)] = avg_acc
#
#
#
# m=0
# for key,value in dict_val.items():
#     # print(key,value)
#     if m<value:
#         m=value
#         lr,mu=key
#
# print("Hyper Parameter Learning Rate=",lr,"Hyper Parameter Margin=",mu," accuracy = ",m)
#



epoch_accuracies=simple_perceptron(train.to_numpy(),0.1,0.01,20)
#
# Max values
max_acc=max_vals(epoch_accuracies)[-1]
max_w=max_vals(epoch_accuracies)[0]
max_b=max_vals(epoch_accuracies)[1]
print("Accuracy on Train for Margin Perceptron",max_acc)
print("updates in margin",max_vals(epoch_accuracies)[2])
# Test the data
test_accuracy=accuracy(test.to_numpy(),max_w,max_b)
print("Accuracy on Test for Margin Perceptron",test_accuracy)

def predict(data,w,b):
    y_pred=[]
    for j in range(len(data)):



        if np.dot(np.transpose(w), data[j, 1:]) + b < 0:
            y_pred.append(-1)
        else:
            y_pred.append(1)

    return y_pred

predict_vals=predict(eval.to_numpy(),max_w,max_b)
#print(predict_vals[0:10])
predict_vals_1= [0 if i ==-1 else 1 for i in predict_vals ]


# writing the data into the file
predict_df=pd.DataFrame(columns = ['example_id', 'label'])

predict_df['example_id']=list(range(0,5250))
predict_df['label']=predict_vals_1

predict_df.to_csv('predict_mar.csv',index=False)



