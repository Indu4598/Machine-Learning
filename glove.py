import numpy as np
import pandas as pd


bow_train = open('glove.train.libsvm',"r")
bow_test = open('glove.test.libsvm',"r")
bow_eval = open('glove.eval.anon.libsvm',"r")

def csv(data):
    row_list=[]
    for f in data:
        words = f.split()
        x_dict={}
        for j in range(len(words)):
            if j==0:
                x_dict[0] = int(words[0])
            else:
                col, val = [s for s in words[j].split(':')]
                x_dict[int(col)]=float(val)
        row_list.append(x_dict)

    df = pd.DataFrame.from_dict(row_list)
    return df


glove_train_df=csv(bow_train)
glove_test_df=csv(bow_test)
glove_eval_df=csv(bow_eval)

# test_cols=list(bow_test_df.columns)
# for i in range(0,10001):
#     if  i not in test_cols:
#         bow_test_df[i]=[0.0 for i in range(len(bow_test_df))]
#
#
# bow_test_df = bow_test_df.reindex(sorted(bow_test_df.columns), axis=1)
# print(bow_test_df.head())
#
# eval_cols=list(bow_eval_df.columns)
# for i in range(0,10001):
#     if  i not in eval_cols:
#         bow_eval_df[i]=[0.0 for i in range(len(bow_eval_df))]
#
# bow_eval_df = bow_eval_df.reindex(sorted(bow_eval_df.columns), axis=1)
# print(bow_eval_df.head())

# bow_test_csv.to_csv('bow_test.csv',index=False,header=False)
# bow_eval_csv.to_csv('bow_eval.csv',index=False,header=False)





def simple_perceptron(data,lr,e):

    np.random.seed(98)
    w = np.random.uniform(0, 0, size=data.shape[1] - 1)
    b = np.random.uniform(0, 0)
    epoch_dict={}
    update=0
    for i in range(0, e):
        acc = 0

        np.random.shuffle(data)
        # print(len(data))
        for j in range(len(data)):

            if np.dot(np.transpose(w), data[j, 1:]) + b <= 0:
                y_pred = -1
            else:
                y_pred = 1
            if y_pred != int(data[j, 0]):
                w = w + lr * data[j, 0] * data[j, 1:]
                b = b + lr * data[j, 0]
                update+=1
            else:
                acc += 1



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



def accuracy(data,w,b):
    acc=0
    for j in range(len(data)):



        if np.dot(np.transpose(w), data[j, 1:]) + b <= 0:
            y_pred = -1
        else:
            y_pred = 1
        if y_pred == int(data[j, 0]):
            acc+=1

    return acc/len(data)*100








#
# epoch_accuracies=simple_perceptron(bow_train_df.to_numpy(),0.9,10)
# #
# # Max values
# max_acc=max_vals(epoch_accuracies)[-1]
# max_w=max_vals(epoch_accuracies)[0]
# max_b=max_vals(epoch_accuracies)[1]
#
# print("Accuracy on Train for Simple Perceptron",max_acc)
# print("Updates on Train is",max_vals(epoch_accuracies)[2])
#
#
# test_accuracy=accuracy(bow_test_df.to_numpy(),max_w,max_b)
# print("Accuracy on Test for Simple Perceptron",test_accuracy)




# print(glove_train_df.isna().sum().sum())
# print(glove_test_df.isna().sum().sum())
# print(glove_eval_df.isna().sum().sum())

glove_train_df.to_csv('glove_train.csv',index=False)
glove_test_df.to_csv('glove_test.csv',index=False)
glove_eval_df.to_csv('glove_eval.csv',index=False)