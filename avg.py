import numpy as np
import pandas as pd
import csv


bow_train = open('bow.train.libsvm',"r")
bow_test = open('bow.test.libsvm',"r")
bow_eval = open('bow.eval.anon.libsvm',"r")
misc_train=pd.read_csv("misc-attributes-train.csv")
misc_test=pd.read_csv("misc-attributes-test.csv")
misc_eval=pd.read_csv("misc-attributes-eval.csv")

# misc_train['defendant_gender'].replace([1,2,3],['female','male',''],inplace=True)
#
misc_train['defendant_gender'].replace(1, 'female',inplace=True)
misc_train['defendant_gender'].replace(2, 'male',inplace=True)
misc_train['defendant_gender'].replace(3, 'indeterminate',inplace=True)
#
# misc_train['victim_genders'].replace(int(1), 'female',inplace=True)
# misc_train['victim_genders'].replace(int(2), 'male',inplace=True)
# misc_train['victim_genders'].replace(int(3), 'indeterminate',inplace=True)

def csv(data):
    
    row_list=[]
    for f in data:
        words = f.split()
        x_dict={}
        for j in range(len(words)):
            if j==0:
                if int(words[0])==1:
                    x_dict[0] = int(words[0])
                else:
                    x_dict[0] = -1



            else:
                col, val = [int(s) for s in words[j].split(':')]
                x_dict[col]=val
        row_list.append(x_dict)

    df = pd.DataFrame.from_dict(row_list)
    df = df.fillna(0)
    return df


bow_train_df=csv(bow_train)
bow_test_df=csv(bow_test)
bow_eval_df=csv(bow_eval)

test_cols=list(bow_test_df.columns)
for i in range(0,10001):
    if  i not in test_cols:
        bow_test_df[i]=[0.0 for i in range(len(bow_test_df))]

bow_train_df = bow_train_df.reindex(sorted(bow_train_df.columns), axis=1)
bow_train_df[10001]=misc_train['num_victims']
bow_test_df = bow_test_df.reindex(sorted(bow_test_df.columns), axis=1)
bow_test_df[10001]=misc_test['num_victims']
#print(bow_train_df.head())
#print(bow_test_df.head())


eval_cols=list(bow_eval_df.columns)
for i in range(0,10001):
    if  i not in eval_cols:
        bow_eval_df[i]=[0.0 for i in range(len(bow_eval_df))]

bow_eval_df = bow_eval_df.reindex(sorted(bow_eval_df.columns), axis=1)
bow_eval_df[10001]=misc_eval['num_victims']
#print(bow_eval_df.head())

# print(bow_train_df[0])
# print(bow_train_df[10001])
print("corrleatiob ntw label and age" ,bow_train_df[0].corr(bow_train_df[10001]))

# bow_test_csv.to_csv('bow_test.csv',index=False,header=False)
# bow_eval_csv.to_csv('bow_eval.csv',index=False,header=False)







def predict(data,w,b):
    y_pred=[]
    for j in range(len(data)):



        if np.dot(np.transpose(w), data[j, 1:]) + b < 0:
            y_pred.append(-1)
        else:
            y_pred.append(1)

    return y_pred

def avg_perceptron(data,lr,e):

    aw=np.zeros(data.shape[1]-1)
    ab=0
    np.random.seed(89)
    update=0
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    b = np.random.uniform(-0.01, 0.01)
    epoch_dict={}
    for i in range(0, e):
        acc = 0
        counter=0

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

            aw=aw+w
            ab=ab+b
            counter+=1

        aw=aw/counter
        ab=ab/counter
        epoch_list=[]
        epoch_list.append(aw.tolist())
        epoch_list.append(ab)
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



epoch_accuracies=avg_perceptron(bow_train_df.to_numpy(),1,10)
#
# Max values
max_acc=max_vals(epoch_accuracies)[-1]
max_w=max_vals(epoch_accuracies)[0]
max_b=max_vals(epoch_accuracies)[1]
#
print("Accuracy on Train",max_acc)
print("Updates on Train is",max_vals(epoch_accuracies)[2])


test_accuracy=accuracy(bow_test_df.to_numpy(),max_w,max_b)
print("Accuracy on Test",test_accuracy)
predict_vals=predict(bow_eval_df.to_numpy(),max_w,max_b)
predict_vals_1= [0 if i ==-1 else 1 for i in predict_vals ]


# writing the data into the file
predict_df=pd.DataFrame(columns = ['example_id', 'label'])

predict_df['example_id']=list(range(0,5250))
predict_df['label']=predict_vals_1
predict_df.to_csv('predict_avg',index=False)


