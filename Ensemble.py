import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

glove_train = open('glove.train.libsvm',"r")
glove_test = open('glove.test.libsvm',"r")
glove_eval = open('glove.eval.anon.libsvm',"r")

def csv(data):
    row_list=[]
    for f in data:
        words = f.split()
        x_dict={}
        for j in range(len(words)):
            if j==0:
                if int(words[0])==1:
                    x_dict[0]=1
                else:
                    x_dict[0]=-1
            else:
                col, val = [s for s in words[j].split(':')]
                x_dict[int(col)]=float(val)
        row_list.append(x_dict)

    df = pd.DataFrame.from_dict(row_list)
    return df


glove_train_df=csv(glove_train)
glove_test_df=csv(glove_test)
glove_eval_df=csv(glove_eval)




# print(glove_train_df.head())
# print(glove_test_df.head())
# print(glove_eval_df.head())


bow_train_numpy = glove_train_df.to_numpy()
bow_test_numpy = glove_test_df.to_numpy()
bow_eval_numpy = glove_eval_df.to_numpy()



def simple_perceptron(data,lr,e):

    np.random.seed(89)
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1] - 1)
    b = np.random.uniform(-0.01, 0.01)
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

def predict(data,w,b):
    y_pred = []
    for i in range(len(data)):
        x = data[i, 1:]
        if np.dot(w,x) + b <=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred

def predict1(data,w):
    y_pred = []
    for i in range(len(data)):
        x = data[i, 1:]
        if np.dot(w,x) + b <=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred


partition = int(len(glove_train_df)/5)
f1= glove_train_df[0:partition].reset_index(drop=True)
f2 = glove_train_df[partition:2*partition].reset_index(drop=True)
f3 = glove_train_df[2*partition:3*partition].reset_index(drop=True)
f4 = glove_train_df[3*partition:4*partition].reset_index(drop=True)
f5 = glove_train_df[4*partition:].reset_index(drop=True)
# print(f1.head())
# print(f2.head())
# print(f3.head())
# print(f4.head())
# print(f5.head())
f1_numpy = f1.to_numpy()
f2_numpy = f2.to_numpy()
f3_numpy = f3.to_numpy()
f4_numpy = f4.to_numpy()
f5_numpy = f5.to_numpy()
learning_rate=[1,0.1,0.01]

max_cv_acc=0
for i in learning_rate:

    # print(i)

    train1 = f1.append([f2,f3,f4],ignore_index=True)
    max_dict=simple_perceptron(train1.to_numpy(),i,10)
    max_values=max_vals(max_dict)
    t1_acc = accuracy(f5.to_numpy(),max_values[0],max_values[1])


    train2 = f1.append([f3, f4, f5], ignore_index=True)
    max_dict = simple_perceptron(train2.to_numpy(),i,10)
    max_values = max_vals(max_dict)
    t2_acc = accuracy(f2.to_numpy(), max_values[0], max_values[1])


    train3 = f2.append([f3, f4, f5], ignore_index=True)
    max_dict = simple_perceptron(train3.to_numpy(),i,10)
    max_values = max_vals(max_dict)
    t3_acc = accuracy(f1.to_numpy(), max_values[0], max_values[1])


    train4 = f2.append([f3, f1, f5], ignore_index=True)
    max_dict = simple_perceptron(train4.to_numpy(), i,10)
    max_values = max_vals(max_dict)
    t4_acc = accuracy(f4.to_numpy(), max_values[0], max_values[1])


    train5 = f2.append([f4, f1, f5], ignore_index=True)
    max_dict = simple_perceptron(train5.to_numpy(), i,10)
    max_values = max_vals(max_dict)
    t5_acc = accuracy(f3.to_numpy(), max_values[0], max_values[1])

    # print(t1_acc,t2_acc,t3_acc,t4_acc,t5_acc)
    avg_acc=(t1_acc+t2_acc+t3_acc+t4_acc+t5_acc)/5
    # print(avg_acc)

    if max_cv_acc<avg_acc:
        max_cv_acc=avg_acc
        lr=i

print("Hyper Parameter=",lr," accuracy=",max_cv_acc)





w_list=[]
b_list=[]
#building 50 perceptrons
for i in range(50):
    df = glove_train_df.sample(frac=0.1,replace=True,random_state=i).reset_index(drop=True)
    # if i==2:
    #     print(df.iloc[2,:])
    # if i==4:
    #     print(df.iloc[2,:])
    data=df.to_numpy()
    diction=simple_perceptron(data,lr,10)
    w = max_vals(diction)[0]
    b = max_vals(diction)[1]
    w_list.append(w)
    b_list.append(b)




# print(w_list[0])
# print(w_list[-1])
# print(b_list[0])
# print(b_list[-1])
t1 = np.zeros((len(glove_train_df), 50))
t2 = np.zeros((len(glove_test_df), 50))
t3 = np.zeros((len(glove_eval_df),50))


g_train_numpy = glove_train_df.to_numpy()
g_test_numpy = glove_test_df.to_numpy()
g_eval_numpy = glove_eval_df.to_numpy()


for i in range(len(glove_train_df)):
    for j in range(50):
        x= g_train_numpy[i,1:]
        if np.dot(w_list[j],x) + b_list[j] <=0:
            t1[i,j] = -1
        else:
            t1[i,j] = 1

for i in range(len(glove_test_df)):
    for j in range(50):
        x= g_test_numpy[i,1:]
        if np.dot(w_list[j],x) + b_list[j] <=0:
            t2[i,j]=-1
        else:
            t2[i,j]=1

for i in range(len(glove_eval_df)):
    for j in range(50):
        x= g_eval_numpy[i,1:]
        if np.dot(w_list[j],x) + b_list[j] <=0:
            t3[i,j]=-1
        else:
            t3[i,j]=1


#print(t1)
#print(t2)
#print(t3)



def svm(data,C,lr):
    np.random.seed(7)
    lr_init = lr
    d={}
    loss_dict={}
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1])
    prev = float('inf')
    for i in range(0,50):
        #print(i)
        np.random.shuffle(data)
        for j in range(len(data)):
            x = data[j,:]

            if np.dot(w,x) <= 1:
                try:
                    w = w * (1 -lr) + lr*C*data[j,0]*x
                except:
                    pass
            else:
                try:
                    w = w * (1 - lr)
                except:
                    pass
        loss = 0.5*np.dot(w,w)
        d[i] = w, accuracy1(data, w)
        for j in range(len(data)):
            x = data[j, 1:]
            x = np.append(x, 1)
            loss = loss + C*max(0, 1 - data[j,0]*np.dot(w,x))
        if abs(prev - loss) <250:
            break
        loss_dict[i]=loss
        prev=loss
        lr = lr_init/(i+1)

    return loss_dict,d


def predict1(data,w):
    y_pred = []
    for i in range(len(data)):
        x = data[i, 1:]
        x = np.append(x, 1)
        if np.dot(w,x)<=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred


def accuracy1(data,w):
    acc=0
    for j in range(len(data)):
        x = data[j, 1:]
        x = np.append(x, 1)


        if np.dot(np.transpose(w), x)  <= 0:
            y_pred = -1
        else:
            y_pred = 1
        if y_pred == int(data[j, 0]):
            acc+=1

    return acc/len(data)*100



def max_acc(D):
    max_a=0
    max_w=0
    for key,value in D.items():
        if value[1]>max_a:
            max_a=value[1]
            max_w=value[0]

    return max_a,max_w


loss_dict,w_d = svm(t1,100,0.001)
a,w = max_acc(w_d)
#print(w)

print("Train_Acc=",accuracy1(t1,w))
print("Test_Acc=",accuracy1(t2,w))


predict_vals=predict1(t3,w)
# print(predict_vals[0:10])
predict_vals_1= [0 if i ==-1 else 1 for i in predict_vals ]


# writing the data into the file
predict_df=pd.DataFrame(columns = ['example_id', 'label'])

predict_df['example_id']=list(range(0,5250))
predict_df['label'] = predict_vals_1

predict_df.to_csv('predict_ensemble.csv',index=False)
