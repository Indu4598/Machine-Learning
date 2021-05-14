import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




bow_train = open('bow.train.libsvm',"r")
bow_test = open('bow.test.libsvm',"r")
bow_eval = open('bow.eval.anon.libsvm',"r")

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
#
# bow_train_df = bow_train_df.reindex(sorted(bow_train_df.columns), axis=1)
# bow_test_df = bow_test_df.reindex(sorted(bow_test_df.columns), axis=1)
# bow_eval_df = bow_eval_df.reindex(sorted(bow_eval_df.columns), axis=1)


# print(bow_train_df.head())
# print(bow_test_df.head())
# print(bow_eval_df.head())
#
# bow_train_numpy = bow_train_df.to_numpy()
# bow_test_numpy = bow_test_df.to_numpy()
# bow_eval_numpy = bow_eval_df.to_numpy()
#
# print(bow_train_numpy.shape)
# print(bow_test_numpy.shape)
# print(bow_eval_numpy.shape)


test_cols=list(bow_test_df.columns)
for i in range(0,10000):
    if  i not in test_cols:
        bow_test_df[i]=[0.0 for i in range(len(bow_test_df))]

bow_train_df = bow_train_df.reindex(sorted(bow_train_df.columns), axis=1)
bow_test_df = bow_test_df.reindex(sorted(bow_test_df.columns), axis=1)


eval_cols=list(bow_eval_df.columns)
for i in range(0,10000):
    if  i not in eval_cols:
        bow_eval_df[i]=[0.0 for i in range(len(bow_eval_df))]

bow_eval_df = bow_eval_df.reindex(sorted(bow_eval_df.columns), axis=1)

# print(bow_train_df.head())
# print(bow_test_df.head())
# print(bow_eval_df.head())


bow_train_numpy = bow_train_df.to_numpy()
bow_test_numpy = bow_test_df.to_numpy()
bow_eval_numpy = bow_eval_df.to_numpy()

# print(bow_train_numpy.shape)
# print(bow_test_numpy.shape)
# print(bow_eval_numpy.shape)

def svm_cv(data,C,lr):
    np.random.seed(7)
    lr_init=lr
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1])
    prev=float("inf")
    loss=0
    d={}
    loss_dict={}
    for i in range(0,5):
        #print(i)
        np.random.shuffle(data)
        for j in range(len(data)):
            x = data[j,1:]
            x = np.append(x,1)

            if data[j,0]*np.dot(w,x) <= 1:
                try:
                    w = w * (1 -lr) + lr*C*data[j,0]*x
                except:
                    pass
            else:
                try:
                   w = w * (1 - lr)
                except:
                   pass
        d[i]=w,accuracy(data,w)
        loss = 0.5*np.dot(w,w)
        for j in range(len(data)):
            x = data[j, 1:]
            x = np.append(x, 1)
            loss = loss + C*max(0, 1 - data[j,0]*np.dot(w,x))
        if abs(prev - loss) <250:
            break
        loss_dict[i]=loss
        prev=loss
        lr = lr_init/(i+1)

    return d


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
            x = data[j,1:]
            x = np.append(x,1)

            if data[j,0]*np.dot(w,x) <= 1:
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
        d[i] = w, accuracy(data, w)
        # for j in range(len(data)):
        #     x = data[j, 1:]
        #     x = np.append(x, 1)
        #     loss = loss + C*max(0, 1 - data[j,0]*np.dot(w,x))
        # if abs(prev - loss) <250:
        #     break
        # loss_dict[i]=loss
        # prev=loss
        lr = lr_init/(i+1)

    return loss_dict,d


def predict(data,w):
    y_pred = []
    for i in range(len(data)):
        x = data[i, 1:]
        x = np.append(x, 1)
        if np.dot(w,x)<=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred


def accuracy(data,w):
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






partition = int(len(bow_train_df)/5)
f1= bow_train_df[0:partition].reset_index(drop=True)
f2 = bow_train_df[partition:2*partition].reset_index(drop=True)
f3 = bow_train_df[2*partition:3*partition].reset_index(drop=True)
f4 = bow_train_df[3*partition:4*partition].reset_index(drop=True)
f5 = bow_train_df[4*partition:].reset_index(drop=True)
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





max_cv_acc=0


lr = [0.1,0.01,0.001,0.0001]
C = [1000, 100, 10, 1]



def max_acc(D):
    max_a=0
    max_w=0
    for key,value in D.items():
        if value[1]>max_a:
            max_a=value[1]
            max_w=value[0]

    return max_a,max_w
#
#
#for i in C:
    #for j in lr:
        #print(i,j)
        #train1 = f1.append([f2, f3, f4], ignore_index=True)
        #t1_d = svm_cv(train1.to_numpy(), i, j)
        #t1_a,t1_w = max_acc(t1_d)
        #t1_acc = accuracy(f5_numpy, t1_w)


        #train2 = f2.append([f1, f3, f5], ignore_index=True)
        #t2_d =svm_cv(train2.to_numpy(),i,j)
        #t2_a,t2_w = max_acc(t2_d)
        #t2_acc = accuracy(f4_numpy, t2_w)


        #train3 = f5.append([f1, f2, f4], ignore_index=True)
        #t3_d = svm_cv(train3.to_numpy(), i, j)
        #t3_a, t3_w = max_acc(t3_d)
        #t3_acc = accuracy(f3_numpy, t3_w)


        #train4 = f4.append([f1, f3, f4], ignore_index=True)
        #t4_d = svm_cv(train4.to_numpy(), i, j)
        #t4_a,t4_w = max_acc(t4_d)
        #t4_acc = accuracy(f2_numpy, t4_w)


        #train5 = f4.append([f2, f3, f5], ignore_index=True)
        #t5_d = svm_cv(train5.to_numpy(), i, j)
        #t5_a, t5_w = max_acc(t5_d)
        #t5_acc = accuracy(f1_numpy, t5_w)



        #avg_acc = (t1_acc + t2_acc + t3_acc + t4_acc + t5_acc) / 5


        #if max_cv_acc < avg_acc:
            #max_cv_acc = avg_acc
            #lr_b = j
            #C_b=i


#print("Cross Validation Accuracy= ", max_cv_acc)
#print("best learing rate=", lr_b)
#print("best C=",C_b)

loss_dict,w_d = svm(bow_train_numpy,100,0.0001)
a,w = max_acc(w_d)
print("Train_Acc=",accuracy(bow_train_numpy,w))
print("Test_Acc=",accuracy(bow_test_numpy,w))


predict_vals=predict(bow_eval_df.to_numpy(),w)
# print(predict_vals[0:10])
predict_vals_1= [0 if i ==-1 else 1 for i in predict_vals ]


# writing the data into the file
predict_df=pd.DataFrame(columns = ['example_id', 'label'])

predict_df['example_id']=list(range(0,5250))
predict_df['label'] = predict_vals_1

predict_df.to_csv('predict_SVM.csv',index=False)

