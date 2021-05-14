import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math




# bow_train = open('glove.train.libsvm',"r")
# bow_test = open('glove.test.libsvm',"r")
# bow_eval = open('glove.eval.anon.libsvm',"r")
#
# def csv(data):
#     row_list=[]
#     for f in data:
#         words = f.split()
#         x_dict={}
#         for j in range(len(words)):
#             if j==0:
#                 if int(words[0])==1:
#                     x_dict[0] = int(words[0])
#                 else:
#                     x_dict[0] = -1
#
#
#
#             else:
#                 col, val = [int(s) for s in words[j].split(':')]
#                 x_dict[col]=val
#         row_list.append(x_dict)
#
#     df = pd.DataFrame.from_dict(row_list)
#     df = df.fillna(0)
#     return df

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




#print(glove_train_df.head())
#print(glove_test_df.head())
#print(glove_eval_df.head())


bow_train_numpy = glove_train_df.to_numpy()
bow_test_numpy = glove_test_df.to_numpy()
bow_eval_numpy = glove_eval_df.to_numpy()

# print(bow_train_numpy.shape)
# print(bow_test_numpy.shape)
# print(bow_eval_numpy.shape)

def log_reg(data,lr,sigma):
    lr_init = lr
    np.random.seed(80)
    dict_acc = {}
    dict_loss = {}
    prev_loss = float("inf")
    #print(lr)
    #print(sigma)

    w = np.random.uniform(-0.01, 0.01, size=data.shape[1]-1)
    for i in range(50):

        np.random.shuffle(data)

        for row in range(len(data)):
            y = data[row, 0]
            x = data[row, 1:]

            try:
                w = w - lr * (((2 * w) / sigma) + (((-y * x) * (math.exp(-y * np.dot(np.transpose(w), x)))) / (1 + math.exp(-y * np.dot(np.transpose(w), x)))))
            except:
                pass

        dict_acc[i] = accuracy(data, w), w

        loss = (2 / sigma) * np.dot(w, w)
        for row in range(len(data)):
            y = data[row, 0]
            x = data[row, 1:]
            loss = loss + math.log(1 + math.exp(-y * np.dot(np.transpose(w), x)))
        dict_loss[i] = loss
        if abs(prev_loss - loss) < 0.1:
            break
        lr = lr_init / (1 + i)
        prev_loss = loss

    return dict_acc, dict_loss

def lr_cv(data,lr,sigma):
    lr_init = lr
    dict_acc = {}
    prev_loss= float("inf")
    np.random.seed(70)
    w = np.random.uniform(-0.01, 0.01, size=data.shape[1]-1)
    for i in range(10):
        np.random.shuffle(data)
        for j in range(len(data)):
            y = data[j, 0]
            x = data[j, 1:]


            try:
                w = w - lr * (((2 * w) / sigma) + (((-y * x) * (math.exp(-y * np.dot(np.transpose(w), x)))) / (1 + math.exp(-y * np.dot(np.transpose(w), x)))))
            except:
                pass

        dict_acc[i] = accuracy(data, w), w

        loss = (2 / sigma) * np.dot(w, w)
        for row in range(len(data)):
            y = data[row, 0]
            x = data[row, 1:]
            x = np.append(x, 1)
            try:
                loss = loss + math.log(1 + math.exp(-y * np.dot(np.transpose(w), x)))
            except:
                pass
        if abs(prev_loss - loss) < 2:
            break
        lr = lr_init / (1 + i)

    return dict_acc


def predict(data,w):
    y_pred = []
    for i in range(len(data)):
        x = data[i, 1:]
        if np.dot(w,x)<=0:
            y_pred.append(-1)
        else:
            y_pred.append(1)
    return y_pred


def accuracy(data,w):
    acc=0
    for j in range(len(data)):
        x = data[j, 1:]



        if np.dot(w, x)  <= 0:
            y_pred = -1
        else:
            y_pred = 1
        if y_pred == int(data[j, 0]):
            acc+=1
    return acc/len(data)*100




lr = [ 0.1, 0.01, 0.001, 0.0001, 0.00001]
sig = [ 1, 10, 100, 1000, 10000]


# train_numpy = train.to_numpy()
# test_numpy = test.to_numpy()
#
#
#
#
#



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





max_cv_acc=0




def max_acc(D):
    max_a=0
    max_w=0
    for key,value in D.items():
        if value[0]>max_a:
            max_a=value[0]
            max_w=value[1]

    return max_a,max_w


#for i in sig:
    #for j in lr:
        #print(i,j)
        #train1 = f1.append([f2, f3, f4], ignore_index=True)
        #t1_d = lr_cv(train1.to_numpy(), j, i)
        #t1_a,t1_w = max_acc(t1_d)
        #t1_acc = accuracy(f5_numpy, t1_w)


        #train2 = f2.append([f1, f3, f5], ignore_index=True)
        #t2_d = lr_cv(train2.to_numpy(),j,i)
        #t2_a,t2_w = max_acc(t2_d)
        #t2_acc = accuracy(f4_numpy, t2_w)


        #train3 = f5.append([f1, f2, f4], ignore_index=True)
        #t3_d = lr_cv(train3.to_numpy(), j, i)
        #t3_a, t3_w = max_acc(t3_d)
        #t3_acc = accuracy(f3_numpy, t3_w)


        #train4 = f4.append([f1, f3, f4], ignore_index=True)
        #t4_d = lr_cv(train4.to_numpy(), j, i)
        #t4_a,t4_w = max_acc(t4_d)
        #t4_acc = accuracy(f2_numpy, t4_w)


        #train5 = f4.append([f2, f3, f5], ignore_index=True)
        #t5_d = lr_cv(train5.to_numpy(), j, i)
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

w_d,loss_dict = log_reg(glove_train_df.to_numpy(),0.0001,10000)
a,w = max_acc(w_d)
print("Train_Acc=",accuracy(glove_train_df.to_numpy(),w))
print("Test_Acc=",accuracy(glove_test_df.to_numpy(),w))


predict_vals=predict(glove_eval_df.to_numpy(),w)
# print(predict_vals[0:10])
predict_vals_1= [0 if i ==-1 else 1 for i in predict_vals ]


# writing the data into the file
predict_df=pd.DataFrame(columns = ['example_id', 'label'])

predict_df['example_id']=list(range(0,5250))

predict_df['label'] = predict_vals_1
predict_df.to_csv('predict_LR.csv',index=False)

# plt.plot(loss_dict.keys(), loss_dict.values())
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title("SVM LOSS")
# plt.show()
