import pandas as pd
from word2number import w2n
from nltk.corpus import wordnet


train =  pd.read_csv('misc-attributes-train.csv')
defendants_gender = list(train['defendant_gender'])
num_victims = list(train['num_victims'])
offense_category = list(train['offence_category'])
offense_subcategory = list(train['offence_subcategory'])


defendants_age = []
for i,e in enumerate(train['defendant_age']):
    e = e.strip("(  ) -")
    if e != "not known":
        e = e.replace("years","")
        e = e.replace("about", "")
        e = e.replace("age","")
        e = e.replace("of", "" )
        e = e.replace("old", "")
        e = e.strip()
        if e.find(" ") >= 0:
            temp = e.split(" ")

            e = '-'.join(temp)
        syns = wordnet.synsets(e.strip())

        e = syns[0].lemmas()[0].name()
        if e.find("-") >= 0:
            temp = e.split("-")

            e = ' '.join(temp)
        defendants_age.append(w2n.word_to_num(e.strip()))
    else:
        defendants_age.append(int(0))
# print("**")
mean =  int(sum(defendants_age)/sum(1 for x in defendants_age if x > 0))

#print(mean)
for i in range(len(defendants_age)):
    if defendants_age[i] == 0:
        defendants_age[i] = mean




df = pd.DataFrame(list(zip(defendants_age, defendants_gender,num_victims,offense_category,offense_subcategory)),  columns =["defendant_age", "defendant_gender","num_victims","offence_category","offence_subcategory"])
df.to_csv('misc_train_preprocess.csv' ,index = False)