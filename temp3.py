import pandas as pd
from word2number import w2n
from nltk.corpus import stopwords
from nltk.corpus import wordnet



eval =  pd.read_csv('misc-attributes-eval.csv')


defendants_gender = list(eval['defendant_gender'])
num_victims = list(eval['num_victims'])
offense_category = list(eval['offence_category'])
offense_subcategory = list(eval['offence_subcategory'])
defendants_age = []
for i,age in enumerate(eval['defendant_age']):

    age = age.strip("(  ) -")
    if age != "not known":
        age = age.replace("eleven years old and six months", "eleven")
        age = age.replace("11 or 12", "12")
        age = age.replace("years","")
        age = age.replace("Year", "")
        age = age.replace("his", "")
        age = age.replace("about", "")
        age = age.replace("age","")
        age = age.replace("Age", "")
        age = age.replace("of", "" )
        age = age.replace("old", "")
        age = age.replace("of", "")
        age= age.replace("83d","83")
        age = age.replace("eleven years old and six months", "eleven")
        age = age.strip()
        if age.find(" ") >= 0:
            temp = age.split(" ")

            age = '-'.join(temp)
        syns = wordnet.synsets(age.strip())

        if len(syns)==0:
            print(age)

        age = syns[0].lemmas()[0].name()
        if age.find("-") >= 0:
            temp = age.split("-")
            age = ' '.join(temp)
        if age.strip()=="meter":
            defendants_age.append(0)
            continue
        defendants_age.append(w2n.word_to_num(age.strip()))
    else:
        defendants_age.append(int(0))

mean =  int(sum(defendants_age)/sum(1 for x in defendants_age if x > 0))



for i in range(len(defendants_age)):
    if defendants_age[i] == 0:
        defendants_age[i] = mean

# print("**")
# print(len(defendants_age))
# print(len(labels))
# print("****")








# print(len(defendants_gender))
# print(len(defendants_age))
# print(len(offense_category))
# print(len(num_victims))
df = pd.DataFrame(list(zip(defendants_age, defendants_gender,num_victims,offense_category,offense_subcategory)),  columns =["defendant_age", "defendant_gender","num_victims","offence_category","offence_subcategory"])
df.to_csv('misc_eval_preprocess.csv' ,index = False)