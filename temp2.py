import pandas as pd
from word2number import w2n
from nltk.corpus import wordnet




test =  pd.read_csv('misc-attributes-test.csv')


defendants_gender = list(test['defendant_gender'])
num_victims = list(test['num_victims'])
offense_category = list(test['offence_category'])
offense_subcategory = list(test['offence_subcategory'])

defendants_age = []
for i,e in enumerate(test['defendant_age']):
    # print("age",e)
    # print(i)
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
            # print(temp)
            e = '-'.join(temp)
        syns = wordnet.synsets(e.strip())

        e = syns[0].lemmas()[0].name()
        if e.find("-") >= 0:
            temp = e.split("-")
            # print(temp)
            e = ' '.join(temp)
        # print(e.strip())
        defendants_age.append(w2n.word_to_num(e.strip()))
    else:
        defendants_age.append(int(0))
# print("**")
mean =  int(sum(defendants_age)/sum(1 for x in defendants_age if x > 0))

#subsitute not known with mean value
for i in range(len(defendants_age)):
    if defendants_age[i] == 0:
        defendants_age[i] = mean

# print("**")
# print(len(defendants_age))
# print(len(labels))
# print("****")








#
# print(len(labels))
# print(len(defendants_gender))
# print(len(defendants_age))
# print(len(offense_category))
# print(len(num_victims))
df = pd.DataFrame(list(zip(defendants_age, defendants_gender,num_victims,offense_category,offense_subcategory)),  columns =["defendant_age", "defendant_gender","num_victims","offence_category","offence_subcategory"])
df.to_csv('misc_test_prepeocess.csv',index = False)