<h1> Old Baileys </h1>

<h3>Overview of the Project: </h3>
<p align = "left">Old Bailey Proceedings Online project made available a fully searchable and digitized collection of approximately 197,000 trail proceedings from the year 1676 to 1772. **The objective of this project is to explore different classfiers to  predict decision of trails i.e guilty(1) or not-guilty(1) present in the Old Bailey Proceedings Online project.** The transcribed dialogue of a trail is available in three formats as highlighted in the Table 1. Along with the dialogue of the trail, defendant’s age. 
defendant’s gender, number of victims, victims’ gender. offence category, and offence subcategory details are available to develop classification systems. These details from here on are referred as miscellaneous attributes.  </p>
<h4>Table1: Formats of dialuges transcribed during the trail</h4>

 | Formats of Data | 
 |:-------------:  |
 | Bag of Words    | 
 | Glove           |   
 | Tf-idf          |   

<h4>Table 2: Algorithms used to develop classification systems and accuracy achieved </h4>


| Tables        | Format of Dialouge used          | Evaluation Accuracy  |
| ------------- |:-------------:| -----:|
|  Logistic Regression    | Glove | 66.400 %  |
| Ensemble Method      | Glove     |  62.628 %  |
| Neural Network | Bag-of-words    |   67.085 %  |
| Support Vector Machine (SVM)  | Bag-of-words    |   72.876  |
| Margin Perceptron | Glove    |   62.857 %  |
| Average Perceptron| Bag-of-Words   |   68.952 %   |


Of all the six classification systems highlighted above, SVM achieved the highest accuracy. Theoretical understanding of SVM suggests that SVM has better generalization ability and lower risk of over-fitting the data and results achieved in this project also suggests that. However, it is also to be noted that there is lot of scope to implement improvements in all the six classification systems and accuracies might improve significantly.  
