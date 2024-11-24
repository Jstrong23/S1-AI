
import pandas as pd


class AI:
    def decision(self, test_data, SetosaMeans, versicolourMeans):
        if (test_data.values[3] > SetosaMeans.values[3] - 0.5) and (test_data.values[3] < SetosaMeans.values[3] + 0.5):
            out = "setosa"
        elif (test_data.values[0] > versicolourMeans.values[0] - 0.5) and (test_data.values[0] < versicolourMeans.values[0] + 0.5):
            out = "versicolour" 
        else:
             out = "undetermined"
        return (out)


#create instance of AI
ai = AI()
#---------------------------------------------
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
#print(iris.metadata) 
  
# variable information 
#print(iris.variables) 
#---------------------------------------------This code was created to import the iris dataset (Fisher, 1936) into python by (UCI Machine Learning Repository,https://archive.ics.uci.edu/dataset/53/iris)

#choose training data
data = iris.data.features
species = iris.data.targets
df = pd.DataFrame(data)
df_Target = pd.DataFrame(species)

#chose training data
Setosa_train = df.loc[0:34]
versicolour_train = df.loc[50:74]
virginica_train = df.loc[100:134]


#calculate means of training data
SetosaMeans = Setosa_train.mean()
versicolourMeans = versicolour_train.mean()
virginicaMeans = virginica_train.mean()
#print(SetosaMeans)
#print(versicolourMeans)
#print(virginicaMeans)



#choose test data
setosa_test = df.loc[35:49]
versicolour_test = df.loc[75:99]
virginica_test = df.loc[135:149]
FP = 0
FN = 0
TP = 0
TN = 0
true_array = []
score_array = []

for x in range (45):
    i = x + 35
    test_data = setosa_test
    test = "setosa"
    if x > 14:
        test = "versicolour"
        i += 35
        test_data = versicolour_test
        if x > 29:
            test = "virginica"
            i += 35
            test_data = virginica_test
    
    output = ai.decision(test_data.loc[i], SetosaMeans, versicolourMeans)
    if output == test:
        correct = "true"
        TP += 1
        true_array.append(1)
    elif output == "undetermined" and test == "virginica":
        correct = "true"
        TN += 1
        true_array.append(1)
    elif output == "undetermined":
        correct = "false"
        FN += 1
        true_array.append(0)
    else:
        correct = "false"
        FP += 1
        true_array.append(0)
    print (x, output)

#accuracy
accuracy = (TP+FN)/(TP+TN+FP+FN)
print(accuracy)

#precision
precision = (TP)/(TP+FP)
print(precision)

#recall
recall = (TP)/(TP+FN)
print (recall)

#F1 score
F1Score = 2*((precision*recall)/(precision+recall))
print (F1Score)