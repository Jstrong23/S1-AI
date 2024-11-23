
import pandas as pd


class AI:
    def decision(self, test_data, SetosaMeans, versicolourMeans, virginicaMeans):
        if (test_data.values[3] > SetosaMeans.values[3] - 0.5) and (test_data.values[3] < SetosaMeans.values[3] + 0.5):
            return ("setosa")
        elif (test_data.values[2] > virginicaMeans.values[2] - 0.3) and (test_data.values[2] < virginicaMeans.values[2] + 0.3):
            return ("virginica")  
        elif (test_data.values[0] > versicolourMeans.values[0] - 0.6) and (test_data.values[0] < versicolourMeans.values[0] + 0.6):
            return ("versicolour")  
        else:
            return ("undetermined")        


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

for x in range (45):
    i = x + 35
    test_data = setosa_test
    if x > 14:
        i += 35
        test_data = versicolour_test
        if x > 29:
            i += 35
            test_data = virginica_test
    output = ai.decision(test_data.loc[i], SetosaMeans, versicolourMeans, virginicaMeans)
    print (x, output)

print (virginicaMeans)
print (versicolourMeans)