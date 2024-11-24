import pandas as pd


class AI:
    def decision(self, test_data, SetosaMeans, versicolourMeans, virginicaMeans):
        setosa_score = 0
        versicolour_score = 0
        virginica_score = 0
        total_score = 0
        if (test_data.values[0] > SetosaMeans.values[0] - 0.5) and (test_data.values[0] < SetosaMeans.values[0] + 0.5):
            setosa_score += 1
        if (test_data.values[1] > SetosaMeans.values[1] - 0.5) and (test_data.values[1] < SetosaMeans.values[1] + 0.5):
            setosa_score += 1
        if (test_data.values[2] > SetosaMeans.values[2] - 0.5) and (test_data.values[2] < SetosaMeans.values[2] + 0.5):
            setosa_score += 1
        if (test_data.values[3] > SetosaMeans.values[3] - 0.5) and (test_data.values[3] < SetosaMeans.values[3] + 0.5):
            setosa_score += 1
        if (test_data.values[0] > versicolourMeans.values[0] - 0.5) and (test_data.values[0] < versicolourMeans.values[0] + 0.5):
            versicolour_score += 1
        if (test_data.values[1] > versicolourMeans.values[1] - 0.5) and (test_data.values[1] < versicolourMeans.values[1] + 0.5):
            versicolour_score += 1
        if (test_data.values[2] > versicolourMeans.values[2] - 0.5) and (test_data.values[2] < versicolourMeans.values[2] + 0.5):
            versicolour_score += 1
        if (test_data.values[3] > versicolourMeans.values[3] - 0.5) and (test_data.values[3] < versicolourMeans.values[3] + 0.5):
            versicolour_score += 1
        if (test_data.values[0] > virginicaMeans.values[0] - 0.5) and (test_data.values[0] < virginicaMeans.values[0] + 0.5):
            virginica_score += 1
        if (test_data.values[1] > virginicaMeans.values[1] - 0.5) and (test_data.values[1] < virginicaMeans.values[1] + 0.5):
            virginica_score += 1
        if (test_data.values[2] > virginicaMeans.values[2] - 0.5) and (test_data.values[2] < virginicaMeans.values[2] + 0.5):
            virginica_score += 1
        if (test_data.values[3] > virginicaMeans.values[3] - 0.5) and (test_data.values[3] < virginicaMeans.values[3] + 0.5):
            virginica_score += 1
        
        total_score = setosa_score + versicolour_score + virginica_score

        if setosa_score > versicolour_score and setosa_score > virginica_score:
            return("setosa", int(setosa_score / total_score * 100))
        elif virginica_score > setosa_score and virginica_score > versicolour_score:
            return("virginica", int(virginica_score / total_score * 100))
        elif versicolour_score > setosa_score and versicolour_score > virginica_score:
            return("versicolour", int(versicolour_score / total_score * 100))
        else:
            return("undetermined", 0)
                
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
    prob = 0
    test = "setosa"
    i = x + 35
    test_data = setosa_test
    if x > 14:
        test = "versicolour"
        i += 35
        test_data = versicolour_test
        if x > 29:
            test = "virginica"
            i += 35
            test_data = virginica_test
    output, prob = ai.decision(test_data.loc[i], SetosaMeans, versicolourMeans, virginicaMeans)
    if output == test:
        correct = "Correct"
    else:
        correct = "false"
    print (x, "---", output, "---",prob, "%", "---", correct)
    

