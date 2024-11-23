

#---------------------------------------------
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  g
# variable information 
print(iris.variables) 
#---------------------------------------------This code was created to import the iris dataset (Fisher, 1936) into python by (UCI Machine Learning Repository,https://archive.ics.uci.edu/dataset/53/iris)
