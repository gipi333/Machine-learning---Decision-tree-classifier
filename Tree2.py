#=============================================================================================
#=============================================================================================
#                     IDASM102 Machine learning et data mining : projet 1                                    
#=============================================================================================
#=============================================================================================
# Nom: Laboureur
# Prénom: Guillaume
# Date: 15/10/21


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from utils import DT_to_PNG



#===========================
# Importation du dataset                                  
#===========================

cancer= datasets.load_breast_cancer() #Importation du dataset

keys = []

data = []
target = []
target_names= []
DESCR= []
feature_names= []
filename= []
frame = []

listes = [data,target,frame,target_names,DESCR,feature_names,filename]

for key in cancer.keys():
    keys.append(key)
    
for i in range(len(keys)):
    listes[i].append(cancer.get(keys[i])) # Enregistrement de chaque partie du dataset dans une liste 



#=================================================
# Création d'un arbre à 2 et d'un autre à 30 nodes                                  
#=================================================

X_train, X_test, y_train, y_test = train_test_split(data[0], target[0], test_size=0.33, random_state=42) # Division du dataset en une partie test et une partie entrainement   

leaf = [2,30] # Valeurs de max_leaf_nodes
file = ['file1','file2'] #Nom sous lesquels les arbres vont être enregistrés
accuracy_2_30 = []
for i in range(2):
    model1 = []
    model1 = DecisionTreeClassifier(max_leaf_nodes=leaf[i],criterion = "gini")
    model1.fit(X_train, y_train)
    DT_to_PNG(model1, feature_names[0],target_names[0], file[i]) 
    
    
    accuracy_2_30.append(model1.score(X_train, y_train)) 
    accuracy_2_30.append(model1.score(X_test, y_test))
    
print(accuracy_2_30)



#============================
# Metaparameters Selection                                 
#============================


accuracy_train = [] # accuracy for test set
accuracy_test = [] # accuracy for train set
y_predict =[]
for i in range(29):
    model2 = []
    model2 = DecisionTreeClassifier(max_leaf_nodes=2+i,criterion = "gini")
    model2.fit(X_train, y_train)  # Le modèle d'arbre de décision est entraîné à partir du l'ensemble de données d'entrainement
    
    accuracy_test.append(model2.score(X_test, y_test))
    accuracy_train.append(model2.score(X_train, y_train))
     



# Création du graphique
#-----------------------

maxi =  [i for i in range(2,31)] 

plt.figure()
plt.plot(maxi,accuracy_train,linewidth=4.0,label="Ensemble de données d'entrainement")
plt.plot(maxi,accuracy_test,linewidth=4.0,label='Ensemble de données test')

plt.xlabel('Nombre maximum de feuilles',size=25)
plt.ylabel('Précision moyenne',size=25)
plt.title('Précision moyenne en fonction du nombre maximum de feuilles',size=35)
plt.xlim([0,32])
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

plt.legend(fontsize=22, loc='upper left', framealpha=0)










