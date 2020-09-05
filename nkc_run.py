import random		#allows use of random 
from sklearn import datasets   #getting the dataset
import NKC
iris = datasets.load_iris()  #loading a big dataset

#print(iris.feature_names)
#print(iris.target_names)
x = iris.data
y = iris.target

#print(x)
#print(y)
#print(len(x))

x_train = []
y_train = []
x_test = []
y_test = []
for j in [0,50,100]:
	for i in range(25):
		x_train.append(x[i+j])
		y_train.append(y[i+j])
		x_test.append(x[i+j+25])
		y_test.append(y[i+j+25])

# enter the classifier call here
classi = NKC.KNN()						#classifier call
classi.fit(x_train, y_train)		#train the classifier

cnt =0 
predicts = classi.predict(x_test)
#print(predicts)
for i in range(len(predicts)):
	try:
		if predicts[i] == y_test[i]:
			cnt+=1
	except:
			print(predicts[i])
			print(y_test[i])
#	else:
#		print(i)
print(" Prediction accuracy for iris is :  ", end="")		
print(cnt/len(x_test))



# by Harishankar S Kumar @HariSK20