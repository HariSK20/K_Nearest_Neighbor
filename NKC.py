import random		#allows use of random 
from scipy.spatial import distance # for the euc() function

def euc(a,b):		#this is magnitude of distance when considering each thing as points in graph 
	return distance.euclidean(a,b)

# this is the classifier class
class KNN():
	x_train = []
	y_train=[]
	def fit(self, x,y):        #converts the dataset to point
		self.x_train = x
		self.y_train = y

	def closest(self, row):   #finding a point closest to test point
		best_dist = euc(row, self.x_train[0])
		best_ind = 0
		for i in range(1, len(self.x_train)):
				dist = euc(row,self.x_train[i])  #get the distance of a point 
				if dist < best_dist:  
					best_dist = dist
					best_ind = i
		return self.y_train[best_ind]

	def tie_check(self, l):		#if we cant find a single closest point, we try for other points
		labels = []
		for i in l:
			labels.append(self.y_train[i[1]])
		set_label = list(set(labels))
		mrks = []
		for i in set_label:
			mrks.append([labels.count(i),i])
		mrks.sort()
		return max(mrks, key = lambda x: x[1])

	def closestk(self,row):				#finds the closest group
		dists=[]
		for i in range(len(self.x_train)):
			dists.append([euc(row,self.x_train[i]),i])
		dists.sort(key = lambda x: x[0])
		least = [dists[0]]
		for i in range(1,len(self.x_train)-1):
			if dists[i][0] == dists[i+1][0]:
				least.append(dists[i])
				continue
			if dists[i][0] == dists[i-1][0]:
				least.append(dists[i])
				continue
			e = self.tie_check(least)
			break
		return e[1]

	def predict(self,x):		#considers every point in the test samples and tries to find the closest group to the test point
		predicts = []
		for row in x:
#				label = random.choice(self.y_train)
#				label = self.closest(row)
				label = self.closestk(row)
				predicts.append(label)
		return predicts

# by Harishankar S Kumar @HariSK20