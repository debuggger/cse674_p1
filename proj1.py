from collections import defaultdict
import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import random
#from matplotlib import pyplot as plt
import math
# ignore DeprecateWarnings by sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

attr = {
		0: {'name': 'age' , 'type':'continuous'}, 
		1: {'name': 'class_worker' , 'type':'discrete'}, 
		2: {'name': 'ind_code' , 'type':'discrete'}, 
		3: {'name': 'occ_code' , 'type':'discrete'}, 
		4: {'name': 'edu' , 'type':'discrete'}, 
		5: {'name': 'wage' , 'type':'continuous'}, 
		6: {'name': 'enroll_edu' , 'type':'discrete'}, 
		7: {'name': 'marital_stat' , 'type':'discrete'}, 
		8: {'name': 'maj_ind_code' , 'type':'discrete'}, 
		9: {'name': 'maj_occ_code' , 'type':'discrete'}, 
		10: {'name': 'race' , 'type':'discrete'}, 
		11: {'name': 'hispanic' , 'type':'discrete'}, 
		12: {'name': 'sex' , 'type':'discrete'}, 
		13: {'name': 'labor_un' , 'type':'discrete'}, 
		14: {'name': 'unemployment' , 'type':'discrete'}, 
		15: {'name': 'full_part_time' , 'type':'discrete'}, 
		16: {'name': 'cap_gain' , 'type':'continuous'}, 
		17: {'name': 'cap_loss' , 'type':'continuous'}, 
		18: {'name': 'dividend' , 'type':'continuous'}, 
		19: {'name': 'tax_filer' , 'type':'discrete'}, 
		20: {'name': 'reg_prev_res' , 'type':'discrete'}, 
		21: {'name': 'state_prev_res' , 'type':'discrete'}, 
		22: {'name': 'house_family_Stat' , 'type':'discrete'}, 
		23: {'name': 'household_summ' , 'type':'discrete'}, 
		24: {'name': 'instance_wt' , 'type':'continuous'}, 
		25: {'name': 'mig_msa' , 'type':'discrete'}, 
		26: {'name': 'mig_reg' , 'type':'discrete'}, 
		27: {'name': 'mig_within_reg' , 'type':'discrete'}, 
		28: {'name': 'live_1year' , 'type':'discrete'}, 
		29: {'name': 'mig_prev_sun' , 'type':'discrete'}, 
		30: {'name': 'num_per_worked' , 'type':'continuous'}, 
		31: {'name': 'fam_un_18' , 'type':'discrete'}, 
		32: {'name': 'country_father' , 'type':'discrete'}, 
		33: {'name': 'country_mother' , 'type':'discrete'}, 
		34: {'name': 'country_self' , 'type':'discrete'}, 
		35: {'name': 'citizenship' , 'type':'discrete'}, 
		36: {'name': 'own_self_emp' , 'type':'discrete'}, 
		37: {'name': 'questionnaire' , 'type':'discrete'}, 
		38: {'name': 'veteran_ben' , 'type':'discrete'}, 
		39: {'name': 'weeks_worked' , 'type':'continuous'}, 
		40: {'name': 'year' , 'type':'discrete'}, 
		41: {'name': 'income_class' , 'type':'discrete'} 
	
	}

def init_network():
	network = np.zeros((len(attr), len(attr)))
	network[0, [1, 6, 7, 8, 9]] = 1
	network[1, [14, 15, 22]] = 1
	network[4, [0, 9]] = 1
	network[6, [8, 14, 39]] = 1
	network[7, [22, 23]] = 1
	network[8, [1, 9, 14]] = 1
	network[9, [12, 39, 41]] = 1
	network[11, [10, 29]] = 1
	network[12, 22] = 1
	network[14, 39] = 1
	network[15:19, 41] = 1
	network[20, [21, 28, 29]] = 1
	network[22, 23] = 1
	network[25, 21] = 1
	network[26, [25, 27]] = 1
	network[27, [20, 25]] = 1
	network[29, 21] = 1
	network[31, 23] = 1
	network[32, [10, 11, 34]] = 1
	network[33, [10, 32, 34]] = 1
	network[34, [4, 10, 35]] = 1
	network[35, [11, 26]] = 1
	network[36, 41] = 1

	return network

def defaultLambda():
	return 0

def rem(n):
	return np.where(np.sum(n, (1))> 0)[0]

def topoSort(network):
	n = np.array(network)
	t = []

	dead = set(np.where(np.sum(n, 0) == 0)[0]).intersection(set(np.where(np.sum(n, 1) == 0)[0]))
	remaining = set(attr.keys()) - dead

	while np.sum(n) > 0:
		for i in range(network.shape[0]):
			if sum(n[:,i]) == 0 and sum(n[i, :]) > 0:
				t.append(i)
				n[i, :] = 0
				remaining.remove(i)
	for i in remaining:
		t.append(i)

	return t

def bfs(network):
	topoSort(network)
	traversal = []
	q = []

	visited = [False for i in range(network.shape[0])]
	for i in range(network.shape[1]):
		if (sum(network[:,i]) == 0) and sum(network[i, :]) > 0:
			visited[i] = True
			traversal.append(i)
			q.append(i)
	while len(q) > 0:
		vertex = q.pop(-1)
		neighbours = np.where(network[vertex, :] == 1)[0]
		for v in neighbours:
			if not visited[v]:
				q.append(v) 
				traversal.append(v)
				visited[v] = True

	return traversal
		

class Sample:
	def __init__(self, cpd, network):
		self.cpd = cpd
		self.network = network
		self.indNodes = self.getIndependentNodes()
		self.traversal = topoSort(self.network)
		self.deadNodes = [i for i in allNodes if sum(network[:, i]) == 0 and sum(network[i,:]) == 0]

	def getIndependentNodes(self):
		ind = []
		for i in range(self.network.shape[1]):
			if (sum(self.network[:,i]) == 0) and (sum(self.network[i, :]) > 0):
				ind.append(i)
		return ind
		
	def selectSampleValue(self, node, cpd):
		if attr[node]['type'] == 'discrete':
			if len(cpd) == 0:
				return -1
			prob = random.uniform(0.0, 1.0)
			j = 0
			s = cpd[0][1]

			while (prob > s):
				j += 1
				s += cpd[j][1]

			return cpd[j][0]
		else:
			if cpd['variance'] == 0.0:
				return cpd['mean']
			return np.random.normal(cpd['mean'], math.sqrt(cpd['variance']))

	def getKey(self, node, sample):
		parents = np.where(self.network[:, node] == 1)[0] 
		discreteParents = [i for i in parents if attr[i]['type'] == 'discrete']
		continuousParents = [i for i in parents if attr[i]['type'] == 'continuous']

		discreteParents.sort()
		if len(continuousParents) == 0:
			key = [sample[i] for i in discreteParents]
			return (tuple(key), None)
		elif len(discreteParents) == 0:
			return ('continuous', np.array([sample[i] for i in continuousParents]))
		else:
			key = [sample[i] for i in discreteParents]
			return (tuple(key), np.array([sample[i] for i in continuousParents]))


	def getSample(self):
		gotAValidSample = False

		count = 0
		while not gotAValidSample:
			traversal = self.traversal[:]
			sample = [-1 for i in range(self.network.shape[1])]
			count += 1
			gotAValidSample = True
			for node in self.indNodes:
				traversal.remove(node)
				cpd = self.cpd[node]['independent'].getFullProbTable()
				sample[node] = self.selectSampleValue(node, cpd)

			while sample.count(-1) > len(self.deadNodes):
				node = traversal.pop(0)
				key, X = self.getKey(node, sample)
				cpd = self.cpd[node][key].getFullProbTable(X)
				
				sampledValue = self.selectSampleValue(node, cpd)
				if sampledValue == -1:
					gotAValidSample = False
					break
				
				sample[node] = sampledValue

		return sample



class SpecialDiscreteChild:
	def __init__(self, classLabel):
		self.classLabel = classLabel
		self.clf = []
		if len(classLabel) == 1:
			self.clf.append(1.0)
	
	def predict_proba(self, X):
		return [self.clf for x in X]
		

class CPD:
	def __init__(self, parentType, X, y, childType='discrete'):
		self.childType = childType
		self.parentType = parentType
		self.classLabels = list(set(y))
		self.classLabels.sort()

		if childType == 'discrete':
			if self.parentType == 'hybrid' or self.parentType == 'continuous':
				if len(set(y)) > 1:
					clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
					clf.fit(X, y)
				else:
					clf = SpecialDiscreteChild(set(y))
			
				self.obj = clf
			
			else:
				if len(set(y)) > 0:
					counts = stats.itemfreq(y)
					denom = sum([i[1] for i in counts])
					probTable = [(i[0], i[1]/denom) for i in counts]
					self.obj = defaultdict(defaultLambda, probTable)
				else:
					self.obj = defaultdict(defaultLambda)
				'''
				else:
					self.obj = SpecialDiscreteChild(set(y))
				'''

		elif childType == 'continuous':
			if self.parentType == 'hybrid' or self.parentType == 'continuous':
				degree= len(continuousParents)*2
				model_lr = make_pipeline(PolynomialFeatures(degree), Ridge())
				model_lr.fit(X, y)
				self.obj = model_lr
				self.lr_variance = 1-q.score(X,y)
				# y_pre = model.predict(x)
			else:
				self.mean=np.mean(y)
				self.variance=np.var(y, dtype=np.float64)


	def getFullProbTable(self, conditionalVariables=None):
		if self.childType == 'discrete':
			if self.parentType == 'hybrid' or self.parentType == 'continuous':
				clfResults = self.obj.predict_proba(conditionalVariables)
				return zip(self.classLabels, clfResults[0])
			else:
				res = self.obj.items()
				res.sort(key = lambda x: x[0])
				return res

		else:
			if self.parentType == 'hybrid' or self.parentType == 'continuous':
				lr_predict = self.obj.predict(conditionalVariables)
				res={"mean":lr_predict, "variance":self.lr_variance}
				return res
			else:
				res={"mean":self.mean, "variance":self.variance}
				return res


	  
		

class Preprocess:
	def __init__(self, dataFile, network, N=199523):
		self.dataFile = dataFile
		self.network = network
		self.rawData = []
		self.cpd = {}

		f = open(self.dataFile)
		lines = f.readlines()[:N]
		f.close()


		self.numAttrs = len(lines[0].split(','))
		self.map = [defaultdict(defaultLambda) for attributeIndex in range(self.numAttrs)]

		for dataRow in lines:
			dataSample = dataRow.split(',')
			self.rawData.append([attributeVal.strip() for attributeVal in dataSample])

			for attributeIndex in range(len(dataSample)):
				self.map[attributeIndex][dataSample[attributeIndex].strip()] += 1 

		self.generateValueMappings()
		self.compactifyData()
	
	def generateValueMappings(self):
		self.valueEnumMap = [defaultdict(defaultLambda) for i in range(self.numAttrs)]
		self.enumValueMap = [defaultdict(defaultLambda) for i in range(self.numAttrs)]
		for attributeIndex in range(self.numAttrs):
			if attr[attributeIndex]['type'] == 'discrete':
				uniqueVals = self.map[attributeIndex].keys()
				for uniqueVal in uniqueVals:
					self.valueEnumMap[attributeIndex][uniqueVal.lower()] = uniqueVals.index(uniqueVal)
					self.enumValueMap[attributeIndex][uniqueVals.index(uniqueVal)] = uniqueVal.lower()

	def compactifyData(self):
		#self.data is  a numpy matrix representation of the data
		self.data = np.array(np.zeros(len(self.rawData[0])))
		for row in self.rawData:
			self.data = np.vstack([self.data, [self.valueEnumMap[attributeIndex][row[attributeIndex].lower()] if attr[attributeIndex]['type'] == 'discrete' else float(row[attributeIndex]) for attributeIndex in range(len(row))]])
	
	def flatten(self, bla):
		output = []
		for item in bla:
			output += self.flatten(item) if hasattr (item, "__iter__") or hasattr (item, "__len__") else [item]
		return output

	def combinations(self, discreteParents):
		res = []
		a = set(self.data[:,discreteParents[0]])
		
		if len(discreteParents) == 1:
			return [tuple([i]) for i in a]

		for i in range(1, len(discreteParents)):
			a = product(a, set(self.data[:,discreteParents[i]]))
	
		for i in a:
			res.append(self.flatten(i))

		return res

	# continuous parent to discrete child
	def _generateDiscreteChildCPD(self, child, parents):
		discreteParents = [i for i in parents if attr[i]['type'] == 'discrete']
		continuousParents = [i for i in parents if attr[i]['type'] == 'continuous']
			   
		discreteParents.sort()
		continuousParents.sort()

		self.cpd[child] = defaultdict(defaultLambda)
		childDomain = set(self.data[:, child]) 

		if len(discreteParents) > 0:
			discreteParentValues = self.combinations(discreteParents)
			for i in discreteParentValues:
				cols = {}
				for j in range(len(discreteParents)):
					cols[discreteParents[j]] = i[j]
			   
				# select rows with column values matching each of the combination for parent variables
				y = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, child]
				if len(continuousParents) > 0:
					X = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, continuousParents]
					self.cpd[child][tuple(i)] = CPD('hybrid', X, y)
				else:
					self.cpd[child][tuple(i)] = CPD('discrete', None, y, 'discrete')

		elif len(continuousParents) > 0:
			X = self.data[:, continuousParents]
			y = self.data[:, child]
			self.cpd[child]['continuous'] = CPD('continuous', X, y, 'discrete')

		else:
			y = self.data[:, child]
			self.cpd[child]['independent'] = CPD('independent', None, y, 'discrete')

	def _generateContinuousChildCPD(self, child, parents):
		discreteParents = [i for i in parents if attr[i]['type'] == 'discrete']
		continuousParents = [i for i in parents if attr[i]['type'] == 'continuous']
			   
		discreteParents.sort()
		continuousParents.sort()

		self.cpd[child] = defaultdict(defaultLambda)
		childDomain = set(self.data[:, child])

		if len(discreteParents) > 0:
			discreteParentValues = self.combinations(discreteParents)
			for i in discreteParentValues:
				cols = {}
				for j in range(len(discreteParents)):
					cols[discreteParents[j]] = i[j]

				y = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, child]
				
				if len(continuousParents) > 0:
					# discrete and continuous
					X = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, continuousParents]
					self.cpd[child][tuple(i)] = CPD('hybrid', X, y, 'continuous')
				else:
					self.cpd[child][tuple(i)] = CPD('discrete', None, y, 'continuous')

		elif len(continuousParents) > 0:
			# only continuous
			X = self.data[:, continuousParents]   
			y = self.data[:, child]
			self.cpd[child][tuple(i)] = CPD('continuous', X, y, 'continuous')
		else:
			# no parent
			y = self.data[:, child]
			self.cpd[child]['independent'] = CPD('independent', None, y, 'continuous')

		
	def getCPD(self, child, parents):
		newParents = {}
		for attributeIndex in parents:
			parentValue = parents[attributeIndex]
			newParents[attributeIndex] = self.valueEnumMap[attributeIndex][parentValue.lower()]

		_cpd = self._getCPD(child, newParents)

		cpd = {}
		for enumVal in _cpd:
			val = self.enumValueMap[child][enumVal]
			cpd[val] = _cpd[enumVal]

		return (cpd, _cpd)

	
def getParents(node, network):
	return np.where(network[:, node] == 1)[0]

def train(filename, nodes, network):
	p = Preprocess(filename, network)
	for i in nodes:
		if attr[i]['type'] == 'discrete':
			p._generateDiscreteChildCPD(i, getParents(i))
		else:
			p._generateContinuousChildCPD(i, getParents(i))
			pass
		filename = str(i)+'.p'
		pickle.dump( p.cpd[i], open( filename, "wb" ) )

def inference(cpd, nodes, inferenceVariables, evidence):
	
	s = Sample(cpd, init_network())
	
	samples = np.array([np.zeros(42)])

	while samples.shape[0] < 5:
		sample = s.getSample()
		match = True
		for i in evidence:
			if sample[i] != evidence[i]:
				match = False
				break
		if match:
			samples = np.vstack([samples, sample])


	inferenceResult = samples[np.logical_or.reduce([np.logical_and.reduce([samples[:, i] == inferenceVariables[j][i] for i in inferenceVariables[j]]) for j in range(len(inferenceVariables))])]


	inferenceProb  = inferenceResult.shape[0]/float(samples.shape[0])
	print inferenceProb

	return inferenceProb


if __name__ == '__main__':
	np.set_printoptions(threshold='nan')
	network = init_network()
	allNodes = attr.keys()
	deadNodes = [i for i in allNodes if sum(network[:, i]) == 0 and sum(network[i,:]) == 0]
	nodes = list(set(allNodes) - set(deadNodes))
	nodes.sort()
	
	train('census-income.data', nodes, network)

	cpd = {}
	
	for i in nodes:
		cpd[i] = pickle.load(open(str(i)+'.p'))

	inference(cpd, nodes, [{12: 1}], {4: 8})
	inference(cpd, nodes, [{4: 0}, {4: 3}, {4: 4}, {4: 6}, {4: 10}, {4: 12}, {4: 13}, {4: 5}, {4: 15}], {14: 0})
	inference(cpd, nodes, [{41: 0}], {25: 0, 27: 8})
	inference(cpd, nodes, [{41: 0}], {})
	inference(cpd, nodes, [{41: 0}], {12: 1})
	inference(cpd, nodes, [{15: 0}, {15: 4}, {15: 5}], {12: 1})

