from collections import defaultdict
import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import random
#from matplotlib import pyplot as plt

# ignore DeprecateWarnings by sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
	network[0, [1, 6, 7, 8, 9] = 1
	network[1, [14, 15, 22]] = 1
	network[4, [0, 9]] = 1
	network[6, [8, 14, 29]] = 1
	network[7, [22, 23]] = 1
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


def bfs(network):
	traversal = []
	q = []
	visited = [False for i in range(network.shape[0])]
	for i in range(network.shape[1]):
		if (sum(network[:,i]) == 0) and sum(network[i, :]) > 0:
			visited[i] = True
			traversal.append(i)
			q.insert(0, i)
	while len(q) > 0:
		vertex = q.pop()
		neighbours = np.where(network[vertex, :] == 1)[0]
		for v in neighbours:
			if not visited[v]:
				q.insert(0, v) 
				traversal.append(v)
				visited[v] = True

	return traversal
		

class Sample:
	def __init__(self, p):
		self.p = p
		self.network = p.network
		self.indNodes = self.getIndependentNodes()

	def getIndependentNodes(self):
		ind = []
		for i in self.network.shape[1]:
			if (sum(self.network[:,i]) == 0) and (sum(self.network[i, :]) > 0):
				ind.append(i)
		return ind
		
	def selectSampleValue(self, node, cpd):
		if attr[node]['type'] == 'discrete':
			prob = random.uniform(0.0, 1.0)
			j = 0
			s = cpd[0][1]

			while (prob > s):
				j += 1
				s += cpd[j][1]

			return cpd[j][0]
		else:
			return np.random.normal(cpd['mean'], cpd['variance'])

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
		traversal = bfs(self.network)
		sample = [-1 for i in range(self.p.shape[1])]
		for i in self.indNodes:
			traversal.remove(i)
			cpd = p.cpd[i]['independent'].getFullProbTable()
			sample[i] = self.selectSampleValue(cpd)

		while sample.count(-1) > 0:
			node = traversal.pop(0)
			key, X = self.getKey(node, sample)
			cpd = p.cpd[node][key].getFullProbTable(X)
			sample[i] = self.selectSampleValue(node, cpd)



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
		self.childType
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
				unique, counts = np.unique(y, return_counts=True)
				denom = sum(counts)
				probTable = zip(unique, [i/denom for i in counts])
				self.obj = defaultdict(lambda: 0, probTable)

		elif childType = 'continuous':
			if self.parentType == 'hybrid' or self.parentType == 'continuous':
				pass
			else:
				pass


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
				pass
			else:
				pass


      
		

class Preprocess:
    def __init__(self, dataFile, N=199523, cols = 42):
        self.dataFile = dataFile
        self.rawData = []
        self.cpd = {}

        f = open(self.dataFile)
        lines = f.readlines()[:N]
        f.close()


        self.numAttrs = len(lines[0].split(','))
        self.map = [defaultdict(lambda: 0) for attributeIndex in range(self.numAttrs)]

        for dataRow in lines:
            dataSample = dataRow.split(',')
            self.rawData.append([attributeVal.strip() for attributeVal in dataSample])

            for attributeIndex in range(len(dataSample)):
                self.map[attributeIndex][dataSample[attributeIndex].strip()] += 1 

        self.generateValueMappings()
        self.compactifyData()
    
    def generateValueMappings(self):
		self.valueEnumMap = [defaultdict(lambda: 0) for i in range(self.numAttrs)]
		self.enumValueMap = [defaultdict(lambda: 0) for i in range(self.numAttrs)]
		for attributeIndex in range(self.numAttrs):
			uniqueVals = self.map[attributeIndex].keys()
			for uniqueVal in uniqueVals:
				self.valueEnumMap[attributeIndex][uniqueVal.lower()] = uniqueVals.index(uniqueVal)
				self.enumValueMap[attributeIndex][uniqueVals.index(uniqueVal)] = uniqueVal.lower()

    def compactifyData(self):
        #self.data is  a numpy matrix representation of the data
		self.data = np.array(np.zeros(len(self.rawData[0])))
		for row in self.rawData:
			self.data = np.vstack([self.data, [self.valueEnumMap[attributeIndex][row[attributeIndex].lower()] for attributeIndex in range(len(row))]])
	
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

		self.cpd[child] = defaultdict(lambda: 0.0)
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

    def _generateCCCPD(self, child, parents):
        continuousParents = [i for i in parents]
        X = self.data[:, continuousParents]   
        y = self.data[:, child]

        xp = np.linspace(-5, 50, 100)
        all_residual=[]
        all_p=[]
        for i in range(1,15,1):
            p, residual, _, _, _ = np.polyfit(X[:,0],y[:,0], i, full=True)
            all_residual.append(residual[0])
            all_p.append(p)

        best_fit=all_p[all_residual.index(min(all_residual))-1]
        print best_fit

    def _generateDCCPD(self, child, parents):
        discreteParents = [i for i in parents]
        discreteParentValues = self.combinations(discreteParents)
        
        gaus_for_all=[]
        for i in discreteParentValues:
            cols = {}
            for j in range(len(discreteParents)):
                cols[discreteParents[j]] = i[j]

            x = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])]
            if len(x[:, child])!=0:
                y=x[:, child]
        
                cols["mean"]=np.mean(y)
                cols["var"]=np.var(y, dtype=np.float64)
                gaus_for_all.append(cols)
        print gaus_for_all

        
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

	
def getParents(node):
	return np.where(network[:, node] == 1)

if __name__ == '__main__':
    #lines_to_read=100
    #a=Preprocess('/home/karan/Downloads/census-income.data', lines_to_read)
    #a=Preprocess('/home/karan/Downloads/census-income.data')
    #a._generateDCCPD([0], [1,8])
    #a._generateCCCPD([0],[39])
	allNodes = attr.keys()
	deadNodes = [i for i in allNodes if sum(network[:, i]) == 0 and sum(network[i,:]) == 0]

	nodes = list(set(allNodes) - set(deadNodes))
	nodes.sort()

	p = Preprocess('census-income.data', 5000)
	for i in nodes:
		if attr[i]['type'] == 'discrete':
			p._generateDiscreteChildCPD(i, getParents(i))
		else:
			pass


    pass
