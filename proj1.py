from collections import defaultdict
import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

# ignore DeprecateWarnings by sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Graph:
	def __init__(self, adjacencyMatrix, data):
		self.adjM = adjacencyMatrix
		self.data = data

	def generateTabularCPD(self):
		pass

class CPD:
    def __init__(self, parentType, X, y):
        self.parentType = parentType
        
        if self.parentType == 'hybrid' or self.parentType == 'continous':
            clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
            clf.fit(X, y)
            self.obj = clf
        else:
            unique, counts = np.unique(y, return_counts=True)
            self.obj = dict(zip(unique, counts))
            denom = sum(self.obj.values())

            for i in self.obj:
                self.obj[i] /= denom

    def getProb(X, c):
        if self.parentType == 'hybrid' or self.parentType == 'continous':
            return self.obj.predict_proba(X)[c]
        else:
            return self.obj[c]
        
		

class Preprocess:
    def __init__(self, dataFile, N=199523):
        self.dataFile = dataFile
        self.rawData = []
        self.cpd = {}
        self.attr = {
                0: {'name': 'age' , 'type':'continous'}, 
                1: {'name': 'class_worker' , 'type':'discrete'}, 
                2: {'name': 'ind_Code' , 'type':'discrete'}, 
                3: {'name': 'occ_code' , 'type':'discrete'}, 
                4: {'name': 'edu' , 'type':'discrete'}, 
                5: {'name': 'wage' , 'type':'continous'}, 
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
                16: {'name': 'cap_gain' , 'type':'continous'}, 
                17: {'name': 'cap_loss' , 'type':'continous'}, 
                18: {'name': 'dividend' , 'type':'continous'}, 
                19: {'name': 'tax_filer' , 'type':'discrete'}, 
                20: {'name': 'reg_prev_res' , 'type':'discrete'}, 
                21: {'name': 'state_prev_res' , 'type':'discrete'}, 
                22: {'name': 'house_family_Stat' , 'type':'discrete'}, 
                23: {'name': 'household_summ' , 'type':'discrete'}, 
                24: {'name': 'instance_wt' , 'type':'continous'}, 
                25: {'name': 'mig_msa' , 'type':'discrete'}, 
                26: {'name': 'mig_reg' , 'type':'discrete'}, 
                27: {'name': 'mig_within_reg' , 'type':'discrete'}, 
                28: {'name': 'live_1year' , 'type':'discrete'}, 
                29: {'name': 'mig_prev_sun' , 'type':'discrete'}, 
                30: {'name': 'num_per_worked' , 'type':'continous'}, 
                31: {'name': 'fam_un_18' , 'type':'discrete'}, 
                32: {'name': 'country_father' , 'type':'discrete'}, 
                33: {'name': 'country_mother' , 'type':'discrete'}, 
                34: {'name': 'country_self' , 'type':'discrete'}, 
                35: {'name': 'citizenship' , 'type':'discrete'}, 
                36: {'name': 'own_self_emp' , 'type':'discrete'}, 
                37: {'name': 'questionnaire' , 'type':'discrete'}, 
                38: {'name': 'veteran_ben' , 'type':'discrete'}, 
                39: {'name': 'weeks_worked' , 'type':'continous'}, 
                40: {'name': 'year' , 'type':'discrete'}, 
                41: {'name': 'income_class' , 'type':'discrete'} 
            
            }

        f = open(self.dataFile)
        lines = f.readlines()
        f.close()

        with open(self.dataFile) as data_file:
            lines = [next(data_file) for x in xrange(N)]

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

    # continous parent to discrete child
    def _generateDiscreteChildCPD(self, child, parents):
        discreteParents = [i for i in parents if self.attr[i]['type'] == 'discrete']
        continousParents = [i for i in parents if self.attr[i]['type'] == 'continous']
               
        self.cpd[child] = {}

        if len(discreteParents) > 0:
            discreteParentValues = self.combinations(discreteParents)
            for i in discreteParentValues:
                cols = {}
                for j in range(len(discreteParents)):
                    cols[discreteParents[j]] = i[j]
               
                # select rows with column values matching each of the combination for parent variables
                y = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, child]
                if len(continousParents) > 0:
                    X = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, continousParents]
                    self.cpd[child][tuple(i)] = CPD('hybrid', X, y)
                else:
                    self.cpd[child][tuple(i)] = CPD('discrete', None, y)

        else:
            X = self.data[:, continousParents]
            y = self.data[:, child]
            self.cpd[child]['continous'] = CPD('continous', X, y)

    def _generateCCCPD(self, child, parents):
        continousParents = [i for i in parents]
        X = self.data[:, continousParents]   
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
	

if __name__ == '__main__':
    #lines_to_read=100
    #a=Preprocess('/home/karan/Downloads/census-income.data', lines_to_read)
    #a=Preprocess('/home/karan/Downloads/census-income.data')
    #a._generateDCCPD([0], [1,8])
    #a._generateCCCPD([0],[39])
    pass
