from collections import defaultdict
import numpy as np
from itertools import product

class Graph:
	def __init__(self, adjacencyMatrix, data):
		self.adjM = adjacencyMatrix
		self.data = data

	def generateTabularCPD(self):
		pass
		

class Preprocess:
    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.rawData = []

        f = open(self.dataFile)
        lines = f.readlines()
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
	
    def _generateDDCPD(self, child, parents):
        resultRows = [dataRow for dataRow in self.data for attributeIndex in parents if dataRow[attributeIndex] == parents[attributeIndex]]
        childValueCounts = defaultdict(lambda: 0)
        for resultRow in resultRows:
            childValueCounts[resultRow[child]] += 1

        _cpd = {}
        denom = len(resultRows)
        for uniqueChildValue in childValueCounts:	
            _cpd[uniqueChildValue] = childValueCounts[uniqueChildValue]/float(denom)

        return _cpd

    def flatten(self, bla):
        output = []
        for item in bla:
            output += flatten(item) if hasattr (item, "__iter__") or hasattr (item, "__len__") else [item]
        return output

    def combinations(self, discreteParents):
        a = set(self.data[:,discreteParents[0]])
        for i in range(1, len(discreteParents))
            a = product(a, set(self.data[:,discreteParents[i]]))
    
        res = []
        for i in a:
            res.append(flatten(i))

        return res

    # continous parent to discrete child
    def _generateCDCPD(self, child, parents):
        continousParents = [i for i in parents if self.attrType[i] == 'continous']
        discreteParents = [i for i in parents if self.attrType[i] == 'discrete']

        discreteParentValues = combinations(discreteParents)
        for i in discreteParentValues:
            cols = {}
            for j in range(len(discreteParents)):
                cols[discreteParents[j]] = i[j]
            # select rows with column values matching each of the combination for parent variables
            X = self.data[np.logical_and.reduce([self.data[:, k] == cols[k] for k in cols])][:, continousParents]
            y = self.data[:, child]
            clf = svm.SVC()
            clf.fit(X, y)
	
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
	#a = Preprocess('a')
	pass
