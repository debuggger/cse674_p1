

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
		lines = open(self.dataFile).readlines()
		self.numAttrs = len(lines[0].split(','))
		self.map = [{} for attributeIndex in range(self.numAttrs)]

		for dataRow in lines:
			dataSample = dataRow.split(',')
			self.rawData.append([attributeVal.strip() for attributeVal in dataSample])
			for attributeIndex in range(len(dataSample)):
				self.updateMap(attributeIndex, dataSample[attributeIndex])

		self.generateValueMappings()
		self.compactifyData()

	def updateMap(self, attributeIndex, rawVal):
		val = rawVal.strip()
		if not self.map[attributeIndex].has_key(val):
			self.map[attributeIndex][val] = 1 
		else:
			self.map[attributeIndex][val] += 1 

	def generateValueMappings(self):
		self.valueMap = [{} for i in range(self.numAttrs)]
		for attributeIndex in range(self.numAttrs):
			uniqueVals = self.map[attributeIndex].keys()
			for uniqueVal in uniqueVals:
				self.valueMap[attributeIndex][uniqueVal.lower()] = uniqueVals.index(uniqueVal)
	
	def compactifyData(self):
		self.data = []
		for row in self.rawData:
			self.data.append([self.valueMap[attributeIndex][row[attributeIndex].lower()] for attributeIndex in range(len(row))])
	
	def _getCPD(self, child, parents):
		resultRows = [dataRow for dataRow in self.data for attributeIndex in parents if dataRow[attributeIndex] == parents[attributeIndex]]
		childValueCounts = {}
		for resultRow in resultRows:
			key = resultRow[child]
			if not childValueCounts.has_key(key):
				childValueCounts[key] = 1
			else:
				childValueCounts[key] += 1

		cpd = {}
		denom = len(resultRows)
		for uniqueChildValue in childValueCounts:	
			cpd[uniqueChildValue] = childValueCounts[uniqueChildValue]/float(denom)

		return cpd
	
	def getCPD(self, child, parents):
		newParents = {}
		for attributeIndex in parents:
			parentValue = parents[attributeIndex]
			newParents[attributeIndex] = self.valueMap[attributeIndex][parentValue.lower()]

		_cpd = self._getCPD(child, newParents)

		cpd = {}
		for enumVal in _cpd:
			val = self.reverseValueMap(child, enumVal)
			cpd[val] = _cpd[enumVal]

		return (cpd, _cpd)
	
	def reverseValueMap(self, attributeIndex, enum):
		items = self.valueMap[attributeIndex].items()
		for item in items:
			if item[1] == enum:
				return item[0]
	

if __name__ == '__main__':
	#a = Preprocess('a')
	pass
