import spacy,re,os,random,csv
import numpy as np
from nltk.corpus import cess_esp
from nltk.corpus import words as ewords
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer,SnowballStemmer

#fix random seeds!
np.random.seed(666)
torch.manual_seed(666)
random.seed(666)
spacy.util.fix_random_seed(666)

testInput = True
useNew = True
#saved state doesn't work as well for some reason!
savedstate = '/Users/hammond/Desktop/adobo.pt'
epochs = 600
lr = .001
batchsize = 32
coalas = '/Users/hammond/Desktop/coalas-main/corpus/'
#coalas = '/mhdata/coalas-main/corpus/'

#logistic regression or neural net
logreg = False

englishwordsfile = '/usr/share/dict/words'
#more english words from here: https://github.com/dwyl/english-words
ewords2file = '/Users/hammond/Desktop/english-words-master/words.txt'

#load spacy parser
#python -m spacy download es_core_news_md
sp = spacy.load('es_core_news_md',disable=['parser','ner'])

#read in input data
ilines = []
f = open('input.csv','r')
ilines = f.read()
f.close()
ilines = ilines.split('\n')

#read in reference data
rlines = []
with open('reference.csv', 'r') as f:
	csvreader = csv.reader(f,quotechar='"',delimiter=';')
	for row in csvreader:
		rlines.append(row)

#tag input.csv data
isentences = []
iwords = []
print('tagging...\n\tinput data')
for i,line in enumerate(ilines):
	res = sp(line)
	isentences.append(line)
	for word in res:
		#word,pos,sentence
		iwords.append((word,word.pos_,i))

#tag reference data
rsentences = []
rwords = []
print('tagging...\n\treference data')
for i,line in enumerate(rlines):
	res = sp(line[0])
	rsentences.append(line[0])
	for word in res:
		if str(word) in line[1:]:
			status = 'ENG'
		else:
			status = 'O'
		#word,pos,borr,sentence
		rwords.append((word,word.pos_,status,i))

#read in second list of english words
f = open(ewords2file,'r')
ewords2 = f.read()
f.close()

ewords2 = ewords2.lower()
ewords2 = ewords2.split('\n')[:-1]
ewords2 = set(ewords2)

#read in local English words
f = open(englishwordsfile,'r')
english = f.read()
f.close()
english = english.lower()
english = set(english.split('\n'))

#read in NLTK Spanish words
spanish = [w.lower() for w in cess_esp.words()]
spanish = set(cess_esp.words())

#character bigrams
def makebigrammodel(wds):
	#do counts
	bigrams = {}
	bigramtotal = 0
	unigrams = {}
	unigramtotal = 0
	for word in wds:
		word = '#' + word + '#'
		for i in range(1,len(word)):
			bigram = word[i-1:i+1]
			bigramtotal += 1
			unigram = word[i-1]
			unigramtotal += 1
			if unigram in unigrams:
				unigrams[unigram] += 1
			else:
				unigrams[unigram] = 1
			if bigram in bigrams:
				bigrams[bigram] += 1
			else:
				bigrams[bigram] = 1
	#do add-1 smoothing
	for b1 in unigrams:
		for b2 in unigrams:
			if b1+b2 not in bigrams:
				bigrams[b1+b2] = 0
	for bigram in bigrams:
		bigrams[bigram] += 1
		bigramtotal += 1
	#convert to probabilities
	for bigram in bigrams:
		bigrams[bigram] /= bigramtotal
	for unigram in unigrams:
		unigrams[unigram] /= unigramtotal
	#convert to log conditional probabilities
	#p(b|a) = p(ab)/p(a)
	for bigram in bigrams:
		bigrams[bigram] = np.log(bigrams[bigram]/unigrams[bigram[0]])
	return bigrams

#make two bigram models
englishbigrams = makebigrammodel(english)
spanishbigrams = makebigrammodel(spanish)

#sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

#read in coalas data
parts = {}
files = os.listdir(coalas)
for filename in files:
	part = re.sub('\\.conll$','',filename)
	f = open(coalas + filename,'r')
	t = f.read()
	f.close()
	lines = t.split('\n')
	parts[part] = lines

#tag all words
print('tagging...')
allresults = {}
alltags = set()
for part in parts:
	print(f'\t{part}')
	words = []
	borrowed = []
	results = []
	#group words into sentences for tagging
	for i,line in enumerate(parts[part]):
		if len(line) == 0:
			sentence = ' '.join(words)
			res = sp(sentence)
			for i in range(len(words)):
				#word,pos,borr
				results.append((res[i],res[i].pos_,borrowed[i]))
				alltags.add(res[i].pos_)
			words = []
			borrowed = []
		else:
			word,status = line.split('\t')
			words.append(word)
			borrowed.append(status)
	allresults[part] = results

#add words from reference.csv
allresults['reference'] = rwords
#add words from input.csv
allresults['input'] = iwords

#RULES FOR GENERATING FEATURES

#look for words in all caps
def allcaps(word):
	if re.search('^[A-Z]+$',str(word[0])):
		return 1
	return 0

#exclude proper names (capitalized in Spanish)
def capitalized(word):
	if re.search('^[A-Z]',str(word[0])):
		return 1
	return 0

#exclude non-alphabetic characters
def nonalpha(word):
	if re.search('[^a-zA-Z]',str(word[0])):
		return 1
	return 0

#count letters
def countletters(word):
	return len(str(word[0]))

#common spanish endings
def spanishendings(word):
	if re.search('[^i][ao][sn]?$',str(word[0])):
		return 1
	return 0

#check english dictionaries

engwords = [w.lower() for w in ewords.words()]
engwords = set(engwords)

#engwords = set(ewords.words())
def checkEnglish(word):
	if str(word[0]).lower() in english:
		return 1
	if str(word[0]).lower() in engwords:
		return 1
	if str(word[0]).lower() in ewords2:
		return 1
	return 0

#check spanish dictionary
def checkSpanish(word):
	if str(word[0]).lower() in spanish:
		return 1
	return 0

#get bigram probability
def testwd(wd,bgs):
	wd = wd.lower()
	res = 0
	for i in range(1,len(wd)):
		bg = wd[i-1:i+1]
		if bg not in bgs:
			res -= 30
		else:
			res += bgs[bg]
	return res

#bigram score for english
def engNG(wd):
	wdstr = str(wd[0])
	res = testwd(wdstr,englishbigrams)
	return res

#bigram score for spanish
def spaNG(wd):
	wdstr = str(wd[0])
	res = testwd(wdstr,spanishbigrams)
	return res

#stem english words
port = PorterStemmer()
def engStem(wd):
	wdstr = str(wd[0]).lower()
	stem = port.stem(wdstr)
	if stem != wdstr:
		return 1
	return 0

#stem spanish words
snow = SnowballStemmer('spanish')
def spanStem(wd):
	wdstr = str(wd[0]).lower()
	stem = snow.stem(wdstr)
	if stem != wdstr:
		return 1
	return 0

#functions for individual tags
def maketagfunc(t):
	def f(word):
		if word[1] == t:
			return 1
		return 0
	return f
tagfunctions = []
for tag in alltags:
	tagfunctions.append(maketagfunc(tag))

#put all the functions together
allfuncs = tagfunctions + \
	[nonalpha,capitalized,countletters, \
	spanishendings,checkEnglish, \
	checkSpanish,engNG,spaNG, \
	spanStem,engStem,allcaps]

#get scores for all items
print('making vectors...')
means = np.zeros(len(allfuncs))
totalitems = 0
allcoded = {}
for part in allresults:
	print(f'\t{part}')
	coded = []
	for word in allresults[part]:
		totalitems += 1
		vec = []
		for func in allfuncs:
			vec.append(func(word))
		means += vec
		if part != 'input':
			if word[2] == 'O':
				code = 0
			else:
				code = 1
		if part == 'reference':
			#vec,borr,word,sent
			coded.append((np.array(vec),code,str(word[0]),word[3]))
		elif part == 'input':
			#vec,word,sent
			coded.append((np.array(vec),str(word[0]),word[2]))
		else:
			#vec,borr,word
			coded.append((np.array(vec),code,str(word[0])))
	allcoded[part] = coded
#means for all features
means /= totalitems

print('converting to z-scores')

#get standard deviations
sds = np.zeros(len(allfuncs))
for part in allcoded:
	for word in allcoded[part]:
		sds += (word[0]-means)**2
sds /= totalitems
sds = np.sqrt(sds)

#convert to z-scores
newcoded = {}
for part in allcoded:
	newcoded[part] = []
	for word in allcoded[part]:
		if part == 'reference':
			newcoded[part].append(
				#vec,borr,word,sent
				((word[0]-means)/sds,word[1],word[2],word[3])
			)
		elif part == 'input':
			newcoded[part].append(
				#vec,word,sent
				((word[0]-means)/sds,word[1],word[2])
			)
		else:
			newcoded[part].append(
				#vec,borr,word
				((word[0]-means)/sds,word[1],word[2])
			)

#train with coalas training, dev, and test data
trainingdata = newcoded['training'] + newcoded['dev'] + newcoded['test']
testdata = newcoded['reference']

if testInput:
	trainingdata += newcoded['reference']
	testdata = newcoded['input']

#logistic regression version
if logreg:

	#random initial weights for each feature
	weights = np.random.rand(len(allfuncs))
	bias = 5.0

	#train
	print('training...')
	losses = []
	for epoch in range(epochs):
		epochloss = 0
		print(f'epoch: {epoch+1}')
		#shuffle training data for each epoch
		#random.shuffle(newcoded['training'])
		random.shuffle(trainingdata)
		#for vector,y,_ in trainingdata:
		#for vector,y,_ in newcoded['training']:
		for item in trainingdata:
			vector = item[0]
			y = item[1]
			#calculate predicted output
			yhat = sigmoid(weights.dot(vector) + bias)
			#compute updates
			weightupdates = (yhat - y)*vector*lr
			biasupdate = (yhat - y)*lr
			#do updates
			weights -= weightupdates
			bias -= biasupdate
			epochloss += (yhat-y)
		epochloss /= len(allcoded['training'])
		losses.append(epochloss)

	#test
	print('testing...')
	results = 0
	precise = {}
	#for vector,y,_ in newcoded['test']:
	for item in testdata:
		vector = item[0]
		#compute output
		yhat = sigmoid(weights.dot(vector) + bias)
		#convert to categorical with threshold
		if yhat > .5:
			yhat = 1
		else:
			yhat = 0

		if not testInput:
			y = item[1]
			if yhat == y:
				results += 1
			code = str(yhat) + str(y)
			if code in precise:
				precise[code] += 1
			else:
				precise[code] = 1

	if not testInput:
		#print(results/len(allcoded['test']))
		print(results/len(testdata))

		for code in precise:
			print(f'{code}: {precise[code]}')

	plt.plot(losses)
	plt.show()

#neural version
else:

	print('neural version')

	#if torch.backends.mps.is_available():
	#	print('mps')
	#	device = 'mps'
	#else:
	#	print('cpu')
	#	device = 'cpu'

	device = 'cpu'

	#define network
	class BinaryClassifier(nn.Module):
		def __init__(self,input_size,hidden_size,output_size):
			super(BinaryClassifier,self).__init__()
			self.hidden1 = nn.Linear(input_size,hidden_size)
			self.hidden2 = nn.Linear(hidden_size,hidden_size)
			self.hidden3 = nn.Linear(hidden_size,hidden_size)
			self.output = nn.Linear(hidden_size,output_size)
			self.sigmoid = nn.Sigmoid()

		def forward(self,x):
			x = self.sigmoid(self.hidden1(x))
			x = self.sigmoid(self.hidden2(x))
			x = self.sigmoid(self.hidden3(x))
			#x = torch.relu(self.hidden(x))
			x = self.sigmoid(self.output(x))
			return x

	input_size = len(allfuncs)
	hidden_size = len(allfuncs)
	output_size = 1

	model = BinaryClassifier(input_size,hidden_size,output_size)
	model.to(device)

	#optimizer = optim.SGD(model.parameters(),lr=lr)
	optimizer = optim.Adam(model.parameters(),lr=lr)

	#loss function
	criterion = nn.BCELoss()

	if useNew:
		losses = []
		for epoch in range(epochs):
			print('epoch',epoch+1)
			random.shuffle(trainingdata)
			epochloss = 0
			i = 0
			while i < len(trainingdata):
				vectors = np.array([i[0] for i in trainingdata[i:i+batchsize]])
				ys = np.array([[i[1]] for i in trainingdata[i:i+batchsize]])
				optimizer.zero_grad()
				vectors = torch.tensor(vectors).float().to(device)
				ys = torch.tensor(ys).float().to(device)

				#forward pass
				output = model(vectors)
				loss = criterion(output,ys)
				loss.backward()
				optimizer.step()
				epochloss += loss.item()
				i += batchsize
			losses.append(epochloss/len(trainingdata))

		torch.save(model.state_dict(),'/Users/hammond/Desktop/adobo.pt')

	else:
		#load old system
		model.load_state_dict(
			torch.load(
				savedstate,
				weights_only=True
			)
		)

	#do testing
	with torch.no_grad():
		model.eval()
		results = 0
		precise = {}
		f = open('/Users/hammond/Desktop/res.txt','w')
		#for vector,y,w in newcoded['test']:
		#for vector,y,w,i in newcoded['reference']:
		#for vector,y,w,i in testdata:
		for item in testdata:
			vector = item[0]
			w = item[-2]
			i = item[-1]
			#compute output
			vector = torch.tensor(vector).float().unsqueeze(0).to(device)
			output = model(vector)
			yhat = output.item()
			#convert to categorical with threshold
			if yhat > .5:
				yhat = 1
			else:
				yhat = 0
			if not testInput:
				if yhat == y:
					results += 1
				code = str(yhat) + str(y)
				if code in precise:
					precise[code] += 1
				else:
					precise[code] = 1
				#f.write(f'{code}: {w}\n')
				f.write(f'{code}: {w}\t{i}\n') 
			else:
				f.write(f'{yhat}: {w}\t{i}\n') 
		f.close()

	if not testInput:

		#print(results/len(allcoded['test']))
		print(results/len(allcoded['reference']))

		print(f'total test items: {len(allcoded['reference'])}')

		for code in precise:
			print(f'{code}: {precise[code]}')

	if useNew:
		torch.save(model.state_dict(),'/Users/hammond/Desktop/adobo.pt')

		plt.plot(losses)
		plt.show()

