import csv,re

#assemble results into sentences

#read in reference sentences
rlines = []
#with open('reference.csv', 'r') as f:
with open('input.csv', 'r') as f:
   csvreader = csv.reader(f,quotechar='"',delimiter=';')
   for row in csvreader:
      rlines.append(row[0])

#read in results massaged by m1.py
f = open('/Users/hammond/Desktop/res2.txt','r')
res = f.read()
f.close()

res = res.split('\n')[:-1]

#make fields
triples = []
for line in res:
	code = re.sub(': .*$','',line)
	line = re.sub('^(..|.): ','',line)
	word,sentence = line.split('\t')
	sentence = int(sentence)
	triples.append((code,word,sentence))

#merge adjacent borrowings
merged = [triples[0]]
for code,word,sentence in triples:
	lastcode,lastword,lastsentence = merged[-1]
	if lastcode[0] == code[0] == '1' and sentence == lastsentence:
		merged[-1] = (
			code,
			lastword + ' ' + word,
			sentence
		)
	else:
		merged.append((code,word,sentence))

borrowings = {}
for code,word,sentence in merged:
	if code[0] == '1':
		if sentence in borrowings:
			borrowings[sentence].add(word)
		else:
			borrowings[sentence] = set([word])

data = []
for i,sentence in enumerate(rlines):
	if i in borrowings:
		sfx = list(borrowings[i])
	else:
		sfx = ['','','','']
	item = [sentence] + sfx
	data.append(item)

with open('mh.csv','w',newline='\n') as f:
    writer = csv.writer(f,quotechar='"',delimiter=';')
    writer.writerows(data)

