import spacy,csv,re

#check words in caps or parens

#write to results.csv

#python -m spacy download es_core_news_md
sp = spacy.load('es_core_news_md',disable=['parser','ner'])

rlines = []
with open('mh.csv', 'r') as f:
	csvreader = csv.reader(f,quotechar='"',delimiter=';')
	for row in csvreader:
		rlines.append(row)

#fix caps
newlines = []
for line in rlines:
	sentence = sp(line[0])
	words = []
	for word in sentence:
		if re.search('^[A-Z]{2,}$',str(word)):
			words.append(str(word))
	if len(words) < 5:
		newwords = []
		for word in line[1:]:
			if len(word) > 0:
				newwords.append(word)
		newwords = words + newwords
		newlines.append([line[0]] + newwords)
	else:
		newlines.append(line)

#fix quotes
verynewlines = []
for line in newlines:
	matches = re.findall('"[^"]*"',line[0])
	borrowings = line[1:]
	for match in matches:
		match = re.sub('"','',match)
		words = match.split(' ')
		if len(words) < 4:
			borrowings = [e for e in borrowings if e not in words]
			#don't include short words (len(e) > 1 in results13)
			borrowings = [e for e in borrowings if len(e) > 0]
			borrowings = borrowings + [match]
	borrowings = list(set(borrowings))
	newline = [line[0]] + borrowings
	verynewlines.append(newline)

with open('results.csv','w',newline='\n') as f:
	writer = csv.writer(f,quotechar='"',delimiter=';')
	writer.writerows(verynewlines)

