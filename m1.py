import re
from nltk.corpus import words as ewords
from nltk.corpus import cess_esp

englishwordsfile = '/usr/share/dict/words'
#more english words from here: https://github.com/dwyl/english-words
ewords2file = '/Users/hammond/Desktop/english-words-master/words.txt'

#read in three english wordlists and merge
f = open(englishwordsfile,'r')
e1 = f.read()
f.close()
e1 = e1.split('\n')[:-1]
e1 = [e for e in e1 if re.search('^[a-z]',e)]

f = open(ewords2file,'r')
e2 = f.read()
f.close()
e2 = e2.split('\n')[:-1]
e2 = [e for e in e2 if re.search('^[a-z]',e)]

e3 = [w for w in ewords.words() if re.search('^[a-z]',w)]

english = set(e1 + e2 + e3)

#read in spanish wordlist

spanish = [w for w in cess_esp.words() if re.search('^[a-z]',w)]

spanish = set(cess_esp.words())

#if something is english and not spanish make it a borrowing

f = open('/Users/hammond/Desktop/res.txt','r')
res = f.read()
f.close()

res = res.split('\n')[:-1]

f = open('/Users/hammond/Desktop/res2.txt','w')
for i,line in enumerate(res):
	prefix = line[:3]
	suffix = line[3:]
	word,sentence = suffix.split('\t')
	word = word.lower()
	if prefix[0] == '0' and word in english and word not in spanish:
		f.write(f'1{line[1:]}\n')
	elif prefix[0] == '0' and word in english and \
			word not in ['un','de','las','el','la','en'] and \
			res[i+1][0] == '1':
		f.write(f'1{line[1:]}\n')
	else:
		f.write(f'{line}\n')
f.close()

