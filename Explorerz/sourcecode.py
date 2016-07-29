
import os, sys
import nltk
import random
file_paths1=[]
file_paths=[]
count = {}

DIR = r"C:\Users\Admin\AppData\Local\Programs\Python\Python35\pos" #location of pos folder
for root,directories,files in os.walk(DIR):  #root diectory and files in the given dir
   for filename in files:
     filepath=os.path.join(root,filename)  #Joins one or more path components
     file_paths.append(filepath)  #file_paths contains the whole pathname

all_words=[]
lnames=[]
lpos=[[[],'pos']] #list of list of feature words from pos
for p in file_paths:
   lnames=open(p,'r').read().split() #reads the file and splits it into smaller parts
   lpos.append([lnames,'pos']) #we append the small words with pos tag to lpos
   for w in lnames:
      all_words.append(w) #all_words contains all split words from pos files
#print(lpos[1])
      



DIR1 = r"C:\Users\Admin\AppData\Local\Programs\Python\Python35\neg" #location of neg folder
for root,directories,files in os.walk(DIR1):
   for filename in files:
     filepath1=os.path.join(root,filename)
     file_paths1.append(filepath1)

for q in file_paths1:
   lnames=open(q,'r').read().split()
   lpos.append([lnames,'neg'])
   for w in lnames:
      all_words.append(w)
#print(lpos[1])
random.shuffle(lpos)
#print(len(all_words))
#print(lpos)

word_features=list(all_words)[:2000] #take first 2000 most frequent words

#print(all_words)
def document_features(document): #checks whether each of these words is present in a given document
    document_words = set(document)  
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets=[(document_features(d),c) for (d,c) in lpos]
train_set,test_set=featuresets[100:],featuresets[:100] #splitting train set and test set

classifier = nltk.NaiveBayesClassifier.train(train_set) #train a classifier to label new movie reviews
print (nltk.classify.accuracy(classifier, test_set)) #printing accuracy
classifier.show_most_informative_features(5)       # features the classifier found to be most informative         
