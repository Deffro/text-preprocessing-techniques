""" Copyright 2017, Dimitrios Effrosynidis, All rights reserved. """

from time import time
import numpy as np
import string

from techniques import *

print("Starting preprocess..\n")

""" Tokenizes a text to its words, removes and replaces some of them """    
finalTokens = [] # all tokens
stoplist = stopwords.words('english')
my_stopwords = "multiexclamation multiquestion multistop url atuser st rd nd th am pm" # my extra stopwords
stoplist = stoplist + my_stopwords.split()
allowedWordTypes = ["J","R","V","N"] #  J is Adject, R is Adverb, V is Verb, N is Noun. These are used for POS Tagging
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer

def tokenize(text, wordCountBefore, textID, y):
    totalAdjectives = 0
    totalAdverbs = 0
    totalVerbs = 0
    onlyOneSentenceTokens = [] # tokens of one sentence each time

    tokens = nltk.word_tokenize(text)
    
    tokens = replaceNegations(tokens) # Technique 6: finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym
    
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) # Technique 7: remove punctuation

    tokens = nltk.word_tokenize(text) # it takes a text as an input and provides a list of every token in it
    
### NO POS TAGGING BEGIN (If you don't want to use POS Tagging keep this section uncommented) ###
    
##    for w in tokens:
##
##        if (w not in stoplist): # Technique 10: remove stopwords
##            final_word = addCapTag(w) # Technique 8: Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_
##            final_word = final_word.lower() # Technique 9: lowercases all characters
##            final_word = replaceElongated(final_word) # Technique 11: replaces an elongated word with its basic form, unless the word exists in the lexicon
##            if len(final_word)>1:
##                final_word = spellCorrection(final_word) # Technique 12: correction of spelling errors
##            final_word = lemmatizer.lemmatize(final_word) # Technique 14: lemmatizes words
##            final_word = stemmer.stem(final_word) # Technique 15: apply stemming to words

### NO POS TAGGING END ###


### POS TAGGING BEGIN (If you want to exclude words using POS Tagging, keep this section uncommented and comment the above) ###          
            
    tagged = nltk.pos_tag(tokens) # Technique 13: part of speech tagging  
    for w in tagged:

        if (w[1][0] in allowedWordTypes and w[0] not in stoplist):
            final_word = addCapTag(w[0])
            #final_word = final_word.lower()
            final_word = replaceElongated(final_word)
            if len(final_word)>1:
                final_word = spellCorrection(final_word)
            final_word = lemmatizer.lemmatize(final_word)
            final_word = stemmer.stem(final_word)

### POS TAGGING END ###
                
            onlyOneSentenceTokens.append(final_word)           
            finalTokens.append(final_word)

         
    onlyOneSentence = " ".join(onlyOneSentenceTokens) # form again the sentence from the list of tokens
    #print(onlyOneSentence) # print final sentence

    
    """ Write the preprocessed text to file """
    with open("result.txt", "a") as result:
        result.write(textID+"\t"+y+"\t"+onlyOneSentence+"\n")
        
    return finalTokens


f = open("ss-twitterfinal.txt","r", encoding="utf8", errors='replace').read()

t0 = time()
totalSentences = 0
totalEmoticons = 0
totalSlangs = 0
totalSlangsFound = []
totalElongated = 0
totalMultiExclamationMarks = 0
totalMultiQuestionMarks = 0
totalMultiStopMarks = 0
totalAllCaps = 0

for line in f.split('\n'):
    totalSentences += 1
    feat = []
    columns = line.split('\t')
    columns = [col.strip() for col in columns]

    textID = (columns[0])
    y = (columns[2])

    text = removeUnicode(columns[1]) # Technique 0
    #print(text) # print initial text
    wordCountBefore = len(re.findall(r'\w+', text)) # word count of one sentence before preprocess    
    #print("Words before preprocess: ",wordCountBefore,"\n")
    
    text = replaceURL(text) # Technique 1
    text = replaceAtUser(text) # Technique 1
    text = removeHashtagInFrontOfWord(text) # Technique 1

    temp_slangs, temp_slangsFound = countSlang(text)
    totalSlangs += temp_slangs # total slangs for all sentences
    for word in temp_slangsFound:
        totalSlangsFound.append(word) # all the slangs found in all sentences
    
    text = replaceSlang(text) # Technique 2: replaces slang words and abbreviations with their equivalents
    text = replaceContraction(text) # Technique 3: replaces contractions to their equivalents
    text = removeNumbers(text) # Technique 4: remove integers from text

    emoticons = countEmoticons(text) # how many emoticons in this sentence
    totalEmoticons += emoticons
    
    text = removeEmoticons(text) # removes emoticons from text

    
    totalAllCaps += countAllCaps(text)

    totalMultiExclamationMarks += countMultiExclamationMarks(text) # how many repetitions of exlamation marks in this sentence
    totalMultiQuestionMarks += countMultiQuestionMarks(text) # how many repetitions of question marks in this sentence
    totalMultiStopMarks += countMultiStopMarks(text) # how many repetitions of stop marks in this sentence
    
    text = replaceMultiExclamationMark(text) # Technique 5: replaces repetitions of exlamation marks with the tag "multiExclamation"
    text = replaceMultiQuestionMark(text) # Technique 5: replaces repetitions of question marks with the tag "multiQuestion"
    text = replaceMultiStopMark(text) # Technique 5: replaces repetitions of stop marks with the tag "multiStop"

    totalElongated += countElongated(text) # how many elongated words emoticons in this sentence
    
    tokens = tokenize(text, wordCountBefore, textID, y)  
    
    
print("Total sentences: ",totalSentences,"\n")
print("Total Words before preprocess: ",len(re.findall(r'\w+', f)))
print("Total Distinct Tokens before preprocess: ",len(set(re.findall(r'\w+', f))))
print("Average word/sentence before preprocess: ",len(re.findall(r'\w+', f))/totalSentences,"\n")
print("Total Words after preprocess: ",len(tokens))
print("Total Distinct Tokens after preprocess: ",len(set(tokens)))
print("Average word/sentence after preprocess: ",len(tokens)/totalSentences,"\n")


print("Total run time: ",time() - t0," seconds\n")

print("Total emoticons: ",totalEmoticons,"\n")
print("Total slangs: ",totalSlangs,"\n")
commonSlangs = nltk.FreqDist(totalSlangsFound)
for (word, count) in commonSlangs.most_common(20): # most common slangs across all texts
    print(word,"\t",count)

commonSlangs.plot(20, cumulative=False) # plot most common slangs

print("Total elongated words: ",totalElongated,"\n")
print("Total multi exclamation marks: ",totalMultiExclamationMarks)
print("Total multi question marks: ",totalMultiQuestionMarks)
print("Total multi stop marks: ",totalMultiStopMarks,"\n")
print("Total all capitalized words: ",totalAllCaps,"\n")

#print(tokens)
commonWords = nltk.FreqDist(tokens)
print("Most common words ")
print("Word\tCount")
for (word, count) in commonWords.most_common(100): # most common words across all texts
    print(word,"\t",count)

commonWords.plot(100, cumulative=False) # plot most common words


bgm = nltk.collocations.BigramAssocMeasures()
tgm = nltk.collocations.TrigramAssocMeasures()
bgm_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
tgm_finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
bgm_finder.apply_freq_filter(5) # bigrams that occur at least 5 times
print("Most common collocations (bigrams)")
print(bgm_finder.nbest(bgm.pmi, 50)) # top 50 bigram collocations
tgm_finder.apply_freq_filter(5) # trigrams that occur at least 5 times
print("Most common collocations (trigrams)")
print(tgm_finder.nbest(tgm.pmi, 20)) # top 20 trigrams collocations
