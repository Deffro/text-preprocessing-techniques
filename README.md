# text-preprocessing-techniques
## 16 Text Preprocessing Techniques in Python for Twitter Sentiment Analysis.

These techniques were used in comparison in our paper **"A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis"**. If you use this material please [cite](https://link.springer.com/chapter/10.1007/978-3-319-67008-9_31) the paper. An extended paper for this work can be found [here](https://www.sciencedirect.com/science/article/pii/S0957417418303683), with the title **"A comparative evaluation of pre-processing techniques and their interactions for twitter sentiment analysis"**. Please cite.
 
Most of these techniques are generic and can be used in various applications except Sentiment Analysis. 
They are the following:

#### 0. Remove Unicode Strings and Noise
#### 1. Replace URLs, User Mentions and Hashtags
#### 2. Replcae Slang and Abbreviations
#### 3. Replace Contractions
#### 4. Remove Numbers
#### 5. Replace Repetitions of Punctuation
#### 6. Replace Negations with Antonyms
#### 7. Remove Punctuation
#### 8. Handling Capitalized Words
#### 9. Lowercase
#### 10. Remove Stopwords
#### 11. Replace Elongated Words
#### 12. Spelling Correction
#### 13. Part of Speech Tagging
#### 14. Lemmatizing
#### 15. Stemming

This scripts also prints some statistics for the text file like: 

- Total Sentences
- Total Words before and after preprocess
- Total Unique words before and after preprocess
- Average Words per Sentence before and after preprocess
- Total Run time
- Total Emoticons found
- Total Slangs and Abbreviations found
- 20 Most Commong Sland and Abbreviations and plots them
- Total Elongated words
- Total multi Exclamation
- question and stop marks
- Total All Capitalized words
- 100 Most Common words and plots them and most common bigram and trigram collocations

The text file that we included here is a sample (2000 tweets) of the SS-Twitter dataset.

The file "preprocess.py" includes many comments and in order to use a technique you have to uncomment the appropriate line/lines. The initial script uses all techniques. So if you want to use only specific techniques, comment out the others.
