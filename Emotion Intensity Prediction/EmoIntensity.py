# read train files
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk.sentiment.vader as vd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


filePath = 'anger_train.csv' 
angerf = open(filePath, "r")
anger_train = angerf.read()
anger_train = anger_train.split("\n")
anger_train = [i.split("\t") for i in anger_train]
train_anger = pd.DataFrame(anger_train, columns = ['id', 'tweet', 'emotion', 'score'])

filePath2 = 'joy_train.csv' 
joyf = open(filePath2, "r")
joy_train = joyf.read()
joy_train = joy_train.split("\n")
joy_train = [i.split("\t") for i in joy_train]
train_joy = pd.DataFrame(joy_train, columns = ['id', 'tweet', 'emotion', 'score'])

# read test files
filePath = 'anger_test.csv' 
angerf = open(filePath, "r")
anger_test = angerf.read()
anger_test = anger_test.split("\n")
anger_test = [i.split("\t") for i in anger_test]
test_anger = pd.DataFrame(anger_test, columns = ['id', 'tweet', 'emotion', 'score'])

filePath2 = 'joy_test.csv' 
joyf = open(filePath2, "r")
joy_test = joyf.read()
joy_test = joy_test.split("\n")
joy_test = [i.split("\t") for i in joy_test]
test_joy = pd.DataFrame(joy_test, columns = ['id', 'tweet', 'emotion', 'score'])

# pick relevant columns
pick = train_anger[['tweet','score']]
pick_joy = train_joy[['tweet','score']]

test_pick = test_anger[['tweet','score']]
test_pick_joy = test_joy[['tweet','score']]

# preprocessing

nltk.download('punkt')


def preprocess(pick):
  tweets = []
  sents=[]
  tokens=[]
  for index, row in pick.iterrows():
      r = re.sub(r"http\S+|@\S+", "", row['tweet'])
      r.strip()
      text = TweetTokenizer().tokenize(r)
      if len(text)!=0:
        tokens.append(text)
        sents.append(r)
        tweets.append((r, float(row['score'])))
  return tweets, sents, tokens

tweets, sents, tokens = preprocess(pick)
j_tweets, j_sents, j_tokens = preprocess(pick_joy)
t_tweets, t_sents, t_tokens = preprocess(test_pick)
t_j_tweets, t_j_sents, t_j_tokens = preprocess(test_pick_joy)

!pip install vaderSentiment

"""VADER sentiment"""

def vader_sentiment_scores(sentence):
  sid_obj = SentimentIntensityAnalyzer()    
  sentiment_dict = sid_obj.polarity_scores(sentence) 
  return sentiment_dict

#LEXICONS

"""MPQA"""

def loadMPQALexicon():
    negativeWords = {}
    positiveWords = {}
    with open('/lexicons/2. mpqa.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            sentiment=row[1]
            if sentiment == 'positive':
                positiveWords[word] = 1
            elif sentiment == 'negative':
                negativeWords[word] = 1
    return {"pos":positiveWords, "neg":negativeWords}

mpqa = loadMPQALexicon()

"""BING LIU"""

def loadBingLiuLexicon():
    negativeWords = {}
    positiveWords = {}
    with open('/lexicons/1. BingLiu.csv','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            sentiment=row[1]
            if sentiment == 'positive':
                positiveWords[word] = 1
            elif sentiment == 'negative':
                negativeWords[word] = 1
    return {"pos":positiveWords, "neg":negativeWords}

bingLiu = loadBingLiuLexicon()

"""Sentiment140"""

def loadSentiment140Lexicon():
    scores = {}
    with open('/lexicons/3. Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            score=float(row[1])
            scores[word] = score
    return scores

sent140 = loadSentiment140Lexicon()
#print(sent140)

"""AFINN"""

def loadAFINNLexicon():
    scores = {}
    with open('/lexicons/AFINN-en-165.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            score=int(row[1])
            scores[word] = score
    return scores

afinn = loadAFINNLexicon()

"""Sentiwordnet"""

def loadSentiWordNetLexicon():
    scores = {}
    with open('/lexicons/4. SentiWordNet_3.0.0_20130122.txt','r') as iFile:
        for i in iFile:
          i1 = i.strip()
          if i1[0]!='#':
            row = i1.strip().split('\t')
            words=row[4]
            words = words.split()
            words = [w[:w.find('#')] for w in words]
            score = 1-(float(row[2])+float(row[3]))
            for w in words:
              scores[w] = score
    return scores

sentiwordnet = loadSentiWordNetLexicon()

"""NRC HASHTAG SENTIMENT"""

def loadHashtagLexicon():
    scores = {}
    with open('/lexicons/7. NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            score=float(row[1])
            scores[word] = score
    return scores

hashsent = loadHashtagLexicon()

"""NRC word-emotion"""

def loadNRCLexicon():
    angerWords = {}
    joyWords = {}
    with open('/lexicons/8. NRC-word-emotion-lexicon.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            word=row[0]
            sentiment=row[1]
            value=row[2]
            if sentiment == 'anger' and value == '1':
                angerWords[word] = 1
            elif sentiment == 'joy' and value == '1':
                joyWords[word] = 1
    return {"anger":angerWords, "joy":joyWords}

nrc = loadNRCLexicon()

"""NRC-10 Expanded"""

def loadnrc10Lexicon():
    angerScores = {}
    joyScores = {}
    with open('/lexicons/6. NRC-10-expanded.csv','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            if row[1]=="anger":
              continue
            word=row[0]
            angerscore=float(row[1])
            joyscore=float(row[5])
            angerScores[word] = angerscore
            joyScores[word] = joyscore
    return {"angerScores":angerScores, "joyScores":joyScores}
    
nrc10 = loadnrc10Lexicon()

"""NRC Hashtag Emotion"""

def loadHashEmoLexicon():
    angerScores = {}
    joyScores = {}
    with open('/lexicons/5. NRC-Hashtag-Emotion-Lexicon-v0.2.txt','r') as iFile:
        for i in iFile:
          if i.find('\t')!=-1:
            row = i.strip().split('\t')
            word = row[1]
            emotion = row[0]
            score = float(row[2])
            if emotion == "anger":
              angerScores[word] = score
            elif emotion == "joy":
              joyScores[word] = score
    return {"angerScores":angerScores, "joyScores":joyScores}
    
hashemo = loadHashEmoLexicon()

"""AFINN Emoticons"""

def loadEmoticonLexicon():
    scores = {}
    with open('/lexicons/9. AFINN-emoticon-8.txt','r') as iFile:
        for i in iFile:
            row = i.strip().split('\t')
            emoji=row[0]
            score=int(row[1])
            scores[emoji] = score
    return scores

emoticon = loadEmoticonLexicon()

"""Negating word Count"""

neg_words = vd.NEGATE

"""FEATURE VECTORS"""

# UNIGRAM

count_vect = CountVectorizer(max_features=None)
def extract_unigram_train(count_vect, sents):
  count_vect.fit(sents)
  final_counts = count_vect.transform(sents)
  unigram = final_counts.toarray()
  return unigram

def extract_unigram_test(count_vect, sents):
  final_counts = count_vect.transform(sents)
  unigram = final_counts.toarray()
  return unigram

# BIGRAM
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
def extract_bigram_train(vectorizer2, sents):
  vectorizer2.fit(sents)
  X2 = vectorizer2.transform(sents)
  bigram = X2.toarray()
  return bigram

def extract_bigram_test(vectorizer2, sents):
  X2 = vectorizer2.transform(sents)
  bigram = X2.toarray()
  return bigram

# VADER [pos score, neg score, neut score, comp score]
def extract_vader(sents):
  vader = []
  for s in sents:
    v = vader_sentiment_scores(s)
    li = [v['pos'], v['neg'], v['neu'], v['compound']]
    vader.append(li)
  return vader

# MPQA [count positive, count negative] 
def extract_mpqa(tokens):
  mpqa_f = []
  for s in tokens:
    cp = 0
    cn = 0
    for w in s:
      cp += mpqa['pos'].get(w,0)
      cn += mpqa['neg'].get(w,0)
    li = [cp, cn]
    mpqa_f.append(li)
  return mpqa_f

# Bing Liu [count positive, count negative] 
def extract_bingliu(tokens):
  bl_f = []
  for s in tokens:
    cp = 0
    cn = 0
    for w in s:
      cp += bingLiu['pos'].get(w,0)
      cn += bingLiu['neg'].get(w,0)
    li = [cp, cn]
    bl_f.append(li)
  return bl_f

# Sentiment140 [aggregate word score]
def extract_sent140(tokens):
  sent140_f = []
  for s in tokens:
    n = len(s)
    score = 0
    for t in s:
      score += sent140.get(t,0)
    avg = score/n
    sent140_f.append([avg])
  return sent140_f

# AFINN [aggregate word score]
def extract_AFINN(tokens):
  AFINN_f = []
  for s in tokens:
    n = len(s)
    score = 0
    for t in s:
      score += afinn.get(t,0)
    avg = score/n
    AFINN_f.append([avg])
  return AFINN_f

# Sentiwordnet [aggregate word score]
def extract_wordnet(tokens):
  sentiwordnet_f = []
  for s in tokens:
    n = len(s)
    score = 0
    for t in s:
      score += sentiwordnet.get(t,0)
    avg = score/n
    sentiwordnet_f.append([avg])
  return sentiwordnet_f

# Hashtag Sentiment [aggregate score]
def extract_hashtag_sent(tokens):
  hashsent_f = []
  for s in tokens:
    n = 0
    score = 0
    for t in s:
      if(t[0]=='#'):
        n+=1
        tmp = max(hashsent.get(t,0),hashsent.get(t[1:],0))
        score+=tmp
    if n==0:
      avg=0
    else:
      avg = score/n
    hashsent_f.append([avg])
  return hashsent_f

# NRC Word Emotion [count of words with emotion e]
def extract_nrcemo(tokens, e):
  nrc_f = []
  for s in tokens:
    count = 0
    for t in s:
      count += nrc[e].get(t,0)
    nrc_f.append([count])
  return nrc_f

# NRC10 Expanded [sum of emotion scores]
def extract_nrc10(tokens, e):
  nrc10_f = []
  for s in tokens:
    score = 0
    for t in s:
      score+=nrc10[e+'Scores'].get(t,0)
    nrc10_f.append([score])
  return nrc10_f

# NRC Hashtag Emotion [sum of hashtag emotion scores]
def extract_hashemo(tokens, e):
  hashemo_f = []
  for s in tokens:
    score = 0
    for t in s:
      if(t[0]=='#'):
        tmp = max(hashemo[e+'Scores'].get(t,0),hashemo[e+'Scores'].get(t[1:],0))
        score+=tmp
    hashemo_f.append([score])
  return hashemo_f

# AFINN emoticons [emoticon score]
def extract_emoticons(tokens):
  emoticon_f = []
  for s in tokens:
    score = 0
    for t in s:
      score+=emoticon.get(t,0)
    emoticon_f.append([score])
  return emoticon_f

# negating words [count of negating words]

def extract_negation(tokens):
  neg_f = []
  for s in tokens:
    count=0
    for t in s:
      if (t in neg_words):
        count+=1
    neg_f.append([count])
  return neg_f

# get feature vector of dataset sents, settype is train or test, e is emotion (anger or joy)

def get_features(settype, sents, tokens, e):
  if (settype =='train'):
    unigram = extract_unigram_train(count_vect, sents)
    bigram = extract_bigram_train(vectorizer2, sents)
  else:
    unigram = extract_unigram_test(count_vect, sents)
    bigram = extract_bigram_test(vectorizer2, sents)
  vader = extract_vader(sents)
  mpqa_f = extract_mpqa(tokens)
  bl_f = extract_bingliu(tokens)
  sent140_f = extract_sent140(tokens)
  AFINN_f = extract_AFINN(tokens)
  sentiwordnet_f = extract_wordnet(tokens)
  hashsent_f = extract_hashtag_sent(tokens)
  nrc_f = extract_nrcemo(tokens, e)
  nrc10_f = extract_nrc10(tokens, e)
  hashemo_f = extract_hashemo(tokens, e)
  emoticon_f = extract_emoticons(tokens)
  neg_f = extract_negation(tokens)
  feat= np.concatenate((unigram, bigram, vader, mpqa_f, bl_f, sent140_f, AFINN_f, sentiwordnet_f, hashsent_f, nrc_f, nrc10_f, hashemo_f, emoticon_f, neg_f), axis=1).tolist()
  return feat

# get target values
def get_y(tweets):
  y = [i[1] for i in tweets]
  return y

"""Anger"""

# training features
feat_train1 = get_features("train", sents, tokens, "anger")
y_train1 = get_y(tweets)

featfile1 = open('anger_train_features', 'wb')
pickle.dump(feat_train1, featfile1)                      
featfile1.close()

# testing features
feat_test1 = get_features("test", t_sents, t_tokens, "anger")
y_test1 = get_y(t_tweets)

featfile2 = open('anger_test_features', 'wb')
pickle.dump(feat_test1, featfile2)            
featfile2.close()

anger_svr = SVR()
anger_svr.fit(feat_train1, y_train1)
filename = 'svm_anger_model.sav'
pickle.dump(anger_svr, open(filename, 'wb'))

y_pred1 = anger_svr.predict(feat_test1)
dta = test_anger.drop('score', axis=1)
dta['score'] = y_pred1
dta.to_csv("svm_anger_pred.csv", sep='\t', header=False, index=False)

anger_mlp = MLPRegressor()
anger_mlp.fit(feat_train1, y_train1)
filename = 'mlp_anger_model.sav'
pickle.dump(anger_mlp, open(filename, 'wb'))

y_pred2 = anger_mlp.predict(feat_test1)
dta = test_anger.drop('score', axis=1)
dta['score'] = y_pred2
dta.to_csv("mlp_anger_pred.csv", sep='\t', header=False, index=False)

anger_dt = DecisionTreeRegressor()
anger_dt.fit(feat_train1, y_train1)
filename = 'dt_anger_model.sav'
pickle.dump(anger_dt, open(filename, 'wb'))

y_pred3 = anger_dt.predict(feat_test1)
dta = test_anger.drop('score', axis=1)
dta['score'] = y_pred3
dta.to_csv("dt_anger_pred.csv", sep='\t', header=False, index=False)

"""Joy"""

# training features
feat_train2 = get_features("train", j_sents, j_tokens, "joy")
y_train2 = get_y(j_tweets)

featfile3 = open('joy_train_features', 'wb')
pickle.dump(feat_train2, featfile3)                      
featfile3.close()

# testing features
feat_test2 = get_features("test", t_j_sents, t_j_tokens, "joy")
y_test2 = get_y(t_j_tweets)
featfile4 = open('joy_test_features', 'wb')
pickle.dump(feat_test2, featfile4)                      
featfile4.close()

joy_svr = SVR()
joy_svr.fit(feat_train2, y_train2)
filename = 'svm_joy_model.sav'
pickle.dump(joy_svr, open(filename, 'wb'))

y_pred4 = joy_svr.predict(feat_test2)
dta = test_joy.drop('score', axis=1)
dta['score'] = y_pred4
dta.to_csv("svm_joy_pred.csv", sep='\t', header=False, index=False)


joy_mlp = MLPRegressor()
joy_mlp.fit(feat_train2, y_train2)
filename = 'mlp_joy_model.sav'
pickle.dump(joy_mlp, open(filename, 'wb'))

y_pred5 = joy_mlp.predict(feat_test2)
dta = test_joy.drop('score', axis=1)
dta['score'] = y_pred5
dta.to_csv("mlp_joy_pred.csv", sep='\t', header=False, index=False)

regressor = DecisionTreeRegressor()
regressor.fit(feat_train2, y_train2)
filename = 'dt_joy_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

y_pred6 = regressor.predict(feat_test2)
dta = test_joy.drop('score', axis=1)
dta['score'] = y_pred6
dta.to_csv("dt_joy_pred.csv", sep='\t', header=False, index=False)
