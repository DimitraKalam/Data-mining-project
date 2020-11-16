#Καλαματιανού Δήμητρα
#up1054406@upnet.gr

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import nltk
# nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the dataset
all_onion_data = pd.read_csv(r'C:\Users\Δημητρα\Documents\ΣΧΟΛΗ\8o ΕΞΑΜΗΝΟ\εξορυξη\ΥΛΟΠΟΙΗΤΙΚΟ_PROJECT_2020\onion-or-not.csv')
all_onion_data["text"]=all_onion_data["text"].str.lower() #se lower case

# 1 make dianisma lekseon
all_onion_data['dianisma_lekseon']= all_onion_data.apply(lambda row: nltk.word_tokenize(row['text']), axis = 1)

# 2 stemming
stemmer = PorterStemmer()
all_onion_data['stemming']= all_onion_data['dianisma_lekseon'].apply(lambda x: [stemmer.stem(y) for y in x])

# 3 stopwords removal
stops = set(stopwords.words("english"))
all_onion_data['stopwords']= all_onion_data['stemming'].apply((lambda x: [item for item in x if item not in stops]))

# 4 and 5 tf-idf and vector
all_onion_data['tdf']=[" ".join(z) for z in all_onion_data['stopwords'].values] #prepei na ginei join gia na doulepsei to tf-idf
v = TfidfVectorizer()
x = v.fit_transform(all_onion_data['tdf']) # sparse matrix


#train and test
# Convert the dataframe to a numpy array and split the data
npArray = np.array(all_onion_data["label"])
X = x
y = npArray
# Split into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split( X, y, test_size=0.25)
#neural network
mlp = MLPClassifier()
mlp.fit(XTrain,yTrain)
predictions = mlp.predict(XTest)

print(classification_report(yTest,predictions))
print("Overall Accuracy:", round(metrics.accuracy_score(yTest, predictions), 4))
