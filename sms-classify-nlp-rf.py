import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from nltk.stem import PorterStemmer

# Other algorithms can be used for the same functionality: 
# MultinomialNB, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier


file_path = r"TRAIN_info_SMS.csv"
train_df= pd.read_csv(file_path,encoding='ISO-8859-1')
train_df.head()

train_df.shape

train_df.head()

def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
 
    classifier.fit(X_train, y_train)
    print("Accuracy: %s",classifier.score(X_test, y_test))
    return classifier
 
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]
 
nltkPipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('classifier', RandomForestClassifier(n_estimators=1000,criterion='gini',min_samples_split=2)),
	# MultinomialNB(alpha=0.05),
	# AdaBoostClassifier( base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.03)
	# GradientBoostingClassifier(learning_rate=0.1,min_samples_split=500,
	#							min_samples_leaf=50,max_depth=8,max_features='sqrt',
	#							subsample=0.8,random_state=10)
	# BaggingClassifier(base_estimator=cart, n_estimators=num_trees)
])


 
model=train(nltkPipeline, train_df['Message'], train_df['Label'])

file_path = r"DEV_info_SMS.csv"
test_df= pd.read_csv(file_path,encoding='ISO-8859-1')

predictions = model.predict(test_df["Message"])

submission = pd.DataFrame()

submission["RecordNo"] = test_df["RecordNo"]
submission["Label"] = predictions

#We save the submission as a '.csv' file
submission.to_csv("output-result-rf.csv", index=False)
