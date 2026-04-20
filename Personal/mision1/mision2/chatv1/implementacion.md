 guia de implementacion del chatbot(supervisado)
 import nltk 
 nltk.download('punkt')
 
 python -m pip instal 

 from sklearn.feature_extraction.text import CountVectorizer
 # CountVectorizer convertir texto en un vector
from sklearn.naive_bayes import MultinomialNB
# MultinomialNB modelo de inteligencia artificial que aprende relaciones entre texto y respuesta 

#funcion build_and_train_model
def build_and_train_model(training_pairs):
