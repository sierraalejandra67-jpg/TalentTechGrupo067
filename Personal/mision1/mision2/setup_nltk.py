import nltk
try
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading 'punkt': {e}")
try:    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading 'stopwords': {e}")
    
    