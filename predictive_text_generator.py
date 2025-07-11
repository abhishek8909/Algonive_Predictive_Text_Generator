
# Predictive Text Generator using N-gram model
import random
import nltk
from nltk.corpus import reuters
from collections import defaultdict

# Download dataset if not already downloaded
nltk.download('reuters')
nltk.download('punkt')

class PredictiveTextGenerator:
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(list)
        self.custom_words = set()
        self._train_model()

    def _train_model(self):
        """
        Train the model using the Reuters corpus
        """
        for fileid in reuters.fileids():
            words = list(reuters.words(fileid))
            for i in range(len(words) - self.n):
                key = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[key].append(next_word)

    def predict(self, input_text):
        """
        Predict the next word based on input_text
        """
        tokens = nltk.word_tokenize(input_text)
        if len(tokens) < self.n - 1:
            print("Please enter more words for context.")
            return None
        key = tuple(tokens[-(self.n-1):])
        possible_words = self.ngrams.get(key, [])
        if not possible_words:
            print("No prediction found. Try adding custom words.")
            return None
        return random.choice(possible_words)

    def add_custom_word(self, word):
        """
        Add a custom word to the user's dictionary
        """
        self.custom_words.add(word)
        print(f"Added '{word}' to custom dictionary.")

# ---------------------------
# Main demonstration
# ---------------------------
if __name__ == "__main__":
    ptg = PredictiveTextGenerator(n=2)
    print("Welcome to Predictive Text Generator")
    while True:
        print("\nOptions:\n1. Predict Next Word\n2. Add Custom Word\n3. Exit")
        choice = input("Enter choice: ")
        if choice == '1':
            input_text = input("Enter your sentence: ")
            prediction = ptg.predict(input_text)
            if prediction:
                print(f"Suggested next word: {prediction}")
        elif choice == '2':
            word = input("Enter word to add: ")
            ptg.add_custom_word(word)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
