import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\Coding scripts\Oasis\Email Spam Detection\spam.csv", encoding='latin-1')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v1'], data['v2'], test_size=0.2, random_state=42)

# Convert the text data into a matrix of token counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training set
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_counts)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Example text message to predict
text_message = ["Congratulations! You've won a free trip to Hawaii. Reply 'YES' to claim your prize."]

# Convert the text message into a matrix of token counts using the same vectorizer used for training
text_message_counts = vectorizer.transform(text_message)

# Make prediction on the text message using the trained classifier
prediction = clf.predict(text_message_counts)

# Print the prediction
if prediction[0] == 'spam':
    print("The text message is spam.")
else:
    print("The text message is not spam.")
