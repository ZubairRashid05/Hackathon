import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.model_selection import train_test_split

# Load the dataset
data = []
with open('train.csv', 'r') as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Extract relevant columns
df = df[['question', 'answer']]

# Split the data into training and testing sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the training questions into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(train_df['question'])

# Function to get a response for a given question
def get_response(question):
    # Transform the input question into TF-IDF features
    question_tfidf = vectorizer.transform([question])

    # Compute cosine similarity between the input question and all training questions
    similarities = cosine_similarity(question_tfidf, X_train_tfidf)

    # Find the index of the most similar question
    most_similar_index = similarities.argmax()

    # Return the corresponding answer
    return train_df.iloc[most_similar_index]['answer']

# Evaluate the model on the test set
correct = 0
for i, row in test_df.iterrows():
    predicted_answer = get_response(row['question'])
    if predicted_answer == row['answer']:
        correct += 1

accuracy = correct / len(test_df)
print("Accuracy:", float(accuracy))

# Example usage
new_question = "What are some effective exercises to target the triceps?"
response = get_response(new_question)
print("Response:", response)