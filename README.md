# Book-Recommendation-Engine
Book Recommendation Engine Using KNN
# Import necessary libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the dataset
# Ensure that the dataset is in the correct path, otherwise adjust the file path
data = pd.read_csv('book-crossings.csv', sep=';', encoding='latin-1')

# Data Cleaning: Remove users with fewer than 200 ratings and books with fewer than 100 ratings
user_counts = data['userID'].value_counts()
book_counts = data['bookID'].value_counts()

# Filter the data to only include users with at least 200 ratings and books with at least 100 ratings
filtered_data = data[data['userID'].isin(user_counts[user_counts >= 200].index)]
filtered_data = filtered_data[filtered_data['bookID'].isin(book_counts[book_counts >= 100].index)]

# Create a pivot table with users as rows, books as columns, and ratings as values
pivot_table = filtered_data.pivot(index='userID', columns='bookID', values='rating').fillna(0)

# Fit KNN model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(pivot_table)

# Function to get recommendations based on book title
def get_recommends(book_title):
    # Find the book's index
    book_id = data[data['bookTitle'] == book_title]['bookID'].iloc[0]
    
    # Find the book's index in the pivot table
    book_index = pivot_table.columns.get_loc(book_id)
    
    # Get the nearest neighbors
    distances, indices = knn.kneighbors(pivot_table.iloc[:, book_index].values.reshape(1, -1))
    
    recommendations = []
    
    for idx, distance in zip(indices[0], distances[0]):
        recommended_book = data[data['bookID'] == pivot_table.columns[idx]]['bookTitle'].iloc[0]
        recommendations.append([recommended_book, distance])
    
    return [book_title, recommendations]

# Test the function with a book title
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
