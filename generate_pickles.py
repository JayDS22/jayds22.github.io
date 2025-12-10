"""
generate_pickles.py
===================
Run this script ONCE to generate the pickle files from your CSV data.
After running, you can delete this script.

Usage:
    python generate_pickles.py
"""

import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 50)
print("Generating Pickle Files for Movie Recommender")
print("=" * 50)

# ============================================================================
# STEP 1: Load CSV Files
# ============================================================================
print("\n[1/6] Loading CSV files...")

try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    print(f"      Movies: {len(movies)} rows")
    print(f"      Credits: {len(credits)} rows")
except FileNotFoundError as e:
    print(f"\nERROR: CSV file not found!")
    print(f"Make sure these files are in the same folder:")
    print(f"  - tmdb_5000_movies.csv")
    print(f"  - tmdb_5000_credits.csv")
    exit(1)

# ============================================================================
# STEP 2: Merge Datasets
# ============================================================================
print("\n[2/6] Merging datasets...")

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
print(f"      Merged dataset: {len(movies)} movies")

# ============================================================================
# STEP 3: Handle Missing Values
# ============================================================================
print("\n[3/6] Cleaning data...")

movies.dropna(inplace=True)
print(f"      After cleaning: {len(movies)} movies")

# ============================================================================
# STEP 4: Feature Engineering
# ============================================================================
print("\n[4/6] Engineering features...")

def convert(obj):
    """Convert JSON string to list of names."""
    result = []
    try:
        for item in ast.literal_eval(obj):
            result.append(item['name'])
    except:
        pass
    return result

def convert_top3(obj):
    """Get top 3 cast members."""
    result = []
    try:
        counter = 0
        for item in ast.literal_eval(obj):
            if counter < 3:
                result.append(item['name'])
                counter += 1
    except:
        pass
    return result

def fetch_director(obj):
    """Extract director name from crew."""
    try:
        for item in ast.literal_eval(obj):
            if item['job'] == 'Director':
                return [item['name']]
    except:
        pass
    return []

def remove_spaces(word_list):
    """Remove spaces from names to treat as single tokens."""
    return [word.replace(" ", "") for word in word_list]

# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_top3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: str(x).split())

# Remove spaces from multi-word names
movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(remove_spaces)

# Combine all features into tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create final dataframe
new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print(f"      Features created for {len(new_df)} movies")

# ============================================================================
# STEP 5: Create Similarity Matrix
# ============================================================================
print("\n[5/6] Computing similarity matrix (this may take a minute)...")

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(f"      Vector shape: {vectors.shape}")

similarity = cosine_similarity(vectors)
print(f"      Similarity matrix: {similarity.shape}")

# ============================================================================
# STEP 6: Save Pickle Files
# ============================================================================
print("\n[6/6] Saving pickle files...")

pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))
print(f"      ✓ movies_dict.pkl saved")

pickle.dump(similarity, open('similarity.pkl', 'wb'))
print(f"      ✓ similarity.pkl saved")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 50)
print("SUCCESS! Pickle files generated.")
print("=" * 50)
print("\nYou can now run your app:")
print("    streamlit run app.py")
print()
