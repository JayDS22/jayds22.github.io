# 🎬 CineMatch: A Netflix-Style Movie Recommender System

> A content-based movie recommendation system with a sleek, Netflix-inspired user interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.herokuapp.com)

---

## 📖 Project Overview

CineMatch is a movie recommender system that uses **content-based filtering** to suggest movies based on user preferences. The system analyzes movie metadata including genres, keywords, cast, crew, and plot overviews to find similar movies.

### Key Features

- **Netflix-Style UI**: Dark theme with smooth animations and modern design
- **Smart Recommendations**: Content-based filtering using cosine similarity
- **Movie Posters**: Integration with TMDB API for dynamic poster display
- **Match Scores**: Visual similarity percentages for each recommendation
- **Responsive Design**: Works on desktop and mobile devices

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Algorithm | Cosine Similarity (sklearn) |
| Data Processing | Pandas, NumPy |
| API | TMDB (The Movie Database) |
| Deployment | Heroku |

---

## 📊 How It Works

### 1. Data Preprocessing

The system uses the **TMDB 5000 Movie Dataset** which contains:
- Movie metadata (genres, keywords, overview)
- Cast and crew information
- Production details

### 2. Feature Engineering

Each movie is represented by a "tag" combining:
- Overview text (stemmed)
- Genre names
- Top 3 cast members
- Director name
- Keywords

### 3. Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])
```

### 4. Similarity Calculation

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```

### 5. Recommendation Generation

```python
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), 
                         reverse=True, key=lambda x:x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]
```

---

## 🚀 Deployment Guide (Heroku)

### Prerequisites

1. [Heroku Account](https://signup.heroku.com/)
2. [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. [Git](https://git-scm.com/)
4. [TMDB API Key](https://www.themoviedb.org/settings/api) (optional, for posters)

### Step-by-Step Deployment

```bash
# 1. Clone or create your repository
git init movie-recommender
cd movie-recommender

# 2. Add all files (app.py, requirements.txt, Procfile, setup.sh, pickle files)

# 3. Login to Heroku
heroku login

# 4. Create a new Heroku app
heroku create your-app-name

# 5. Set config variables (optional, for movie posters)
heroku config:set TMDB_API_KEY=your_api_key_here

# 6. Deploy
git add .
git commit -m "Initial deployment"
git push heroku main

# 7. Open your app
heroku open
```

---

## 📁 Project Structure

```
movie-recommender/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku process file
├── setup.sh              # Streamlit configuration
├── runtime.txt           # Python version
├── movies_dict.pkl       # Movie data (serialized)
├── similarity.pkl        # Similarity matrix (serialized)
└── README.md             # Documentation
```

---

## 🎯 Usage

1. **Select a Movie**: Use the dropdown to search for a movie you've enjoyed
2. **Get Recommendations**: Click the "Get Recommendations" button
3. **Explore Results**: View 10 similar movies with match percentages
4. **Discover More**: Click on movies to learn more about them

---

## 📈 Future Enhancements

- [ ] Collaborative filtering integration
- [ ] User authentication and watch history
- [ ] Movie trailers and streaming links
- [ ] Genre-based filtering
- [ ] Multi-language support

---

## 👨‍💻 Author

**Jay**  
MSML602 - Principles of Data Science  
University of Maryland

---

## 📚 References

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Heroku Python Deployment](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

---

## 📄 License

This project is for educational purposes as part of MSML602 coursework.
