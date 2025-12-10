"""
Movie Recommender System - Netflix Style UI
============================================
Generates similarity matrix at runtime - no large pickle files needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# NETFLIX-INSPIRED CSS STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;500;700&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(180deg, #141414 0%, #1a1a2e 50%, #141414 100%);
        color: #ffffff;
    }
    
    .logo {
        font-family: 'Bebas Neue', cursive;
        font-size: 3.5rem;
        background: linear-gradient(90deg, #E50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: 3px;
    }
    
    .tagline {
        font-family: 'Roboto', sans-serif;
        color: #b3b3b3;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    .stSelectbox > div > div {
        background-color: #333333;
        border: 2px solid #E50914;
        border-radius: 8px;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #E50914 0%, #b20710 100%);
        color: white;
        border: none;
        padding: 15px 50px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a 0%, #E50914 100%);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
    }
    
    .movie-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #141420 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.4s ease;
        border: 1px solid #2a2a3a;
    }
    
    .movie-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(229, 9, 20, 0.2);
        border-color: #E50914;
    }
    
    .movie-title {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 12px;
        min-height: 45px;
    }
    
    .section-header {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding-left: 10px;
        border-left: 4px solid #E50914;
    }
    
    .match-score {
        background: linear-gradient(135deg, #E50914, #ff4757);
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    
    .search-container {
        background: rgba(30, 30, 47, 0.8);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #2a2a3a;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TMDB API CONFIGURATION
# ============================================================================

# TMDB API Key for fetching movie posters
TMDB_API_KEY = "cf9270c2b8c60c45de4057135d2c060f"

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    if not TMDB_API_KEY:
        return f"https://via.placeholder.com/500x750/1a1a2e/E50914?text=Movie"
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        return "https://via.placeholder.com/500x750/1a1a2e/E50914?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750/1a1a2e/E50914?text=No+Poster"

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

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
    """Remove spaces from names."""
    return [word.replace(" ", "") for word in word_list]

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_process_data():
    """Load CSV files and generate similarity matrix."""
    
    # Load data
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge datasets
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Handle missing values
    movies.dropna(subset=['overview'], inplace=True)
    
    # Process features
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_top3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: str(x).split())
    
    # Remove spaces
    movies['genres'] = movies['genres'].apply(remove_spaces)
    movies['keywords'] = movies['keywords'].apply(remove_spaces)
    movies['cast'] = movies['cast'].apply(remove_spaces)
    movies['crew'] = movies['crew'].apply(remove_spaces)
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create final dataframe
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    # Vectorize and compute similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return new_df, similarity

# Load data with progress indicator
with st.spinner('🎬 Loading movie database...'):
    try:
        movies, similarity = load_and_process_data()
    except FileNotFoundError:
        st.error("⚠️ CSV files not found. Please ensure tmdb_5000_movies.csv and tmdb_5000_credits.csv are present.")
        st.stop()

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def recommend(movie, num_recommendations=5):
    """Generate movie recommendations based on content similarity."""
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        
        movies_list = sorted(
            list(enumerate(distances)), 
            reverse=True, 
            key=lambda x: x[1]
        )[1:num_recommendations + 1]
        
        recommendations = []
        for i in movies_list:
            movie_data = movies.iloc[i[0]]
            recommendations.append({
                'title': movie_data.title,
                'movie_id': movie_data.movie_id,
                'score': round(i[1] * 100, 1)
            })
        
        return recommendations
    except IndexError:
        return []

# ============================================================================
# UI COMPONENTS
# ============================================================================

st.markdown('<h1 class="logo">🎬 CINEMATCH</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Discover Your Next Favorite Movie</p>', unsafe_allow_html=True)

st.markdown('<div class="search-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    selected_movie = st.selectbox(
        "🔍 Search for a movie you love:",
        movies['title'].values,
        index=0,
        help="Select a movie to get personalized recommendations"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_button = st.button("🎯 Get Recommendations", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# DISPLAY RECOMMENDATIONS
# ============================================================================

if recommend_button:
    with st.spinner('🎬 Finding perfect matches for you...'):
        recommendations = recommend(selected_movie, num_recommendations=10)
    
    if recommendations:
        st.markdown(f'<div class="section-header">Because you liked "{selected_movie}"</div>', 
                   unsafe_allow_html=True)
        
        for row in range(0, len(recommendations), 5):
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                if row + idx < len(recommendations):
                    movie = recommendations[row + idx]
                    with col:
                        poster_url = fetch_poster(movie['movie_id'])
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" 
                                 style="width:100%; height:280px; object-fit:cover; border-radius:8px;">
                            <div class="movie-title">{movie['title']}</div>
                            <div class="match-score">{movie['score']}% Match</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.expander("📊 How does this work?"):
            st.markdown("""
            **CineMatch** uses a content-based filtering algorithm:
            
            1. **Content Analysis**: Each movie is represented by genres, keywords, cast, crew, and overview
            2. **Text Vectorization**: Features are converted to vectors using CountVectorizer
            3. **Similarity Calculation**: Cosine similarity measures how alike movies are
            4. **Ranking**: Movies are ranked by similarity score
            
            The higher the match percentage, the more similar the movie!
            """)
    else:
        st.warning("Couldn't find recommendations. Please try another movie.")

st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with ❤️ using Streamlit | MSML602 Data Science Project</p>
</div>
""", unsafe_allow_html=True)