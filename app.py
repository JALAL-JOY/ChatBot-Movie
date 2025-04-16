import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load cleaned IMDb data using the full Windows file path (raw string)
df = pd.read_csv("imdb_cleaned.csv")

# Check if the 'resume' column exists; if not, print available columns
if 'resume' not in df.columns:
    st.error("The 'resume' column was not found in the CSV file. Available columns: " + ", ".join(df.columns))
    st.stop()

# TF-IDF vectorizer for movie summaries
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['resume'])

# Set up Streamlit configuration
st.set_page_config(page_title="Smart Movie Chatbot", layout="wide")


# --- Custom HTML for visual design ---
import streamlit as st
import streamlit.components.v1 as components
import base64

# Convert the image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return b64_string

# Get the base64 version of Filmo.png
filmo_img_b64 = get_image_base64("Filmo.png")

# Create the HTML image tag with base64
filmo_img_tag = f'<img src="data:image/png;base64,{filmo_img_b64}" alt="Filmo Image" width="350" style="border-radius: 2rem; box-shadow: 0 0 20px #a855f7;" />'

# Display custom HTML using Streamlit components
components.html(f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <style>
    body {{
      background: linear-gradient(90deg, rgba(39,12,126,0.47) 0%, rgba(64,150,192,1) 0%, rgba(27,0,36,1) 93%);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .all {{
      display: flex;
      flex-direction: row;
      justify-content: space-around;
      align-items: center;
      padding: 10px;
    }}
    .text {{
      color: white;
      padding-top: 10px;
    }}
    .text h1 {{
      font-size: 4.5rem;
      margin: 0 0 20px 0;
      background: linear-gradient(45deg, #03384c, #b6e7ef, #09282e);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }}
    .text p {{
      font-size: 20px;
      line-height: 1.5;
      color: #ccc;
    }}
  </style>
</head>
<body>
  <div class="all">
    <div class="text">
      <h1><b>ASK FILMO</b></h1>
      <p>
        Filmo is a friendly movie chatbot that can help you discover movies.<br />
        Speak your query via your microphone or type it in. Enjoy exploring films!
      </p>
    </div>
    {filmo_img_tag}
  </div>
</body>
</html>
""", height=500)

# --- Improved Natural Language Query Parser ---
def parse_query(query):
    query = query.lower()
    filters = {}

    # Extract years (e.g., 90s, 2020-2025, etc.)
    years = re.findall(r"(19\d{2}|20\d{2})|(\d{2})\s?(?:s|st)", query)
    if years:
        filters['date'] = []
        for year in years:
            if year[0]:
                filters['date'].append(int(year[0]))
            elif year[1]:
                filters['date'].append(1900 + int(year[1]))  # e.g., '90s' -> 1990

    # Rating handling
    if "high rating" in query or "bien noté" in query or "well rated" in query:
        filters['rate'] = (7.0, 10.0)
    elif "top" in query or "best" in query:
        filters['rate'] = (8.0, 10.0)
    elif "low rating" in query:
        filters['rate'] = (0.0, 4.0)

    # Genre keywords (expanded)
    genres = ["action", "comedy", "thriller", "romance", "horror", "animation", "drama", "fantasy", "documentary", "crime", "adventure"]
    for genre in genres:
        if genre in query:
            filters['genre'] = genre

    # Handle complex keyword searches
    filters['keywords'] = query

    return filters

# --- Filter & Search Function with Dynamic Range (Sliders) ---
def search_movies(filters):
    results = df.copy()

    # Dynamic year range slider
    if 'date' in filters:
        min_year = st.slider("Select year range", int(df['date'].min()), int(df['date'].max()),
                               (int(df['date'].min()), int(df['date'].max())))
        results = results[(results['date'] >= min_year[0]) & (results['date'] <= min_year[1])]

    # Filter by rating range slider
    if 'rate' in filters:
        min_r, max_r = filters['rate']
        rating_range = st.slider("Select rating range", 0.0, 10.0, (min_r, max_r))
        results = results[(results['rate'] >= rating_range[0]) & (results['rate'] <= rating_range[1])]

    # Filter by genre (using multiselect)
    selected_genre = st.multiselect("Choose genres", ["Action", "Comedy", "Thriller", "Romance", "Horror", "Animation", "Drama", "Fantasy", "Documentary", "Crime", "Adventure"])
    if selected_genre:
        results = results[results['resume'].str.contains('|'.join(selected_genre), case=False, na=False)]

    # Keyword search in title and description
    if 'keywords' in filters:
        kw = filters['keywords']
        results = results[results['nom'].str.contains(kw, case=False, na=False) |
                          results['resume'].str.contains(kw, case=False, na=False)]

    return results.head(15)

# --- Smart Recommendations ---
def recommend_movies(title):
    matches = difflib.get_close_matches(title, df['nom'], n=1, cutoff=0.6)
    if not matches:
        return []
    idx = df[df['nom'] == matches[0]].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:6]  # Top 5 similar movies
    return df.iloc[similar_indices]

# --- User Feedback ---
def movie_feedback():
    st.write("Did you like the movie suggestions?")
    like = st.button("👍 Yes")
    dislike = st.button("👎 No")
    if like:
        st.write("Great! We'll try to suggest more like this.")
    if dislike:
        st.write("Sorry about that! We'll improve our suggestions next time.")

import streamlit as st

# Inject bold, glowing custom styles for the search bar
import streamlit as st

# Inject bold, glowing custom styles for the search bar
st.markdown("""
    <style>
    /* Pulsing animation for glowing effect */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 15px #a855f7, 0 0 25px #9333ea, 0 0 40px #7e22ce;
        }
        50% {
            box-shadow: 0 0 30px #d946ef, 0 0 50px #c026d3, 0 0 80px #7c3aed;
        }
        100% {
            box-shadow: 0 0 15px #a855f7, 0 0 25px #9333ea, 0 0 40px #7e22ce;
        }
    }

    /* Customizing input field for better design */
    .stTextInput>div>div>input {
        background-color: #2e003e; /* Dark background for input */
        border: 2px solid #9333ea; /* Purple border */
        border-radius: 1.5rem; /* Rounded corners */
        padding: 1rem 1.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #fff; /* White text */
        animation: pulse 3s infinite; /* Glowing pulse effect */
        transition: all 0.4s ease-in-out; /* Smooth transitions */
    }

    /* Focus effect on input */
    .stTextInput>div>div>input:focus {
        border-color: #f0abfc;
        box-shadow: 0 0 15px #f0abfc, 0 0 25px #e879f9, 0 0 35px #c026d3;
        outline: none; /* Remove default outline */
    }

    /* Label style for search input */
    label[for="search_input"] {
        font-size: 1.3rem;
        color: #f0abfc;
        text-shadow: 0 0 8px #9333ea, 0 0 15px #e879f9;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    /* Placeholder style inside input field */
    ::placeholder {
        color: #d8b4fe;
        opacity: 0.7;
        font-style: italic;
    }

    /* Hover effect on the input field */
    .stTextInput>div>div>input:hover {
        background-color: #3b0657;
        border-color: #c026d3;
    }

    /* Add subtle glowing effect to the placeholder */
    ::placeholder {
        text-shadow: 0 0 10px #a855f7, 0 0 15px #7c3aed, 0 0 20px #9333ea;
    }

    /* Optional - Add a smooth pulse animation to label when focused */
    .stTextInput:focus-within label[for="search_input"] {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Search input with custom glowing style
with st.form(key="search_form"):
    query = st.text_input("🔮 Entrez un film, une question ou un genre magique:", key="search_input")
    search_button = st.form_submit_button("🔍 Rechercher")
# When the user provides a query, filter and display the results
if query:
    filters = parse_query(query)
    results = search_movies(filters)
    
    # If results are found, display them
    if not results.empty:
        st.subheader("🎯 Résultats de recherche")
        for _, row in results.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(row['cover'], width=120)
            with col2:
                st.markdown(f"{row['nom']} ({row['date']})** - ⭐ {row['rate']}/10")
                st.markdown(row['resume'])
                with st.expander("🔁 Suggestions similaires"):
                    recs = recommend_movies(row['nom'])
                    for _, sim in recs.iterrows():
                        st.markdown(f"- {sim['nom']} ({sim['date']})")
    else:
        st.warning("😓 Aucun film trouvé... Essayez une autre incantation.")
    
    # Movie feedback for user satisfaction
    movie_feedback()