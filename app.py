import streamlit as st
import requests
from models.name_matcher import NameMatcher

st.set_page_config(page_title="Name Matching & Recipe Chatbot", layout="wide")

name_matcher = NameMatcher('data/names.json')

tab1, tab2 = st.tabs(["Name Matching Tool", "Recipe Chatbot"])

with tab1:
    st.header("Name Similarity Matching")
    st.markdown("""
    **Example:** Enter 'John Doe' to find similar names like 'Jane Smith' or 'Bob Brown' based on string similarity.
    
    **How it works:** This tool uses Python's difflib library to compute similarity scores between the input name and a database of names, ranking matches by ratio.
    """)
    input_name = st.text_input("Enter a name to find matches:")
    if input_name:
        best_match, best_score = name_matcher.find_best_match(input_name)
        st.subheader("Best Match")
        st.write(f"Name: {best_match}, Score: {best_score:.2f}")
        
        st.subheader("Ranked Matches")
        matches = name_matcher.find_ranked_matches(input_name)
        for name, score in matches:
            st.write(f"{name}: {score:.2f}")

with tab2:
    st.header("Recipe Chatbot")
    st.markdown("""
    **Example:** Enter 'eggs, onions, tomatoes' to get a recipe suggestion like "Eggs with Onions and Tomatoes: Sauté onions and tomatoes, add scrambled eggs, season with salt and pepper."
    
    **Steps:**
    1. Enter ingredients separated by commas (e.g., eggs, onions, tomatoes).
    2. Click the "Get Recipe" button.
    3. View the AI-generated recipe based on the ingredients.
    
    **How responses are generated:** Recipes are created using a fine-tuned GPT-2 language model from Hugging Face, trained on ingredient-recipe pairs to generate coherent cooking instructions from prompts.
    """)
    ingredients_input = st.text_input("Enter ingredients (comma separated):")
    if st.button("Get Recipe"):
        if ingredients_input:
            ingredients = [ing.strip() for ing in ingredients_input.split(',')]
            response = requests.post("http://localhost:8000/recipe", json={"ingredients": ingredients})
            if response.status_code == 200:
                recipe = response.json()["recipe"]
                st.write(recipe)
            else:
                st.error("Error fetching recipe")
        else:
            st.error("Please enter ingredients")