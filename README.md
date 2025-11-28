# Name Matching & Recipe Chatbot Project

This project implements a name similarity matching tool and a recipe chatbot using a local fine-tuned LLM.

## Tech Stack
- **Frontend:** Streamlit (for the web UI)
- **Backend:** FastAPI (for the API server)
- **LLM:** GPT-2 (fine-tuned for recipe generation)
- **Libraries:** Transformers, PyTorch, difflib (for name matching)
- **Data:** JSON files for names and recipes

## Implementation Details
- **Name Matching:** Uses Python's `difflib.SequenceMatcher` for similarity scoring based on string ratios.
- **Recipe Generation:** Employs a fine-tuned GPT-2 model to generate recipes from ingredient lists. The model is trained on a dataset of ingredient-recipe pairs.
- **Fine-Tuning:** We use Hugging Face's Transformers library to fine-tune GPT-2 on custom recipe data. The training data is prepared by formatting recipes as "Ingredients: [list]. Recipe: [description]". The model learns to generate coherent recipes based on input ingredients.
- **API:** FastAPI exposes a `/recipe` endpoint that takes a list of ingredients and returns a generated recipe.
- **UI:** Streamlit provides a two-tab interface: one for name matching and one for recipe chatbot.

## Project Structure
```
project/
│── app.py              (Streamlit UI)
│── api/
│     └── server.py     (FastAPI server + LLM)
│── models/
│     └── name_matcher.py
│     └── recipe_model.py
│── data/
│     └── names.json
│     └── recipes.json (sample fine-tune dataset)
│── train.py            (Fine-tuning script)
│── requirements.txt
│── README.md
```

## Installation
1. Clone or download the project.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install tf-keras  # For compatibility with transformers
   ```

## Fine-Tuning the Model
To fine-tune the GPT-2 model with the recipes data:
```
python train.py
```
This prepares the data from `data/recipes.json`, fine-tunes GPT-2 for 3 epochs, and saves the model to `./fine_tuned_model`. If fine-tuning fails, the app falls back to the base GPT-2 model.

## Running the Application
1. Start the FastAPI server:
   ```
   uvicorn api.server:app --reload
   ```
   The server runs on http://localhost:8000

2. In a new terminal, start the Streamlit app:
   ```
   streamlit run app.py
   ```
   The UI opens in your browser.

## Usage
- **Name Matching Tool:** Enter a name to find the best match and ranked list of similar names.
- **Recipe Chatbot:** Enter ingredients (comma-separated), click "Get Recipe" to receive an AI-generated recipe.

## API Keys
None required. GPT-2 runs locally.

## Requirements
- Python 3.8+
- PyTorch
- Works on Windows/Linux
- No external cloud dependencies