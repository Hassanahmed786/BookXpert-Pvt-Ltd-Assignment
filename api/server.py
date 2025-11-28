from fastapi import FastAPI
from pydantic import BaseModel
from models.recipe_model import RecipeModel

app = FastAPI()

class RecipeRequest(BaseModel):
    ingredients: list[str]

recipe_model = RecipeModel('data/recipes.json')

@app.post("/recipe")
async def get_recipe(request: RecipeRequest):
    print("Received request:", request.ingredients)
    recipe = recipe_model.generate_recipe(request.ingredients)
    print("Generated recipe:", recipe)
    return {"recipe": recipe}