import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RecipeModel:
    def __init__(self, recipes_file, model_path='./fine_tuned_model'):
        self.recipes = json.load(open(recipes_file, 'r'))
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        except:
            # Fallback to base model if fine-tuned not available
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
    
    def generate_recipe(self, ingredients):
        ingredients_str = ', '.join(ingredients)
        prompt = f"Ingredients: {ingredients_str}. Provide a full recipe with detailed steps: "
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=300,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the recipe part after the prompt
        if "Provide a full recipe with detailed steps:" in full_text:
            recipe = full_text.split("Provide a full recipe with detailed steps:")[1].strip()
        else:
            recipe = full_text.replace(prompt, "").strip()
        # Ensure it's not cut off; if too short, regenerate or fallback
        if len(recipe) < 50:
            recipe = "Sorry, unable to generate a complete recipe. Try different ingredients."
        return recipe