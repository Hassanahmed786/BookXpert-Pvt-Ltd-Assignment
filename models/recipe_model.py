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
        prompt = f"Ingredients: {ingredients_str}. Recipe:"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the recipe part
        if "Recipe:" in full_text:
            recipe = full_text.split("Recipe:")[1].strip()
            # Stop at next "Ingredients:" if present
            if "Ingredients:" in recipe:
                recipe = recipe.split("Ingredients:")[0].strip()
            return recipe
        else:
            return full_text