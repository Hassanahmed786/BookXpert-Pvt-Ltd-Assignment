import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

def prepare_data(recipes_file, output_file):
    with open(recipes_file, 'r') as f:
        recipes = json.load(f)
    
    with open(output_file, 'w') as f:
        for recipe in recipes:
            ingredients = ', '.join(recipe['ingredients'])
            text = f"Ingredients: {ingredients}. Recipe: {recipe['recipe']}\n"
            f.write(text)

def fine_tune_model(train_file, model_name='gpt2', output_dir='./fine_tuned_model'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    prepare_data('data/recipes.json', 'data/train.txt')
    fine_tune_model('data/train.txt')