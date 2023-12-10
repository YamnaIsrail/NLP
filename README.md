# Recipe Generation using Fine-tuned GPT-2 Model

## Introduction
This repository contains code for fine-tuning the GPT-2 language model on a dataset of recipes and using the trained model for recipe generation. The code utilizes the Hugging Face Transformers

library for working with pre-trained transformer models and includes steps for data loading, model training, and recipe generation.

## Prerequisites
Before running the code, ensure that you have the following dependencies installed:

- pandas
- torch
- transformers (Hugging Face)

You can install the required packages using the following command:
```bash
pip install pandas torch transformers
```

## Getting Started
Follow the steps below to use the code for fine-tuning and recipe generation:

1. **Install Dependencies:** Make sure to install the required dependencies as mentioned in the prerequisites.

2. **Load Pre-trained Model and Tokenizer:** The code loads the GPT-2 model and tokenizer from the Hugging Face model hub. You can change the `model_name` variable to specify a different pre-trained model.

3. **Load and Preprocess Data:** Provide the path to your recipe dataset in the `train_path` variable. The code reads the data, combines relevant columns, tokenizes the text, and prepares the input IDs.

4. **Data Collator:** Set up a data collator for language modeling with the specified tokenizer and MLM (Masked Language Modeling) set to `False`.

5. **Training Configuration:** Configure the training settings such as output directory, training epochs, batch size, and save intervals in the `training_args` dictionary.

6. **Initialize Optimizer:** The code initializes the AdamW optimizer for training the model.

7. **Training Loop:** The training loop iterates through the dataset, computes the loss, and updates the model parameters. Adjust the training parameters based on your specific requirements.

8. **Save Fine-tuned Model:** After training, the fine-tuned model and tokenizer are saved to the specified output directory.

9. **Load Fine-tuned Model for Recipe Generation:** Load the fine-tuned model and tokenizer from the saved directory for recipe generation.

10. **Recipe Generation:** Run the provided recipe generation script, which prompts the user to enter an item name. The model then generates a recipe for the specified item.

## Notes
- This code assumes a CSV dataset with columns 'Item' and 'Recipes' for training. Adjust the data loading and preprocessing steps according to your dataset structure.
- Experiment with the hyperparameters, such as batch size, learning rate, and training epochs, to achieve optimal results.
- Ensure that the model is fine-tuned on a sufficiently large and diverse recipe dataset for better recipe generation performance.
 Happy cooking! 🍽️
