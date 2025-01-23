# **Natural Language to SQL Query Generator with Semantic Search**

This project is a natural language question answerer, designed to convert plain English questions into SQL queries that can be executed on a database.

It integrates a fine-tuned Large Language Model (LLM) trained on the WikiSQL dataset with an intelligent table selection mechanism utilizing semantic search with sentence transformer embedding models, allowing for multi-table databases. An extension of this could be used to automate SQL query generation for non-technical users to query databases directly using natural language.


## **Features**

1. **SQL Query Generation**
   - Fine tunes the given LLM (by default `mlx-community/Mistral-7B-Instruct-v0.3-4bit`) on the [WikiSQL dataset](https://github.com/salesforce/WikiSQL/tree/master)
   - Generates accurate SQL queries based on table schemas and user-provided questions.

2. **Dynamic Table Selection with Semantic Search**
   - Automatically identifies the most relevant table(s) for a given question using semantic similarity from the database metadata and schema.
   - Handles multi-table databases efficiently.
   - Caches the semantic search corpus data between runs to significantly speed up the response.

3. **Command-Line Interface (CLI)**
   - Simple CLI for generating the data, fine-tuning the model, and asking questions and retrieving answers.

4. **Customizable and Extensible**
   - Easy to adapt the project to new datasets, fine-tuned models, and database schemas.


## **Setup and Usage**
### **1. Prerequisites**
- Python 3.8+
- Install dependencies:
  ```bash
  pip install .
  ```

### **2. Setup Data**
Download and convert the data into the correct format using `data_gen.py`
```bash
python data_gen.py
```
### **3. Fine-Tune the Model**
Train the model and save the weights as adapters.npz
```bash
python lora.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --train \
        --iters 150 \
        --save-every 10
```
If you want to resume a previously trained model, you can use
```bash
python lora.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --train \
        --iters 150 \
        --save-every 10 \
        --resume-adapter-file adapters.npz
```
### **4. Run the Query**
There are several options to tweak, which can be found in `query.py`.
```bash
python query.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --query "Which year did John Wallace play for Syracuse?"
```

## Acknowledgments
This is based off the examples and implementations from [Apple's MLX](https://github.com/ml-explore/mlx-examples/tree/main).
