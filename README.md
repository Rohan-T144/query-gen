# **Natural Language to SQL Query Generator with Semantic Search**  

This project is a natural language question answerer, designed to convert plain English questions into SQL queries that can be executed on a database. 

It integrates a fine-tuned Large Language Model (LLM) trained on the WikiSQL dataset with an intelligent table selection mechanism utilizing semantic search with sentence transformer embedding models, allowing for multi-table databases. An extension of this could be used to automate SQL query generation for non-technical users to query databases directly using natural language.  

---

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


---

## **Setup and Usage**
### **1. Prerequisites**
- Python 3.8+
- Install dependencies:
  ```bash
  pip install transformers sentence-transformers sqlite3 mlx mlx-lm numpy
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

---

## Acknowledgments
This is based off the examples and implementations from [Apple's MLX](https://github.com/ml-explore/mlx-examples/tree/main).

<!-- 
---

## **File Structure**
```plaintext
.
├── README.md               # Project documentation
├── qa_cli.py               # Main CLI application
├── generate_sql.py         # SQL generation using fine-tuned LLM
├── table_selection.py      # Table selection logic using semantic similarity
├── setup.sql               # Example SQLite database setup
├── sample.db               # Pre-configured sample database
└── requirements.txt        # Python dependencies
``` -->
<!-- 
---

## **Example Walkthrough**
### **Input Question:**
```bash
Which team has the most wins?
```

### **Generated SQL Query:**
```sql
SELECT name FROM teams WHERE wins = (SELECT MAX(wins) FROM teams);
```

### **Query Result:**
```plaintext
name
-----
Lakers
``` -->
<!-- 
---

## **How It Works**
1. **Table Selection**:  
   The app extracts metadata from the database and uses a semantic similarity model to identify the most relevant table for the question.  

2. **SQL Generation**:  
   The selected table schema and the question are combined into a prompt for the fine-tuned LLM, which outputs the corresponding SQL query.  

3. **SQL Execution**:  
   The query is executed on the database, and the results are displayed to the user.

---

## **Future Enhancements**
- **Multi-Table Querying**: Extend support for complex questions that require joins across multiple tables.  
- **Dynamic Schema Learning**: Automatically update the metadata when the database schema changes.  
- **Web App Integration**: Deploy as a web-based app using Flask or Streamlit.  
- **Confidence Scoring**: Show confidence levels for the generated SQL query, with an option for user validation.   -->
<!-- 
---

## **Key Skills Demonstrated**
- **LLM Fine-Tuning**: Customized a language model to handle SQL generation tasks with high accuracy.  
- **Semantic Similarity**: Designed an intelligent table selection mechanism using embedding-based similarity models.  
- **Database Management**: Implemented robust database querying and schema handling techniques.  
- **Software Development**: Built a CLI tool with extensibility and real-world applications in mind.  

---

## **Project Goals**
This project showcases advanced knowledge in **natural language processing**, **large language models**, and **database interaction**, making it an ideal addition to applications for internships and roles in:  
- **Machine Learning/AI**: Demonstrates LLM fine-tuning and intelligent system design.  
- **Data Engineering**: Highlights database querying and schema analysis.  
- **Software Engineering**: Reflects skills in building scalable, user-focused tools.  

---

## **Contact**
**Rohan Timmaraju**  
[GitHub Profile](https://github.com/RohanTimmaraju)  
Feel free to reach out for any questions or collaboration opportunities!   -->