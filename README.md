# Homework Solver

I made this simple RAG application to help me with some school tasks. 


# Project Setup
You can choose to run it directly with python or with the docker image.

## Python
Install the packages
```
pip install -r requirements
```

Running
```
streamlit run main.py
```
## Docker

Build the app image
```
docker build -t homework-solver .
```

Running
```
docker run -p 8501:8501 homework-solver
```

## You'll need a OPENAI KEY
Set your OPEN AI KEY on main.py at line 16. 


# How to use it:

### 1- Upload an unscanned document
### 2- Set chunk_size, chunk_overlap and temperature options
### 3- Set the question corresponding to the inserted document
### 4- Select the chunk that returned the content most precisely
### 5- Finally, get the AI Response 