# FastAPI Word Embeddings Demo App

## Getting Started

Recommended: Setup a virtual environment, for example on linux systems:

```
virtualenv env
source .env/bin/activate 
```

To run this application, you will need to install the requirements. If you haven't installed these yet, you can do so by running:

```
pip install -r requirements.txt
```

Once installed, you can start the server by running this command from the /src/gx/ directory:

```
uvicorn app:app --reload
```

## Endpoints

The application provides the following endpoints:

### 1. `/prepare`

**Method:** GET

Initializes the application by loading word embeddings from CSV file.

- **No input parameters.**

### 2. `/assign_embeddings`

**Method:** GET

Assigns embeddings to words and saves the results to a specified output path.

- **Optional Query Parameters:**
  - `output_path` (str): The file path to save the assigned embeddings. Defaults to "../../data/emb_assigned_to_words.csv".

### 3. `/cosine_distance_all_phrases`

**Method:** GET

Calculates the cosine distance between all phrases and saves the results to a specified output path.

- **Optional Query Parameters:**
  - `output_path` (str): The file path to save the calculated distances. Defaults to "../../data/distances.csv".

### 4. `/get_closest_match`

**Method:** GET

Finds and returns the closest match to a given phrase along with the distance.

- **Required Query Parameters:**
  - `phrase` (str): The phrase to find the closest match for.

## Logging

The application uses Python's built-in logging to log simple info messages about endpoint-task execution.

## Common Module

The application relies on a custom `Solver` class defined in the `common` module for the core functionality.
