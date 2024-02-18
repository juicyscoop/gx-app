from fastapi import FastAPI
from common import Solver
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# INITIALIZE SOLVER
solver = Solver(embeddings_csv_filename='../../data/vectors.csv')    

@app.get("/prepare")
async def prepare():
    solver.prepare(phrases_csv_filename='../../data/phrases.csv')
    logger.info("Successfully prepared word embeddings.")


# Call this endpoint to solve the first task
@app.get("/assign_embeddings")
async def assign(output_path=None):
    
    if not output_path:
        output_path = "../../data/emb_assigned_to_words.csv"
    
    solver.task1(output_path)
    logger.info(f"Saved assigned embeddings to {output_path}.")
    


# Call this endpoint to solve the second task
@app.get("/cosine_distance_all_phrases")
async def cosine_distance(output_path=None):
    
    if not output_path:
        output_path = "../../data/distances.csv"
    
    solver.task2(output_path)
    logger.info(f"Saved calculated distances to {output_path}.")



# Call this endpoint to solve the third task
@app.get("/get_closest_match")
async def cosine_distance(phrase):
    losest_match, distance = solver.task3(phrase)
    logger.info(f"Closest match: {losest_match} with distance: {distance}.")