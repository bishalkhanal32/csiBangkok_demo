from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pickle
import numpy as np

# Pydantic model for input data validation
class InputData(BaseModel):
    BeamSpan: float = Field(..., title="Beam Span")
    BeforeSpan: float = Field(..., title="Before Span")
    AfterSpan: float = Field(..., title="After Span")
    Load: float = Field(..., title="Load")
    Depth: float = Field(..., title="Depth")
    Width: float = Field(..., title="Width")
    ConcreteFc: float = Field(..., title="Concrete Fc")
    RebarsFy: float = Field(..., title="Rebars Fy")

# Function to load your trained model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        # Handle model loading errors
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

app = FastAPI()
model = load_model("/app/multioutput_model.pkl")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(data: InputData):
    """
    API endpoint to receive input data and return predictions.
    """
    try:
        input_data = np.array([data.BeamSpan, data.BeforeSpan, data.AfterSpan, data.Load, data.Depth, data.Width, data.ConcreteFc, data.RebarsFy])
        predictions = model.predict(input_data.reshape(1, -1))
        return predictions.tolist()
    except Exception as e:
        # Handle prediction errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
