from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize

app = FastAPI()

# Define request body schema
class OptimizationRequest(BaseModel):
    initial_guess: float

@app.post("/optimize/")
def optimize(request: OptimizationRequest):
    # Objective function
    def objective(x):
        return x**2 + 5 * np.sin(x)

    # Perform optimization
    result = minimize(objective, [request.initial_guess])
    if result.success:
        return {'optimal_value': result.x[0], 'function_value': result.fun}
    else:
        raise HTTPException(status_code=400, detail='Optimization failed')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)