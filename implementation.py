from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy import integrate, optimize, linalg, stats

app = FastAPI()

class IntegrationRequest(BaseModel):
    function: str
    lower_limit: float
    upper_limit: float

class OptimizationRequest(BaseModel):
    coefficient: float

class LinearEquationRequest(BaseModel):
    a: list
    b: list

class StatisticsRequest(BaseModel):
    samples: list

@app.post('/integrate/')
async def integrate_function(request: IntegrationRequest):
    # Define the function to integrate
    func = eval('lambda x: ' + request.function)
    result = integrate.quad(func, request.lower_limit, request.upper_limit)
    return {'integral': result[0]}

@app.post('/optimize/')
async def optimize_function(request: OptimizationRequest):
    # Define the quadratic function
    def quadratic(x):
        return (x - request.coefficient)**2 + 1
    result = optimize.minimize(quadratic, x0=0)
    return {'minimum_value': result.fun, 'at_x': result.x[0]}

@app.post('/linear_equation/')
async def solve_linear_equation(request: LinearEquationRequest):
    A = np.array(request.a)
    b = np.array(request.b)
    x = linalg.solve(A, b)
    return {'solution': x.tolist()}

@app.post('/statistics/')
async def calculate_statistics(request: StatisticsRequest):
    data = np.array(request.samples)
    mean = np.mean(data)
    std_dev = np.std(data)
    t_stat, p_value = stats.ttest_1samp(data, 0)
    return {'mean': mean, 'std_dev': std_dev, 't_statistic': t_stat, 'p_value': p_value}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

'''This code sets up a FastAPI application with endpoints for:
1. **Integration** of a user-defined function.
2. **Optimization** to find the minimum value of a specified quadratic function.
3. **Solving a linear equation** defined by a matrix and vector.
4. **Calculating statistics** such as mean, standard deviation, t-statistic, and p-value from sample data. 

You can run this code in your local environment after installing the necessary libraries (FastAPI, Pydantic, SciPy, Uvicorn, and NumPy).
'''
