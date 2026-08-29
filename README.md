# NexusML: Comprehensive ML System & MLOps Framework

NexusML is an end-to-end machine learning system and MLOps framework implemented in pure Python. It features a custom automatic differentiation (autograd) engine, modular neural layers, distributed cluster training topology simulation, features scaling and data engineering pipeline, Dynamic request batch serving, and conceptual data drift evaluation logic.

---

## Installation

### Prerequisites
- Python 3.12+
- Poetry or pip packages manager

### Using Pip
To install the dependencies and setup the packages:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### Using Poetry
```bash
poetry install
```

---

## Build

To build the project as a package:
```bash
python setup.py sdist bdist_wheel
```

To build the Docker container:
```bash
docker build -t nexusml:latest .
```

---

## Run

### Run the Simulation Demo
To train the model and check serving batches pipelines:
```bash
python main.py
```

### Start the REST API & Dashboard Server
To start the model serving dashboard locally:
```bash
python examples/serve.py
```
After launching, navigate to:
👉 **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## Dependencies

- **FastAPI**: Serving REST endpoints
- **Uvicorn**: ASGI web server
- **Pydantic**: Data schema serialization
- **NumPy**: Linear mathematical calculations

---

## Usage Example

```python
from nexusml.core.tensor import Tensor
from nexusml.core.nn import Linear
from nexusml.core.losses import MSELoss
from nexusml.core.optimizers import SGD

# Initialize autograd tensors
x = Tensor([1.0, 2.0], requires_grad=True)
target = Tensor([5.0])

# Modular Linear layer projection
layer = Linear(2, 1)
pred = layer(x)

# Evaluate Loss and update weights
loss_fn = MSELoss()
loss = loss_fn(pred, target)
loss.backward()

optimizer = SGD(layer.parameters(), lr=0.01)
optimizer.step()
print("Prediction output data:", pred.data)
```
