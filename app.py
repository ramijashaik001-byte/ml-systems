import uvicorn
from nexusml.serving.server import app

if __name__ == "__main__":
    print("Launching NexusML Serving App on port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080)
