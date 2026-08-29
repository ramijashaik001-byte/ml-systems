import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_server():
    print("*" * 60)
    print("NexusML Model Serving & MLOps Engine starting...")
    print("Access your local dashboard at: http://127.0.0.1:8080")
    print("Press Ctrl+C to stop the server.")
    print("*" * 60)
    uvicorn.run("nexusml.serving.server:app", host="127.0.0.1", port=8080, log_level="info")

if __name__ == "__main__":
    start_server()
