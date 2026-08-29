from nexusml.core.tensor import Tensor
from nexusml.core.nn import Linear
from nexusml.core.losses import MSELoss
from nexusml.core.optimizers import SGD
from nexusml.data.pipeline import Dataset, DataLoader
from nexusml.serving.engine import DynamicBatcher
from nexusml.monitoring.drift_detector import LatencyTracker
import time

def run_demo():
    print("NexusML: Starting End-to-End Simulation")
    
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    labels = [[5.0], [7.0], [9.0], [11.0]]
    dataset = Dataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    print("Training Model...")
    for epoch in range(10):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            pred = Tensor(0.0)
            t_loss = Tensor(0.0)
            for bx, by in zip(batch_x, batch_y):
                x_tensor = Tensor(bx)
                y_tensor = Tensor(by)
                pred = model(x_tensor)
                t_loss = t_loss + loss_fn(pred, y_tensor)
                
            t_loss.backward()
            optimizer.step()
            total_loss += t_loss.data[0]
        print(f"Epoch {epoch+1}/10, Loss: {total_loss:.4f}")
        
    latency_tracker = LatencyTracker()
    
    def process_inference(batch: list) -> list:
        preds = []
        for x in batch:
            start_time = time.time()
            x_tensor = Tensor(x["features"])
            out = model(x_tensor)
            preds.append({"predictions": out.data})
            latency_tracker.record_latency((time.time() - start_time) * 1000.0)
        return preds

    batcher = DynamicBatcher(process_inference, max_batch_size=2)
    batcher.enqueue({"features": [1.5, 2.5]})
    batcher.enqueue({"features": [2.5, 3.5]})
    batcher.enqueue({"features": [3.5, 4.5]})
    
    print("Running batch serving inference...")
    predictions = batcher.process_queue()
    for pred in predictions:
        print("Served prediction:", pred)
        
    print(f"P50 Latency: {latency_tracker.get_p50():.2f} ms")
    print("Simulation complete.")

if __name__ == '__main__':
    run_demo()
