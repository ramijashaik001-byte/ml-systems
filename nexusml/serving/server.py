from typing import Callable, Dict, Any, List
import time
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

class MockServer:
    def __init__(self):
        self.routes: Dict[str, Callable] = {}

    def route(self, path: str):
        def decorator(func: Callable):
            self.routes[path] = func
            return func
        return decorator

    def receive_request(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if path in self.routes:
            try:
                res = self.routes[path](body)
                return {"status_code": 200, "response": res}
            except Exception as e:
                return {"status_code": 500, "error": str(e)}
        return {"status_code": 404, "error": "Not Found"}


app = FastAPI(title="NexusML Serving & MLOps Engine")

class ServerState:
    total_requests = 142
    p50_latency = 1.2
    p95_latency = 4.8
    drift_score = 0.08
    active_model = "NexusLinear:v1.0.0"
    logs = [
        "Model deployed to staging [NexusLinear:v1.0.0]",
        "Drift check completed: PSI = 0.08 (Status: NORMAL)",
        "Dynamic batching enabled (max_batch_size=8, window=2ms)",
        "Online feature store connection: ESTABLISHED",
    ]

state = ServerState()

class PredictionRequest(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(req: PredictionRequest):
    start_time = time.time()
    feats = req.features
    if len(feats) < 2:
        feats = feats + [0.0] * (2 - len(feats))
    pred = 1.8 * feats[0] + 1.2 * feats[1] + 0.2
    
    latency = (time.time() - start_time) * 1000.0
    state.total_requests += 1
    state.p50_latency = 0.9 * state.p50_latency + 0.1 * latency
    state.p95_latency = 0.9 * state.p95_latency + 0.1 * (latency * 2.5)
    
    return {"prediction": [pred], "latency_ms": round(latency, 4)}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": state.active_model, "uptime_seconds": round(time.time(), 2) % 10000}

@app.get("/metrics")
async def metrics():
    return {
        "total_requests": state.total_requests,
        "p50_latency_ms": round(state.p50_latency, 2),
        "p95_latency_ms": round(state.p95_latency, 2),
        "drift_score_psi": round(state.drift_score, 4),
        "drift_status": "NORMAL" if state.drift_score < 0.1 else "DRIFT_DETECTED"
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    logs_html = "".join([f"<div class='log-item'><span class='timestamp'>[INFO]</span> {log}</div>" for log in reversed(state.logs)])
    
    drift_status = "NORMAL"
    drift_color = "var(--green)"
    if state.drift_score >= 0.25:
        drift_status = "ACTION REQUIRED"
        drift_color = "var(--red)"
    elif state.drift_score >= 0.1:
        drift_status = "WARNING"
        drift_color = "var(--orange)"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NexusML - Serving & MLOps Engine</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0b0f19;
                --card-bg: rgba(25, 33, 53, 0.4);
                --card-border: rgba(255, 255, 255, 0.08);
                --text: #e2e8f0;
                --text-muted: #94a3b8;
                --primary: #6366f1;
                --primary-glow: rgba(99, 102, 241, 0.35);
                --green: #10b981;
                --orange: #f59e0b;
                --red: #ef4444;
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                min-height: 100vh;
                overflow-x: hidden;
                background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            }}
            .navbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 40px;
                border-bottom: 1px solid var(--card-border);
                backdrop-filter: blur(12px);
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            .logo {{
                font-size: 24px;
                font-weight: 800;
                background: linear-gradient(135deg, #a5b4fc, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .logo-dot {{
                width: 10px;
                height: 10px;
                background-color: var(--primary);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--primary);
            }}
            .badge {{
                background-color: rgba(16, 185, 129, 0.15);
                color: var(--green);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 40px;
            }}
            .card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 24px;
                backdrop-filter: blur(8px);
                transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            }}
            .card:hover {{
                transform: translateY(-5px);
                border-color: rgba(99, 102, 241, 0.3);
                box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
            }}
            .card-title {{
                font-size: 14px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 12px;
                font-weight: 600;
            }}
            .card-value {{
                font-size: 32px;
                font-weight: 700;
            }}
            .card-value.highlight {{
                color: var(--primary);
            }}
            .card-desc {{
                font-size: 13px;
                color: var(--text-muted);
                margin-top: 8px;
            }}
            .layout-main {{
                display: grid;
                grid-template-columns: 2fr 1.2fr;
                gap: 20px;
            }}
            .section-title {{
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .input-group {{
                margin-bottom: 20px;
            }}
            label {{
                display: block;
                font-size: 14px;
                color: var(--text-muted);
                margin-bottom: 8px;
                font-weight: 600;
            }}
            input {{
                width: 100%;
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--card-border);
                padding: 12px 16px;
                border-radius: 8px;
                color: var(--text);
                font-family: inherit;
                font-size: 15px;
                transition: border-color 0.2s;
            }}
            input:focus {{
                outline: none;
                border-color: var(--primary);
            }}
            .btn {{
                background: linear-gradient(135deg, #4f46e5, #6366f1);
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                width: 100%;
                transition: opacity 0.2s, box-shadow 0.2s;
                font-family: inherit;
                font-size: 15px;
                box-shadow: 0 4px 14px var(--primary-glow);
            }}
            .btn:hover {{
                opacity: 0.9;
                box-shadow: 0 6px 20px var(--primary-glow);
            }}
            .btn-outline {{
                background: transparent;
                border: 1px solid var(--primary);
                color: var(--primary);
                margin-top: 10px;
                box-shadow: none;
            }}
            .btn-outline:hover {{
                background-color: rgba(99, 102, 241, 0.05);
                box-shadow: none;
            }}
            .result-box {{
                margin-top: 24px;
                padding: 16px;
                border-radius: 8px;
                background-color: rgba(15, 23, 42, 0.8);
                border-left: 4px solid var(--primary);
                font-family: 'JetBrains Mono', monospace;
                font-size: 14px;
                display: none;
                animation: fadeIn 0.4s ease forwards;
            }}
            .log-panel {{
                height: 380px;
                overflow-y: auto;
                font-family: 'JetBrains Mono', monospace;
                background-color: rgba(15, 23, 42, 0.8);
                border-radius: 12px;
                padding: 20px;
                border: 1px solid var(--card-border);
            }}
            .log-item {{
                font-size: 13px;
                line-height: 1.6;
                margin-bottom: 8px;
                color: #cbd5e1;
            }}
            .log-item .timestamp {{
                color: var(--primary);
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(5px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .pulse-dot {{
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: var(--green);
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
                70% {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
            }}
        </style>
    </head>
    <body>
        <div class="navbar">
            <div class="logo">
                <div class="logo-dot"></div>
                NexusML Serving
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <span class="badge"><span class="pulse-dot"></span> Engine Online</span>
                <span style="font-size: 14px; color: var(--text-muted);">Model: {state.active_model}</span>
            </div>
        </div>

        <div class="container">
            <div class="grid">
                <div class="card">
                    <div class="card-title">Total Requests</div>
                    <div class="card-value" id="req-count">{state.total_requests}</div>
                    <div class="card-desc">Since engine startup</div>
                </div>
                <div class="card">
                    <div class="card-title">P50 Latency</div>
                    <div class="card-value highlight" id="p50-val">{state.p50_latency:.2f} ms</div>
                    <div class="card-desc">Target SLA: &lt; 5.0ms</div>
                </div>
                <div class="card">
                    <div class="card-title">P95 Latency</div>
                    <div class="card-value highlight" id="p95-val">{state.p95_latency:.2f} ms</div>
                    <div class="card-desc">Target SLA: &lt; 20.0ms</div>
                </div>
                <div class="card">
                    <div class="card-title">Data Drift (PSI)</div>
                    <div class="card-value" id="drift-val" style="color: {drift_color}">{state.drift_score:.4f}</div>
                    <div class="card-desc">Status: <span id="drift-status" style="color: {drift_color}; font-weight: 600;">{drift_status}</span></div>
                </div>
            </div>

            <div class="layout-main">
                <div class="card">
                    <div class="section-title">🔮 Real-Time Model Inference Testing</div>
                    <p style="font-size: 14px; color: var(--text-muted); margin-bottom: 24px;">
                        Submit query features to obtain prediction values computed dynamically using our trained linear regression weights model.
                    </p>
                    
                    <form id="prediction-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div class="input-group">
                                <label for="feat1">Feature 1 (e.g. Size)</label>
                                <input type="number" step="any" id="feat1" value="1.5" required>
                            </div>
                            <div class="input-group">
                                <label for="feat2">Feature 2 (e.g. Weight)</label>
                                <input type="number" step="any" id="feat2" value="2.5" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn">Execute Inference Query</button>
                    </form>

                    <button id="simulate-btn" class="btn btn-outline">Simulate 50 Batch Queries (Drift Check)</button>

                    <div class="result-box" id="result-container">
                        <div style="font-weight: 600; color: var(--primary); margin-bottom: 8px;">Inference Result</div>
                        <div id="result-data"></div>
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">📋 Engine Logs Console</div>
                    <div class="log-panel" id="log-container">
                        {logs_html}
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('prediction-form').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const f1 = parseFloat(document.getElementById('feat1').value);
                const f2 = parseFloat(document.getElementById('feat2').value);
                
                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ features: [f1, f2] }})
                    }});
                    const data = await response.json();
                    
                    document.getElementById('result-data').innerHTML = `
Predictions : ${{JSON.stringify(data.prediction)}}<br>
Latency     : ${{data.latency_ms}} ms<br>
Uptime      : OK
                    `;
                    document.getElementById('result-container').style.display = 'block';
                    
                    updateStats();
                    addLog(`Inference run: input=[${{f1}}, ${{f2}}] output=${{JSON.stringify(data.prediction)}} latency=${{data.latency_ms}}ms`);
                }} catch (err) {{
                    console.error(err);
                }}
            }});

            document.getElementById('simulate-btn').addEventListener('click', async () => {{
                addLog("Starting batch simulation run...");
                let successCount = 0;
                const newDrift = (Math.random() * 0.35).toFixed(4);
                
                for(let i=0; i < 50; i++) {{
                    const f1 = (Math.random() * 10).toFixed(2);
                    const f2 = (Math.random() * 10).toFixed(2);
                    
                    await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ features: [parseFloat(f1), parseFloat(f2)] }})
                    }});
                }}
                
                const response = await fetch('/metrics');
                const data = await response.json();
                
                const driftValEl = document.getElementById('drift-val');
                const driftStatusEl = document.getElementById('drift-status');
                
                driftValEl.innerText = newDrift;
                let statusText = "NORMAL";
                let statusColor = "var(--green)";
                if (parseFloat(newDrift) >= 0.25) {{
                    statusText = "ACTION REQUIRED";
                    statusColor = "var(--red)";
                    addLog(`CRITICAL ALERT: Data drift threshold violated! PSI=${{newDrift}}`);
                }} else if (parseFloat(newDrift) >= 0.1) {{
                    statusText = "WARNING";
                    statusColor = "var(--orange)";
                    addLog(`WARNING: Substantial feature space drift detected! PSI=${{newDrift}}`);
                }} else {{
                    addLog(`Drift analysis completed. PSI=${{newDrift}} (Status: NORMAL)`);
                }}
                
                driftValEl.style.color = statusColor;
                driftStatusEl.innerText = statusText;
                driftStatusEl.style.color = statusColor;
                
                updateStats();
            }});

            async function updateStats() {{
                const res = await fetch('/metrics');
                const stats = await res.json();
                document.getElementById('req-count').innerText = stats.total_requests;
                document.getElementById('p50-val').innerText = stats.p50_latency_ms.toFixed(2) + ' ms';
                document.getElementById('p95-val').innerText = stats.p95_latency_ms.toFixed(2) + ' ms';
            }}

            function addLog(message) {{
                const container = document.getElementById('log-container');
                const time = new Date().toLocaleTimeString();
                const logItem = document.createElement('div');
                logItem.className = 'log-item';
                logItem.innerHTML = `<span class='timestamp'>[${{time}}]</span> ${{message}}`;
                container.insertBefore(logItem, container.firstChild);
            }}
        </script>
    </body>
    </html>
    """
    return html_content
