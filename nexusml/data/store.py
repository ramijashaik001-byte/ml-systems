from typing import Dict, Any, List, Optional
import time

class FeatureStore:
    def __init__(self):
        self.offline_db: List[Dict[str, Any]] = []
        self.online_cache: Dict[str, Dict[str, Any]] = {}
        self.feature_schemas: Dict[str, str] = {}

    def register_feature(self, feature_name: str, data_type: str):
        self.feature_schemas[feature_name] = data_type

    def write_features(self, entity_id: str, features: Dict[str, Any]):
        timestamp = time.time()
        for f, val in features.items():
            if f not in self.feature_schemas:
                raise ValueError(f"Feature '{f}' not registered")
        
        self.online_cache[entity_id] = {**features, "timestamp": timestamp}
        self.offline_db.append({
            "entity_id": entity_id,
            "features": features,
            "timestamp": timestamp
        })

    def read_online_features(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.online_cache.get(entity_id)

    def read_historical_features(self, entity_id: str, before_timestamp: float) -> List[Dict[str, Any]]:
        results = []
        for entry in self.offline_db:
            if entry["entity_id"] == entity_id and entry["timestamp"] <= before_timestamp:
                results.append(entry)
        return results
