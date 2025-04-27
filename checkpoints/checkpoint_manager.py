# -*- coding: utf-8 -*-

import pickle
from datetime import datetime
from pathlib import Path

# This module provides a simple checkpoint manager for saving and loading
# machine learning model states, including parameters, metrics, and metadata.
class CheckpointManager:
    def __init__(self, checkpoint_dir="../checkpoints/"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, X, y, params, metrics=None, metadata=None):
        """Guarda el estado actual del entrenamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'model': model,
            'data_sample': {
                'X': X.iloc[:100].copy(),  # Guarda muestra representativa
                'y': y.iloc[:100].copy()
            },
            'params': params,
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': timestamp
        }
        
        filename = self.checkpoint_dir / f"checkpoint_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"✅ Checkpoint guardado en {filename}")
        return filename
    
    def load_latest_checkpoint(self):
        """Carga el último checkpoint disponible"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        with open(latest, 'rb') as f:
            data = pickle.dump(f)
        
        print(f"♻️ Checkpoint cargado desde {latest}")
        return data, latest
    
    def cleanup_old_checkpoints(self, keep_last=3):
        """Elimina checkpoints antiguos, manteniendo solo los últimos N"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) > keep_last:
            for old in checkpoints[:-keep_last]:
                old.unlink()
            print(f"🧹 Eliminados {len(checkpoints)-keep_last} checkpoints antiguos")