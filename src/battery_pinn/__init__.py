"""Local (Spark-free, MLflow-free) implementation of the battery PINN digital twin."""
import os

# DeepXDE reads this when it is first imported, so it must be set before any submodule imports it.
os.environ.setdefault("DDE_BACKEND", "pytorch")

__version__ = "0.1.0"
