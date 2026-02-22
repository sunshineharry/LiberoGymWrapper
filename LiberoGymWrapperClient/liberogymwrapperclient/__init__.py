"""Libero Gym Wrapper Client - A client library for interacting with remote Libero environments."""

# Re-export the main client class without triggering circular import
from .liberogymwrapperclient_v2 import LiberoGymWrapperClient
print("v2")
__version__ = "0.1.1"
__all__ = ["LiberoGymWrapperClient"]