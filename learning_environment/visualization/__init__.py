"""
Real-time visualization system for RL ticket pricing agent.

Provides a pygame-based live visualization system with:
- Episode view mode (step-by-step)
- Training view mode (aggregate metrics)
- Live graphs and HUD elements
- Action animations and feedback
"""

from .app import VisualizationApp

__all__ = ['VisualizationApp']

