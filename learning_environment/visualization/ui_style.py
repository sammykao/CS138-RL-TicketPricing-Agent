"""
UI style constants: colors, fonts, and layout definitions.
"""

import pygame
from typing import Tuple

# Window dimensions
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 720
FPS = 60

# Color palette (dark theme with accent colors)
COLORS = {
    # Backgrounds
    'bg_dark': (18, 18, 24),
    'bg_panel': (28, 28, 36),
    'bg_hud': (24, 24, 32),
    
    # Text
    'text_primary': (255, 255, 255),
    'text_secondary': (180, 180, 190),
    'text_muted': (120, 120, 130),
    
    # Accent colors
    'accent_blue': (64, 128, 255),
    'accent_green': (64, 255, 128),
    'accent_red': (255, 96, 96),
    'accent_yellow': (255, 220, 64),
    'accent_purple': (192, 128, 255),
    
    # Status colors
    'success': (64, 255, 128),
    'warning': (255, 220, 64),
    'error': (255, 96, 96),
    'info': (64, 128, 255),
    
    # Action colors
    'action_up': (64, 255, 128),      # Green for price increase
    'action_down': (255, 96, 96),     # Red for price decrease
    'action_hold': (64, 128, 255),    # Blue for hold
    
    # Graph colors
    'graph_line': (64, 128, 255),
    'graph_line_actual': (64, 255, 128),
    'graph_grid': (40, 40, 50),
    'graph_axis': (100, 100, 110),
    'graph_marker': (255, 220, 64),
    
    # Time bar
    'time_bar_full': (64, 128, 255),
    'time_bar_empty': (60, 60, 70),
    
    # Reward colors
    'reward_positive': (64, 255, 128),
    'reward_negative': (255, 96, 96),
    'reward_neutral': (180, 180, 190),
}

# Layout constants
LAYOUT = {
    # Top time bar
    'time_bar_height': 8,
    'time_bar_padding': 10,
    
    # Left price panel
    'price_panel_width': 280,
    'price_panel_padding': 20,
    'price_panel_spacing': 15,
    
    # Right graph area
    'graph_area_width': 600,
    'graph_area_padding': 20,
    'graph_height': 280,
    'graph_spacing': 20,
    
    # Center action indicator
    'action_indicator_size': 120,
    
    # Bottom HUD
    'hud_height': 180,
    'hud_padding': 15,
    'hud_line_height': 20,
    'hud_column_width': 200,
    
    # General spacing
    'margin': 10,
    'border_radius': 8,
}

# Font sizes
FONT_SIZES = {
    'huge': 48,
    'large': 32,
    'medium': 24,
    'small': 18,
    'tiny': 14,
}

# Animation durations (in frames at 60 FPS)
ANIMATION_DURATIONS = {
    'action_pulse': 20,      # ~0.33 seconds
    'sale_celebration': 60,  # ~1 second
    'failure_flash': 40,     # ~0.67 seconds
    'reward_flash': 30,      # ~0.5 seconds
}

def init_fonts() -> dict:
    """Initialize pygame fonts."""
    pygame.font.init()
    fonts = {}
    for name, size in FONT_SIZES.items():
        try:
            fonts[name] = pygame.font.Font(None, size)
        except:
            fonts[name] = pygame.font.SysFont('monospace', size)
    return fonts

def get_monospace_font(size: int) -> pygame.font.Font:
    """Get a monospace font of given size."""
    try:
        return pygame.font.Font(None, size)
    except:
        return pygame.font.SysFont('monospace', size)

def get_layout_rects() -> dict:
    """Calculate layout rectangles for UI elements."""
    return {
        # Time bar at top
        'time_bar': pygame.Rect(
            LAYOUT['time_bar_padding'],
            LAYOUT['time_bar_padding'],
            WINDOW_WIDTH - 2 * LAYOUT['time_bar_padding'],
            LAYOUT['time_bar_height']
        ),
        
        # Left price panel (metrics sidebar)
        'price_panel': pygame.Rect(
            LAYOUT['margin'],
            LAYOUT['time_bar_padding'] + LAYOUT['time_bar_height'] + LAYOUT['margin'],
            LAYOUT['price_panel_width'],
            WINDOW_HEIGHT - LAYOUT['time_bar_padding'] - LAYOUT['time_bar_height'] - 2 * LAYOUT['margin']
        ),
        
        # Right graph area
        'graph_area': pygame.Rect(
            WINDOW_WIDTH - LAYOUT['graph_area_width'] - LAYOUT['margin'],
            LAYOUT['time_bar_padding'] + LAYOUT['time_bar_height'] + LAYOUT['margin'],
            LAYOUT['graph_area_width'],
            WINDOW_HEIGHT - LAYOUT['time_bar_padding'] - LAYOUT['time_bar_height'] - 2 * LAYOUT['margin']
        ),
        
        # Center action indicator area
        'action_area': pygame.Rect(
            LAYOUT['price_panel_width'] + LAYOUT['margin'],
            LAYOUT['time_bar_padding'] + LAYOUT['time_bar_height'] + LAYOUT['margin'],
            WINDOW_WIDTH - LAYOUT['price_panel_width'] - LAYOUT['graph_area_width'] - 3 * LAYOUT['margin'],
            WINDOW_HEIGHT - LAYOUT['time_bar_padding'] - LAYOUT['time_bar_height'] - 2 * LAYOUT['margin']
        ),
    }

