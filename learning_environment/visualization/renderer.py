"""
Rendering module for HUD elements, graphs, and animations.

Handles all pygame drawing operations for the visualization.
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from .ui_style import COLORS, LAYOUT, FONT_SIZES, ANIMATION_DURATIONS, get_monospace_font, get_layout_rects


class AnimationState:
    """Track animation states for visual effects."""
    
    def __init__(self):
        self.action_pulse_frame = 0
        self.sale_celebration_frame = 0
        self.failure_flash_frame = 0
        self.reward_flash_frame = 0
        self.reward_flash_value = 0.0
        self.circle_angle = 0.0  # For revolving circle animation
    
    def start_action_pulse(self):
        """Start action indicator pulse animation."""
        self.action_pulse_frame = ANIMATION_DURATIONS['action_pulse']
    
    def start_sale_celebration(self):
        """Start sale celebration animation."""
        self.sale_celebration_frame = 60
    
    def start_failure_flash(self):
        """Start failure flash animation."""
        self.failure_flash_frame = 40
    
    def start_reward_flash(self, reward_value: float):
        """Start reward flash animation."""
        self.reward_flash_frame = 30
        self.reward_flash_value = reward_value
    
    def update(self):
        """Update animation frames (decrement)."""
        if self.action_pulse_frame > 0:
            self.action_pulse_frame -= 1
        if self.sale_celebration_frame > 0:
            self.sale_celebration_frame -= 1
        if self.failure_flash_frame > 0:
            self.failure_flash_frame -= 1
        if self.reward_flash_frame > 0:
            self.reward_flash_frame -= 1
        # Update revolving circle angle (continuous rotation)
        self.circle_angle += 0.05  # Adjust speed as needed
        if self.circle_angle >= 2 * np.pi:
            self.circle_angle -= 2 * np.pi


class Renderer:
    """Main renderer class for all UI elements."""
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize renderer.
        
        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen
        self.fonts = {
            'huge': get_monospace_font(FONT_SIZES['huge']),
            'large': get_monospace_font(FONT_SIZES['large']),
            'medium': get_monospace_font(FONT_SIZES['medium']),
            'small': get_monospace_font(FONT_SIZES['small']),
            'tiny': get_monospace_font(FONT_SIZES['tiny']),
        }
        self.layout_rects = get_layout_rects()
        self.animation_state = AnimationState()
        
        # Graph history buffers
        self.demand_curve_prices: Optional[np.ndarray] = None
        self.demand_curve_probs: Optional[np.ndarray] = None
        self.behavior_prices: deque = deque(maxlen=200)
        self.behavior_probs: deque = deque(maxlen=200)
    
    def clear(self):
        """Clear screen with background color."""
        self.screen.fill(COLORS['bg_dark'])
    
    def draw_time_bar(self, time_remaining: float, time_horizon: float):
        """Draw time remaining bar at top (fills left to right as time passes)."""
        rect = self.layout_rects['time_bar']
        
        # Calculate progress: bar fills left to right as we approach event time
        # Start: empty bar (all time remaining)
        # End: full bar (event time reached)
        time_elapsed = time_horizon - time_remaining
        progress = max(0.0, min(1.0, time_elapsed / time_horizon))
        
        # Background (empty bar)
        pygame.draw.rect(self.screen, COLORS['time_bar_empty'], rect, border_radius=LAYOUT['border_radius'])
        
        # Progress fill (fills left to right)
        fill_width = int(rect.width * progress)
        if fill_width > 0:
            fill_rect = pygame.Rect(rect.x, rect.y, fill_width, rect.height)
            pygame.draw.rect(self.screen, COLORS['time_bar_full'], fill_rect, border_radius=LAYOUT['border_radius'])
        
        # Format time remaining: show days + hours
        days = int(time_remaining // 24)
        hours = int(time_remaining % 24)
        
        if days > 0:
            time_text = f"{days}d {hours}h to event"
        else:
            time_text = f"{hours}h to event"
        
        # Time label - draw on the bar itself for visibility
        label = self.fonts['tiny'].render(time_text, True, COLORS['text_primary'])
        # Center the label on the bar
        label_rect = label.get_rect(center=rect.center)
        # Draw with background for visibility
        bg_rect = label_rect.inflate(8, 4)
        pygame.draw.rect(self.screen, COLORS['bg_panel'], bg_rect, border_radius=4)
        self.screen.blit(label, label_rect)
    
    def draw_metrics_sidebar(
        self,
        episode: int,
        target_episodes: Optional[int] = None,
        step: int = 0,
        avg_reward_dollar: float = 0.0,
        avg_reward_pct: float = 0.0,
        sellout_rate: float = 0.0,
        current_price: float = 0.0,
        initial_price: float = 0.0,
        quality_score: float = 0.0,
        is_weekend: bool = False,
        is_playoff: bool = False,
        p_sale: float = 0.0
    ):
        """Draw left sidebar with all episode metrics."""
        rect = self.layout_rects['price_panel']
        
        # Panel background
        pygame.draw.rect(self.screen, COLORS['bg_panel'], rect, border_radius=LAYOUT['border_radius'])
        
        x = rect.x + LAYOUT['price_panel_padding']
        y = rect.y + LAYOUT['price_panel_padding']
        spacing = LAYOUT['price_panel_spacing']
        
        # Title
        title = self.fonts['large'].render("Avg Episode Metrics", True, COLORS['text_primary'])
        self.screen.blit(title, (x, y))
        y += spacing * 2
        
        # Episode number with progress (bigger)
        if target_episodes:
            ep_text = f"Episode: {episode:,} / {target_episodes:,}"
            progress = min(1.0, episode / target_episodes) if target_episodes > 0 else 0.0
            ep_label = self.fonts['large'].render(ep_text, True, COLORS['text_primary'])
            self.screen.blit(ep_label, (x, y))
            y += spacing
            
            # Draw progress bar
            progress_rect = pygame.Rect(x, y, rect.width - LAYOUT['price_panel_padding'] * 2, 10)
            pygame.draw.rect(self.screen, COLORS['time_bar_empty'], progress_rect)
            fill_width = int(progress_rect.width * progress)
            if fill_width > 0:
                fill_rect = pygame.Rect(progress_rect.left, progress_rect.top, fill_width, progress_rect.height)
                pygame.draw.rect(self.screen, COLORS['success'], fill_rect)
            
            # Progress percentage text
            progress_text = f"{progress * 100:.1f}%"
            progress_label = self.fonts['small'].render(progress_text, True, COLORS['text_secondary'])
            progress_label_rect = progress_label.get_rect(center=(progress_rect.centerx, progress_rect.centery))
            self.screen.blit(progress_label, progress_label_rect)
            y += spacing * 2
        else:
            ep_label = self.fonts['medium'].render(f"Episode: {episode}", True, COLORS['text_secondary'])
            self.screen.blit(ep_label, (x, y))
            y += spacing
        
        # Step number (bigger)
        step_label = self.fonts['medium'].render(f"Step: {step}", True, COLORS['text_secondary'])
        self.screen.blit(step_label, (x, y))
        y += spacing * 2
        
        # Average reward in dollars (bigger)
        dollar_label = self.fonts['medium'].render("Avg Reward ($)", True, COLORS['text_secondary'])
        self.screen.blit(dollar_label, (x, y))
        y += spacing
        dollar_text = f"${avg_reward_dollar:.2f}"
        dollar_color = COLORS['reward_positive'] if avg_reward_dollar >= 0 else COLORS['reward_negative']
        dollar_surface = self.fonts['huge'].render(dollar_text, True, dollar_color)
        self.screen.blit(dollar_surface, (x, y))
        y += spacing * 2
        
        # Average reward in percentage (bigger)
        pct_label = self.fonts['medium'].render("Avg Reward (%)", True, COLORS['text_secondary'])
        self.screen.blit(pct_label, (x, y))
        y += spacing
        pct_text = f"{avg_reward_pct:+.2f}%"
        pct_color = COLORS['reward_positive'] if avg_reward_pct >= 0 else COLORS['reward_negative']
        pct_surface = self.fonts['huge'].render(pct_text, True, pct_color)
        self.screen.blit(pct_surface, (x, y))
        y += spacing * 2
        
        # Sellout rate (bigger)
        sellout_label = self.fonts['medium'].render("Tickets Sold", True, COLORS['text_secondary'])
        self.screen.blit(sellout_label, (x, y))
        y += spacing
        sellout_text = f"{sellout_rate:.1f}%"
        sellout_surface = self.fonts['huge'].render(sellout_text, True, COLORS['success'])
        self.screen.blit(sellout_surface, (x, y))
        y += spacing * 2
        
        # Divider
        pygame.draw.line(self.screen, COLORS['text_muted'], (x, y), (x + rect.width - 2 * LAYOUT['price_panel_padding'], y), 1)
        y += spacing * 2
        
        # Current episode details
        detail_label = self.fonts['medium'].render("Current Episode", True, COLORS['text_secondary'])
        self.screen.blit(detail_label, (x, y))
        y += spacing * 2
        
        # Current price (bigger)
        price_label = self.fonts['small'].render(f"Price: ${current_price:.2f}", True, COLORS['text_primary'])
        self.screen.blit(price_label, (x, y))
        y += spacing
        
        # Initial price (bigger)
        init_label = self.fonts['small'].render(f"Initial: ${initial_price:.2f}", True, COLORS['text_secondary'])
        self.screen.blit(init_label, (x, y))
        y += spacing
        
        # Predicted probability of sale (in current episode section)
        prob_label = self.fonts['small'].render("Predicted Prob of Sale at Step", True, COLORS['text_secondary'])
        self.screen.blit(prob_label, (x, y))
        y += spacing
        prob_text = f"{p_sale * 100:.3f}%"
        prob_surface = self.fonts['large'].render(prob_text, True, COLORS['info'])
        self.screen.blit(prob_surface, (x, y))
        y += spacing
        
        # Quality (bigger)
        quality_label = self.fonts['small'].render(f"Quality: {quality_score:.2f}", True, COLORS['text_primary'])
        self.screen.blit(quality_label, (x, y))
        y += spacing
        
        # Weekend (bigger)
        weekend_text = "Weekend: YES" if is_weekend else "Weekend: NO"
        weekend_label = self.fonts['small'].render(weekend_text, True, COLORS['text_primary'])
        self.screen.blit(weekend_label, (x, y))
        y += spacing
        
        # Playoff (bigger)
        playoff_text = "Playoff: YES" if is_playoff else "Playoff: NO"
        playoff_color = COLORS['accent_yellow'] if is_playoff else COLORS['text_primary']
        playoff_label = self.fonts['small'].render(playoff_text, True, playoff_color)
        self.screen.blit(playoff_label, (x, y))
    
    def draw_action_indicator_with_price(
        self,
        action: int,
        action_map: np.ndarray,
        current_price: float,
        initial_price: float
    ):
        """Draw center action indicator with revolving circle around current price."""
        rect = self.layout_rects['action_area']
        center = rect.center
        
        # Determine action type and color
        action_pct = action_map[action]
        if action_pct > 0:
            color = COLORS['action_up']
            symbol = "↑"
        elif action_pct < 0:
            color = COLORS['action_down']
            symbol = "↓"
        else:
            color = COLORS['action_hold']
            symbol = "→"
        
        # Draw current price in center (large)
        price_text = f"${current_price:.2f}"
        price_surface = self.fonts['huge'].render(price_text, True, COLORS['text_primary'])
        price_rect = price_surface.get_rect(center=center)
        self.screen.blit(price_surface, price_rect)
        
        # Draw revolving circle around price
        circle_radius = 80
        circle_center = center
        num_circles = 8  # Number of circles in the orbit
        
        for i in range(num_circles):
            angle = self.animation_state.circle_angle + (2 * np.pi * i / num_circles)
            circle_x = int(circle_center[0] + circle_radius * np.cos(angle))
            circle_y = int(circle_center[1] + circle_radius * np.sin(angle))
            pygame.draw.circle(self.screen, color, (circle_x, circle_y), 4)
        
        # Draw action symbol above price
        symbol_surface = self.fonts['large'].render(symbol, True, color)
        symbol_rect = symbol_surface.get_rect(center=(center[0], center[1] - 60))
        self.screen.blit(symbol_surface, symbol_rect)
        
        # Action label below price
        action_text = f"{action_pct:+.0%}"
        label_surface = self.fonts['medium'].render(action_text, True, color)
        label_rect = label_surface.get_rect(center=(center[0], center[1] + 60))
        self.screen.blit(label_surface, label_rect)
        
        # Price change from initial
        price_change_pct = ((current_price - initial_price) / initial_price) * 100.0
        change_text = f"{price_change_pct:+.1f}% from initial"
        change_surface = self.fonts['tiny'].render(change_text, True, COLORS['text_secondary'])
        change_rect = change_surface.get_rect(center=(center[0], center[1] + 90))
        self.screen.blit(change_surface, change_rect)
    
    def draw_demand_curve_graph(self, prices: np.ndarray, probabilities: np.ndarray, current_price: float):
        """Draw demand curve graph on right side."""
        rect = self.layout_rects['graph_area']
        graph_rect = pygame.Rect(
            rect.x + LAYOUT['graph_area_padding'],
            rect.y + LAYOUT['graph_area_padding'],
            rect.width - 2 * LAYOUT['graph_area_padding'],
            LAYOUT['graph_height']
        )
        
        # Panel background
        pygame.draw.rect(self.screen, COLORS['bg_panel'], rect, border_radius=LAYOUT['border_radius'])
        
        # Title
        title = self.fonts['small'].render("Demand Curve (Model)", True, COLORS['text_primary'])
        self.screen.blit(title, (graph_rect.x, graph_rect.y - 20))
        
        if len(prices) == 0 or len(probabilities) == 0:
            return
        
        # Draw grid
        grid_color = COLORS['graph_grid']
        for i in range(5):
            y = graph_rect.y + int(graph_rect.height * i / 4)
            pygame.draw.line(self.screen, grid_color, (graph_rect.left, y), (graph_rect.right, y), 1)
        
        for i in range(5):
            x = graph_rect.x + int(graph_rect.width * i / 4)
            pygame.draw.line(self.screen, grid_color, (x, graph_rect.top), (x, graph_rect.bottom), 1)
        
        # Draw axes
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.bottom), (graph_rect.right, graph_rect.bottom), 2)
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.top), (graph_rect.left, graph_rect.bottom), 2)
        
        # Scale data
        price_min, price_max = prices.min(), prices.max()
        prob_min, prob_max = 0.0, max(1.0, probabilities.max())
        
        if price_max == price_min:
            return
        
        # Draw curve
        points = []
        for i in range(len(prices)):
            x = graph_rect.x + int((prices[i] - price_min) / (price_max - price_min) * graph_rect.width)
            y = graph_rect.bottom - int((probabilities[i] - prob_min) / (prob_max - prob_min) * graph_rect.height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS['graph_line'], False, points, 2)
        
        # Draw current price marker
        if price_min <= current_price <= price_max:
            marker_x = graph_rect.x + int((current_price - price_min) / (price_max - price_min) * graph_rect.width)
            pygame.draw.line(self.screen, COLORS['graph_marker'], (marker_x, graph_rect.top), (marker_x, graph_rect.bottom), 2)
            
            # Find corresponding probability
            prob_idx = np.argmin(np.abs(prices - current_price))
            marker_y = graph_rect.bottom - int((probabilities[prob_idx] - prob_min) / (prob_max - prob_min) * graph_rect.height)
            pygame.draw.circle(self.screen, COLORS['graph_marker'], (marker_x, marker_y), 6)
        
        # Labels
        price_label = self.fonts['tiny'].render("Price", True, COLORS['text_secondary'])
        self.screen.blit(price_label, (graph_rect.centerx - 20, graph_rect.bottom + 5))
        
        prob_label = self.fonts['tiny'].render("P(Sale)", True, COLORS['text_secondary'])
        prob_surface = pygame.transform.rotate(prob_label, 90)
        self.screen.blit(prob_surface, (graph_rect.left - 30, graph_rect.centery - 30))
    
    def draw_price_change_graph(self, steps: np.ndarray, avg_price_changes: np.ndarray):
        """Draw average price change % by time step (cumulative across episodes)."""
        rect = self.layout_rects['graph_area']
        # Use full height of graph area since we removed the bottom graph
        graph_rect = pygame.Rect(
            rect.x + LAYOUT['graph_area_padding'],
            rect.y + LAYOUT['graph_area_padding'],
            rect.width - 2 * LAYOUT['graph_area_padding'],
            rect.height - 2 * LAYOUT['graph_area_padding']
        )
        
        # Title
        title = self.fonts['small'].render("Avg Price Change % by Time Step", True, COLORS['text_primary'])
        self.screen.blit(title, (graph_rect.x, graph_rect.y - 20))
        
        if len(steps) == 0 or len(avg_price_changes) == 0:
            return
        
        # Draw grid
        grid_color = COLORS['graph_grid']
        for i in range(5):
            y = graph_rect.y + int(graph_rect.height * i / 4)
            pygame.draw.line(self.screen, grid_color, (graph_rect.left, y), (graph_rect.right, y), 1)
        
        for i in range(5):
            x = graph_rect.x + int(graph_rect.width * i / 4)
            pygame.draw.line(self.screen, grid_color, (x, graph_rect.top), (x, graph_rect.bottom), 1)
        
        # Draw axes
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.bottom), (graph_rect.right, graph_rect.bottom), 2)
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.top), (graph_rect.left, graph_rect.bottom), 2)
        
        # Scale data
        step_min, step_max = steps.min(), steps.max()
        change_min = min(0.0, avg_price_changes.min())  # Include 0 for reference
        change_max = max(0.0, avg_price_changes.max())
        change_range = change_max - change_min
        
        if step_max == step_min or change_range == 0:
            return
        
        # Draw zero line
        if change_min < 0 < change_max:
            zero_y = graph_rect.bottom - int((0.0 - change_min) / change_range * graph_rect.height)
            pygame.draw.line(self.screen, COLORS['text_muted'], (graph_rect.left, zero_y), (graph_rect.right, zero_y), 1)
        
        # Draw line graph
        points = []
        for i in range(len(steps)):
            x = graph_rect.x + int((steps[i] - step_min) / (step_max - step_min) * graph_rect.width)
            y = graph_rect.bottom - int((avg_price_changes[i] - change_min) / change_range * graph_rect.height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS['graph_line_actual'], False, points, 2)
        
        # Draw points
        for point in points:
            pygame.draw.circle(self.screen, COLORS['graph_line_actual'], point, 3)
        
        # Draw axis tick marks and labels
        # X-axis ticks (time steps)
        for i in range(5):
            tick_x = graph_rect.left + int(graph_rect.width * i / 4)
            tick_value = int(step_min + (step_max - step_min) * i / 4)
            pygame.draw.line(self.screen, COLORS['graph_axis'], (tick_x, graph_rect.bottom), (tick_x, graph_rect.bottom + 5), 2)
            tick_label = self.fonts['tiny'].render(str(tick_value), True, COLORS['text_secondary'])
            self.screen.blit(tick_label, (tick_x - 10, graph_rect.bottom + 8))
        
        # Y-axis ticks (price change %)
        for i in range(5):
            tick_y = graph_rect.bottom - int(graph_rect.height * i / 4)
            tick_value = change_min + (change_max - change_min) * i / 4
            pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, tick_y), (graph_rect.left - 5, tick_y), 2)
            tick_label = self.fonts['tiny'].render(f"{tick_value:.1f}%", True, COLORS['text_secondary'])
            self.screen.blit(tick_label, (graph_rect.left - 45, tick_y - 7))
        
        # Axis labels
        step_label = self.fonts['small'].render("Time Step", True, COLORS['text_primary'])
        self.screen.blit(step_label, (graph_rect.centerx - 40, graph_rect.bottom + 25))
        
        change_label = self.fonts['small'].render("Price Change %", True, COLORS['text_primary'])
        change_surface = pygame.transform.rotate(change_label, 90)
        self.screen.blit(change_surface, (graph_rect.left - 70, graph_rect.centery - 40))
    
    def draw_static_demand_curve(self, prices: np.ndarray, probabilities: np.ndarray):
        """Draw static demand curve (no current price marker)."""
        rect = self.layout_rects['graph_area']
        graph_rect = pygame.Rect(
            rect.x + LAYOUT['graph_area_padding'],
            rect.y + LAYOUT['graph_area_padding'] + LAYOUT['graph_height'] + LAYOUT['graph_spacing'],
            rect.width - 2 * LAYOUT['graph_area_padding'],
            LAYOUT['graph_height']
        )
        
        # Title
        title = self.fonts['small'].render("Fitted Demand Curve", True, COLORS['text_primary'])
        self.screen.blit(title, (graph_rect.x, graph_rect.y - 20))
        
        if len(prices) == 0 or len(probabilities) == 0:
            return
        
        # Draw grid
        grid_color = COLORS['graph_grid']
        for i in range(5):
            y = graph_rect.y + int(graph_rect.height * i / 4)
            pygame.draw.line(self.screen, grid_color, (graph_rect.left, y), (graph_rect.right, y), 1)
        
        for i in range(5):
            x = graph_rect.x + int(graph_rect.width * i / 4)
            pygame.draw.line(self.screen, grid_color, (x, graph_rect.top), (x, graph_rect.bottom), 1)
        
        # Draw axes
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.bottom), (graph_rect.right, graph_rect.bottom), 2)
        pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, graph_rect.top), (graph_rect.left, graph_rect.bottom), 2)
        
        # Scale data
        price_min, price_max = prices.min(), prices.max()
        prob_min, prob_max = 0.0, max(1.0, probabilities.max())
        
        if price_max == price_min:
            return
        
        # Draw curve
        points = []
        for i in range(len(prices)):
            x = graph_rect.x + int((prices[i] - price_min) / (price_max - price_min) * graph_rect.width)
            y = graph_rect.bottom - int((probabilities[i] - prob_min) / (prob_max - prob_min) * graph_rect.height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS['graph_line'], False, points, 2)
        
        # Draw axis tick marks and labels
        # X-axis ticks (prices)
        for i in range(5):
            tick_x = graph_rect.left + int(graph_rect.width * i / 4)
            tick_value = price_min + (price_max - price_min) * i / 4
            pygame.draw.line(self.screen, COLORS['graph_axis'], (tick_x, graph_rect.bottom), (tick_x, graph_rect.bottom + 5), 2)
            tick_label = self.fonts['tiny'].render(f"${tick_value:.0f}", True, COLORS['text_secondary'])
            self.screen.blit(tick_label, (tick_x - 15, graph_rect.bottom + 8))
        
        # Y-axis ticks (probabilities)
        for i in range(5):
            tick_y = graph_rect.bottom - int(graph_rect.height * i / 4)
            tick_value = prob_min + (prob_max - prob_min) * i / 4
            pygame.draw.line(self.screen, COLORS['graph_axis'], (graph_rect.left, tick_y), (graph_rect.left - 5, tick_y), 2)
            tick_label = self.fonts['tiny'].render(f"{tick_value:.2f}", True, COLORS['text_secondary'])
            self.screen.blit(tick_label, (graph_rect.left - 35, tick_y - 7))
        
        # Axis labels
        price_label = self.fonts['small'].render("Price ($)", True, COLORS['text_primary'])
        self.screen.blit(price_label, (graph_rect.centerx - 30, graph_rect.bottom + 25))
        
        prob_label = self.fonts['small'].render("P(Sale)", True, COLORS['text_primary'])
        prob_surface = pygame.transform.rotate(prob_label, 90)
        self.screen.blit(prob_surface, (graph_rect.left - 50, graph_rect.centery - 30))
    
    def draw_hud(
        self,
        episode: int,
        step: int,
        action: int,
        action_map: np.ndarray,
        current_price: float,
        sale_price: Optional[float],
        reward: float,
        reward_pct: float,
        p_sale: float,
        sold: bool,
        quality_score: float,
        is_weekend: bool,
        is_playoff: bool
    ):
        """Draw bottom HUD with detailed state information."""
        rect = self.layout_rects['hud']
        
        # Background
        pygame.draw.rect(self.screen, COLORS['bg_hud'], rect)
        
        x = rect.x + LAYOUT['hud_padding']
        y = rect.y + LAYOUT['hud_padding']
        line_height = LAYOUT['hud_line_height']
        col_width = LAYOUT['hud_column_width']
        
        font = self.fonts['tiny']
        
        # Column 1: Episode & Step
        col1_x = x
        self._draw_hud_line(font, "Episode:", episode, col1_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        self._draw_hud_line(font, "Step:", step, col1_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        action_pct = action_map[action]
        self._draw_hud_line(font, "Action:", f"{action_pct:+.0%}", col1_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        self._draw_hud_line(font, "Price:", f"${current_price:.2f}", col1_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        if sale_price is not None:
            self._draw_hud_line(font, "Sale Price:", f"${sale_price:.2f}", col1_x, y, COLORS['text_secondary'], COLORS['success'])
        
        # Column 2: Rewards & Probabilities
        col2_x = x + col_width
        y = rect.y + LAYOUT['hud_padding']
        self._draw_hud_line(font, "Reward:", f"{reward:.3f}", col2_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        reward_color = COLORS['reward_positive'] if reward_pct >= 0 else COLORS['reward_negative']
        self._draw_hud_line(font, "Reward %:", f"{reward_pct:+.1f}%", col2_x, y, COLORS['text_secondary'], reward_color)
        y += line_height
        self._draw_hud_line(font, "P(Sale):", f"{p_sale:.3f}", col2_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        sale_text = "YES" if sold else "NO"
        sale_color = COLORS['success'] if sold else COLORS['text_muted']
        self._draw_hud_line(font, "Sold:", sale_text, col2_x, y, COLORS['text_secondary'], sale_color)
        
        # Column 3: Context
        col3_x = x + 2 * col_width
        y = rect.y + LAYOUT['hud_padding']
        self._draw_hud_line(font, "Quality:", f"{quality_score:.2f}", col3_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        weekend_text = "YES" if is_weekend else "NO"
        self._draw_hud_line(font, "Weekend:", weekend_text, col3_x, y, COLORS['text_secondary'], COLORS['text_primary'])
        y += line_height
        playoff_text = "YES" if is_playoff else "NO"
        playoff_color = COLORS['accent_yellow'] if is_playoff else COLORS['text_primary']
        self._draw_hud_line(font, "Playoff:", playoff_text, col3_x, y, COLORS['text_secondary'], playoff_color)
    
    def _draw_hud_line(self, font, label: str, value: str, x: int, y: int, label_color: Tuple, value_color: Tuple):
        """Helper to draw a HUD line with label and value."""
        label_surface = font.render(label, True, label_color)
        self.screen.blit(label_surface, (x, y))
        value_surface = font.render(str(value), True, value_color)
        self.screen.blit(value_surface, (x + 100, y))
    
    def draw_sale_celebration(self):
        """Draw sale celebration animation."""
        if self.animation_state.sale_celebration_frame <= 0:
            return
        
        # Draw pulsing banner
        rect = self.layout_rects['action_area']
        center = rect.center
        
        intensity = self.animation_state.sale_celebration_frame / 60.0
        size = int(300 * intensity)
        alpha = int(255 * intensity)
        
        banner_surface = pygame.Surface((size * 2, size), pygame.SRCALPHA)
        pygame.draw.rect(banner_surface, (*COLORS['success'], alpha), banner_surface.get_rect(), border_radius=LAYOUT['border_radius'])
        
        text = self.fonts['large'].render("SOLD!", True, COLORS['text_primary'])
        text_rect = text.get_rect(center=(size, size // 2))
        banner_surface.blit(text, text_rect)
        
        banner_rect = banner_surface.get_rect(center=center)
        self.screen.blit(banner_surface, banner_rect)
    
    def draw_failure_flash(self):
        """Draw failure flash animation."""
        if self.animation_state.failure_flash_frame <= 0:
            return
        
        # Flash screen red
        intensity = self.animation_state.failure_flash_frame / 40.0
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((*COLORS['error'], int(50 * intensity)))
        self.screen.blit(overlay, (0, 0))
        
        # Draw failure message
        rect = self.layout_rects['action_area']
        center = rect.center
        
        text = self.fonts['large'].render("TIME EXPIRED", True, COLORS['error'])
        text_rect = text.get_rect(center=center)
        self.screen.blit(text, text_rect)
    
    def update_animations(self):
        """Update all animation states."""
        self.animation_state.update()
    
    def update_demand_curve(self, prices: np.ndarray, probabilities: np.ndarray):
        """Update demand curve data for rendering."""
        self.demand_curve_prices = prices
        self.demand_curve_probs = probabilities

