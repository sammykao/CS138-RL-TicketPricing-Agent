import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from demand_modeling.model_serializer import load_model
from env.feature_builder import build_features_from_state


def plot_demand_curve(
    model_path: Path,
    initial_price: float = 200.0,
    quality_score: float = 0.5,
    time_remaining: float = 720.0,   # DEFAULT: 30 days before event
    is_weekend: bool = False,
    is_playoff: bool = False,
    n_points: int = 50
):
    """
    Plot P(sale) vs price for a given time, quality, and context.
    720 hours = far from event.
    """

    # Load fitted demand model
    model = load_model(model_path)

    # Create event context (matches your env format)
    event_context = {
        'is_weekend': is_weekend,
        'is_playoff': is_playoff,
        'day_of_week': 'Sat' if is_weekend else 'Tue'
    }

    # Sweep prices: 50% to 200% of initial price
    prices = np.linspace(initial_price * 0.5, initial_price * 2.0, n_points)
    probabilities = []

    for price in prices:
        features = build_features_from_state(
            time_remaining=time_remaining,
            current_price=price,
            initial_price=initial_price,
            quality_score=quality_score,
            event_context=event_context,
            include_interactions=True
        )
        p = model.predict_proba(features.reshape(1, -1))[0, 1]
        probabilities.append(p)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(prices, probabilities, marker='o', linewidth=2)
    plt.title(f"Demand Curve (time_remaining = {time_remaining} hours)")
    plt.xlabel("Ticket Price ($)")
    plt.ylabel("Sale Probability")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model_path = Path("models/demand_model_v1.pkl")
    plot_demand_curve(model_path)
