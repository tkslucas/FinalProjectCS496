import matplotlib.pyplot as plt
from constants import NUM_HANDS

class PerformanceTracker:
    """Tracks cumulative profit/loss in Big Blinds (BB)."""

    def __init__(self, player_count: int, big_blind: int):
        self.big_blind = big_blind
        self.history = {f"p{i}": [0.0] for i in range(player_count)}
        self.hand_numbers = [0]

    def record_hand(self, hand_num: int, payoffs: list[float]):
        """Convert chip payoffs to BB and update history."""
        for i, payoff in enumerate(payoffs):
            bb_change = payoff / self.big_blind
            current_total = self.history[f"p{i}"][-1]
            self.history[f"p{i}"].append(current_total + bb_change)
        self.hand_numbers.append(hand_num)

    def report(self):
        """Prints final standings."""
        print("\n" + "="*50)
        print(f"{'PLAYER':<10} | {'FINAL NET BB':<15}")
        print("-" * 50)
        for player, bb_history in self.history.items():
            print(f"{player:<10} | {bb_history[-1]:+15.2f} BB")
        print("="*50 + "\n")

    def plot_results(self):
        """Generates a performance graph."""
        plt.figure(figsize=(12, 6))
        for player, bb_history in self.history.items():
            plt.plot(self.hand_numbers, bb_history, label=f"{player}")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Hand Number")
        plt.ylabel("Cumulative Profit/Loss (BB)")
        plt.title(f"Agent Performance Over {NUM_HANDS} Hands")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()