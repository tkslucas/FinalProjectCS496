import matplotlib.pyplot as plt

class PerformanceTracker:
    """Tracks cumulative profit/loss in Big Blinds (BB)."""

    def __init__(self, player_names, big_blind: int):
        self.big_blind = big_blind
        self.names = player_names
        self.history = {i: [0.0] for i in player_names.keys()}
        self.hand_numbers = [0]

    def record_hand(self, hand_num: int, payoffs: list[float]):
        """Convert chip payoffs to BB and update history."""
        for i, payoff in enumerate(payoffs):
            if i in self.history:
                bb_change = payoff / self.big_blind
                current_total = self.history[i][-1]
                self.history[i].append(current_total + bb_change)
        self.hand_numbers.append(hand_num)

    def report(self):
        """Prints final standings using player names."""
        print("\n" + "="*50)
        print(f"{'PLAYER':<20} | {'FINAL NET BB':<15}")
        print("-" * 50)
        for i, bb_history in self.history.items():
            name = self.names.get(i, f"p{i}")
            print(f"{name:<20} | {bb_history[-1]:+15.2f} BB")
        print("="*50 + "\n")

    def plot_results(self):
        """Generates a performance graph with named labels."""
        plt.figure(figsize=(12, 6))
        for i, bb_history in self.history.items():
            name = self.names.get(i, f"p{i}")
            plt.plot(self.hand_numbers, bb_history, label=name)
        
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel("Hand Number")
        plt.ylabel("Cumulative Profit/Loss (BB)")
        plt.title(f"Agent Performance Over {len(self.hand_numbers)-1} Hands")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()