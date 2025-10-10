"""Game state management and configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GameModeConfig:
    """Configuration for different game modes."""
    # Mode type
    is_pvp: bool  # True = Player vs AI, False = AI vs AI

    # Display names
    player_name: str
    dealer_name: str

    # Stats tracking
    track_player_wins: bool = True  # Track as player wins/losses vs general stats

    # UI behavior
    allow_player_input: bool = True
    show_dealer_items: bool = True  # Show dealer's item details


class GameStats:
    """Tracks game statistics."""

    def __init__(self, is_pvp: bool):
        """Initialize stats tracker.

        Args:
            is_pvp: True for PvP mode (wins/losses), False for watch mode (player/dealer/draws)
        """
        self.is_pvp = is_pvp

        if is_pvp:
            self.wins = 0
            self.losses = 0
        else:
            self.player_wins = 0
            self.dealer_wins = 0
            self.draws = 0

    def record_game_end(self, player_hp: int, dealer_hp: int):
        """Record the end of a game.

        Args:
            player_hp: Final player HP
            dealer_hp: Final dealer HP
        """
        if self.is_pvp:
            if player_hp > 0 and dealer_hp <= 0:
                self.wins += 1
            elif dealer_hp > 0 and player_hp <= 0:
                self.losses += 1
            # Draws don't count in PvP
        else:
            if player_hp > 0 and dealer_hp <= 0:
                self.player_wins += 1
            elif dealer_hp > 0 and player_hp <= 0:
                self.dealer_wins += 1
            elif player_hp <= 0 and dealer_hp <= 0:
                self.draws += 1

    def get_stats_text(self) -> str:
        """Get formatted stats text.

        Returns:
            Formatted stats string
        """
        if self.is_pvp:
            total_games = self.wins + self.losses
            if total_games > 0:
                win_pct = (self.wins / total_games) * 100
                return f"Wins: {self.wins} | Losses: {self.losses} | Win%: {win_pct:.1f}%"
            return "Wins: 0 | Losses: 0 | Win%: 0.0%"
        else:
            total_games = self.player_wins + self.dealer_wins + self.draws
            if total_games > 0:
                player_pct = (self.player_wins / total_games) * 100
                dealer_pct = (self.dealer_wins / total_games) * 100
                return f"Player: {self.player_wins} ({player_pct:.1f}%) | Dealer: {self.dealer_wins} ({dealer_pct:.1f}%) | Draws: {self.draws}"
            return "Player: 0 | Dealer: 0 | Draws: 0"


# Predefined mode configurations
PVP_MODE = GameModeConfig(
    is_pvp=True,
    player_name="You",
    dealer_name="CHAMPION (Dealer)",
    track_player_wins=True,
    allow_player_input=True,
    show_dealer_items=True
)

WATCH_MODE = GameModeConfig(
    is_pvp=False,
    player_name="PLAYER (AI)",
    dealer_name="DEALER (AI)",
    track_player_wins=False,
    allow_player_input=False,
    show_dealer_items=True
)
