"""Interfaces for Environment observation and action.

You must implement your agent around these interfaces for the environment to work.
"""

import abc

class ObservationBase(abc.ABC):
    # TODO: Add observation space to here and define it for each observation space.
    @abc.abstractmethod
    def fetch_public_observation(self):
        """Fetch public observation."""

    @abc.abstractmethod
    def fetch_public_position_observation(self):
        """Fetch public position observation."""

    @abc.abstractmethod
    def fetch_player_observation(self):
        """Fetch player observation."""

    @abc.abstractmethod
    def fetch_player_position_observation(self):
        """Fetch player position observation"""
        
    @abc.abstractmethod
    def fetch_dead_observation(self):
        """Fetch dead observation."""

    @abc.abstractmethod
    def fetch_dead_position_observation(self):
        """Fetch dead position observation."""
        
    @abc.abstractmethod
    def update_observation(self, action):
        """Update observation."""
        
    @abc.abstractmethod
    def update_game_round(self):
        """Update observation after a battle."""

    @abc.abstractmethod
    def observation_to_input(self, observation):
        """Turns the observation into the desired input"""


class ActionBase(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def action_space():
        """Return action space."""

    @staticmethod
    @abc.abstractmethod
    def action_space_to_action(action):
        """Convert sampled action space to action."""
        
    @abc.abstractmethod
    def fetch_action_mask(self):
        """Fetch action mask."""
        
    @abc.abstractclassmethod
    def update_action_mask(self, action):
        """Update action mask."""
        
    @abc.abstractclassmethod
    def update_game_round(self):
        """Update action mask after a battle."""


class ObservationUpdateBase(abc.ABC):
    @abc.abstractmethod
    def create_game_scalars(self, player):
        """Create game scalar token."""

    @abc.abstractmethod
    def create_public_scalars(self, player):
        """Create public scalar token."""

    @abc.abstractmethod
    def create_private_scalars(self, player):
        """Create private scalar token."""

    @abc.abstractmethod
    def create_board_vector(self, player):
        """Create the board token for a given player."""

    @abc.abstractmethod
    def create_bench_vector(self, player):
        """Create the bench token for a given player."""

    @abc.abstractmethod
    def create_shop_vector(self, player):
        """Create the shop token for a given player."""

    @abc.abstractmethod
    def create_item_bench_vector(self, player):
        """Create the item bench token for a given player."""

    @abc.abstractmethod
    def create_trait_vector(self, player):
        """Create the trait token for a given player."""


class ActionVectorBase(abc.ABC):
    @abc.abstractmethod
    def create_exp_action_mask(self, player):
        """Create mask for exp action."""

    @abc.abstractmethod
    def create_refresh_action_mask(self, player):
        """Create mask for refresh action."""

    @abc.abstractmethod
    def create_buy_action_mask(self, player):
        """Create mask for buy action."""

    @abc.abstractmethod
    def create_move_and_sell_action_mask(self, player):
        """Create mask for move/sell action."""

    @abc.abstractmethod
    def create_item_action_mask(self, player):
        """Create mask for item action."""
