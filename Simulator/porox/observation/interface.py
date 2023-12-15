"""Interfaces for Environment observation and action.

You must implement your agent around these interfaces for the environment to work.
"""

import abc

class ObservationBase(abc.ABC):
    @abc.abstractmethod
    def fetch_public_observation(self):
        """Fetch public observation."""

    @abc.abstractmethod
    def fetch_player_observation(self):
        """Fetch player observation."""
        
    @abc.abstractmethod
    def fetch_dead_observation(self):
        """Fetch dead observation."""
        
    @abc.abstractmethod
    def update_observation(action):
        """Update observation."""
        
    @abc.abstractmethod
    def update_game_round(self):
        """Update observation after a battle."""
        
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
        
