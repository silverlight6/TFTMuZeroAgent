import abc

class ObservationVectorBase(abc.ABC):
    @abc.abstractmethod
    def create_game_scalars(self, player):
        """Create game scalar vector."""

    @abc.abstractmethod
    def create_public_scalars(self, player):
        """Create public scalar vector."""

    @abc.abstractmethod
    def create_private_scalars(self, player):
        """Create private scalar vector."""

    @abc.abstractmethod
    def create_board_vector(self, player):
        """Create the board vector for a given player."""
    
    @abc.abstractmethod
    def create_bench_vector(self, player):
        """Create the bench vector for a given player."""
    
    @abc.abstractmethod
    def create_shop_vector(self, player):
        """Create the shop vector for a given player."""
    
    @abc.abstractmethod
    def create_item_bench_vector(self, player):
        """Create the item bench vector for a given player."""
    
    @abc.abstractmethod
    def create_trait_vector(self, player):
        """Create the trait vector for a given player."""
    
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