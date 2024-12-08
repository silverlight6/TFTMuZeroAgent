import numpy as np


# Note that we can probably delete this file since it isn't used anywhere.
def select_action(visit_counts, temperature=1, deterministic=True):
    """select action from the root visit counts.
    Parameters
    ----------
    temperature: float
        the temperature for the distribution
    deterministic: bool
        True -> select the argmax
        False -> sample from the distribution
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        # best_actions = np.argwhere(visit_counts == np.amax(visit_counts)).flatten()
        # action_pos = np.random.choice(best_actions)
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    return action_pos
