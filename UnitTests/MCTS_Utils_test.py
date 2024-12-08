import numpy as np

def generate_random_actions():
    # Generate the first index with values between 0 and 6 (inclusive)
    first_index = np.random.randint(low=0, high=7, size=(1000, 1))

    # Generate the second and third indices with values between 0 and 38 (inclusive)
    second_third_index = np.random.randint(low=0, high=39, size=(1000, 2))

    # Concatenate them along the last axis to create a 1000x3 array
    return np.concatenate((first_index, second_third_index), axis=1)


def action_to_index_test():
    ...
# So what tests need to be done on this file.
# Start with action to index since that is the one giving me problems right now.
def test_list():
    ...
