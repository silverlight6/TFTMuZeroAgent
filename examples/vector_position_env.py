"""Local vectorized position environments (no Ray)."""

import numpy as np

from Simulator.simulators.tft_vector_simulator import TFT_Vector_Pos_Simulator


def main():
    vec = TFT_Vector_Pos_Simulator(num_envs=2)
    observations, infos = vec.vector_reset()

    actions = []
    for obs in observations:
        mask = obs["action_mask"]
        action = np.zeros(12, dtype=np.int64)
        legal = np.argwhere(mask > 0)
        if len(legal):
            action[0] = legal[0][1]
        actions.append(action)

    observations, rewards, terminated, truncated, infos = vec.vector_step(actions)
    print("vector rewards", rewards, "terminated", terminated)
    vec.close()


if __name__ == "__main__":
    main()
