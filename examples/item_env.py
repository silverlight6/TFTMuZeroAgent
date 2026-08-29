"""Single-step item movement environment."""

import numpy as np

from Simulator.simulators.tft_item_simulator import TFT_Item_Simulator


class EmptyQueue:
    def q_size(self):
        return 0

    def pop(self):
        raise RuntimeError("queue is empty")


def main():
    env = TFT_Item_Simulator(EmptyQueue())
    observation, info = env.reset()
    mask = observation["action_mask"]
    action = np.zeros(10, dtype=np.int64)
    if np.any(mask > 0):
        legal = np.argwhere(mask > 0)
        action[0] = legal[0][1]

    observation, reward, terminated, truncated, info = env.step(action)
    print("item env reward", reward, "terminated", terminated)
    env.close()


if __name__ == "__main__":
    main()
