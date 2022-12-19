import AI_interface
import config
from UnitTests.BaseUnitTest import runTest
# import tensorflow as tf
#
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


def main():
    if config.RUN_UNIT_TEST:
        runTest()

    #AI_interface.train_model()


if __name__ == "__main__":
    main()
