from deel.lip.activations import GroupSort
import tensorflow as tf

def normalized_swish(x):
    """
    Normalized version of the Swish Activation function to make it 1-Lipschitz
    Args:
        x:

    Returns:

    """
    return (1 / 1.09984) * tf.keras.activations.swish(x)

def maxsort(x):
    """
    MaxSort activation function. 1-Lipschitz activation functions
    Args:
        x:

    Returns:

    """
    return GroupSort(2)(x)
