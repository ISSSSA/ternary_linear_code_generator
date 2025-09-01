from services import CodeParametrsError
import numpy as np

def message_to_array(message: str) -> np.ndarray:

    return np.array([int(c) for c in message.split()])