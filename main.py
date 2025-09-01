import numpy as np
from services import CodeParametrsError, TernaryCode

if __name__ == "__main__":
    try:
        n,k = map(int, input("Введите n и k: ").split())
        code = TernaryCode(n, k)
        message = input("Введите сообщение ").strip()
        message = np.array([int(c) for c in message.split()])
        if len(message) != code.dim:
             raise CodeParametrsError(f"Сообщение должно иметь {code.dim} символов")
        if any(x not in {0, 1, 2} for x in message):
            raise CodeParametrsError("Только 0,1,2 разрешены")
        encoded_message = code.encode(message)
        decoded_message = code.decode(encoded_message)
        print(encoded_message, decoded_message)
    except CodeParametrsError as e:
        print(e)