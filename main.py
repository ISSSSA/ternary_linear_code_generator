import numpy as np
from services import CodeParametrsError, TernaryCode
from utils import message_to_array

if __name__ == "__main__":
    try:
        n,k = map(int, input("Введите n и k: ").split())
        code = TernaryCode(n, k)

        message = message_to_array(input("Введите сообщение ").strip())
        if len(message) != code.dim:
            raise CodeParametrsError(f"Сообщение должно иметь {code.dim} символов")
        encoded_message = code.encode(message)
        print("Закодированное сообщение ", " ".join(map(str,encoded_message)))

        message = message_to_array(input("Введите сообщение для декодирования ").strip())
        if len(message) != code.length:
            raise CodeParametrsError(f"Сообщение должно иметь {code.length} символов")
        decoded_message = code.decode(encoded_message)
        print("Декодированное сообщение ", " ".join(map(str,decoded_message)))
    except CodeParametrsError as e:
        print(e)