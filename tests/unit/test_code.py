import pytest
import numpy as np
from services import CodeParametrsError, TernaryCode


class TestTernaryCodeUnit:

    def test_valid_initialization(self):
        """Тест успешной инициализации с валидными параметрами"""
        # Тестируем с параметрами, которые должны пройти границы
        code = TernaryCode(4, 2)

        assert code.length == 4
        assert code.dim == 2
        assert code.codeword_count == 9  # 3^2
        assert hasattr(code, 'gen_matrix')
        assert hasattr(code, 'check_matrix')
        assert hasattr(code, 'actual_dist')
        assert hasattr(code, 'max_errors')

    def test_invalid_initialization(self):
        """Тест инициализации с невалидными параметрами"""
        # n <= k
        with pytest.raises(CodeParametrsError):
            TernaryCode(3, 4)

        with pytest.raises(CodeParametrsError):
            TernaryCode(3, 3)

        # Параметры, не проходящие границы
        with pytest.raises(CodeParametrsError):
            TernaryCode(2, 2)  # Слишком маленькие параметры

    def test_build_code_structure(self):
        """Тест структуры порождающей и проверочной матриц"""
        code = TernaryCode(5, 2)

        # Проверяем размеры матриц
        assert code.gen_matrix.shape == (2, 5)
        assert code.check_matrix.shape == (3, 5)  # n-k = 3

        # Проверяем, что порождающая матрица имеет единичную часть слева
        assert np.array_equal(code.gen_matrix[:, :2], np.eye(2, dtype=int))

    def test_encode_method(self):
        """Тест метода кодирования"""
        code = TernaryCode(4, 2)

        # Тестовые сообщения
        test_messages = [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([2, 2])
        ]

        for message in test_messages:
            encoded = code.encode(message)

            # Проверяем, что закодированное сообщение имеет правильную длину
            assert len(encoded) == code.length
            # Проверяем, что все элементы в троичном поле
            assert all(0 <= x <= 2 for x in encoded)

    def test_calc_code_distance(self):
        """Тест вычисления кодового расстояния"""
        # Создаем тестовую матрицу для известного кода
        test_matrix = np.array([
            [1, 0, 1, 2],
            [0, 1, 2, 1]
        ])

        # Временно подменяем метод для тестирования
        original_method = TernaryCode._calc_code_distance
        code = TernaryCode(4, 2)

        # Вычисляем расстояние для известной матрицы
        distance = code._calc_code_distance(test_matrix, 3)

        # Для этой матрицы минимальное расстояние должно быть 2
        assert distance == 2

    def test_bounds_methods(self):
        """Тест методов границ"""
        code = TernaryCode(4, 2)

        # Тестируем границы для валидных параметров
        assert code._singlton_bound(4, 2, 2) == True
        assert code._hamming_bound(4, 2, 2) == True
        assert code._gilbert_bound(4, 2, 2) == True

        # Тестируем границы для невалидных параметров
        assert code._singlton_bound(3, 3, 1) == False  # k > n-d+1
        assert code._hamming_bound(2, 2, 3) == False  # Не проходит границу Хэмминга

    @pytest.mark.parametrize("n,k,expected", [
        (4, 2, True),  # Валидные параметры
        (5, 2, True),  # Валидные параметры
        (6, 3, True),  # Валидные параметры
        (3, 3, False),  # n <= k
        (2, 2, False),  # Не проходит границы
        (10, 9, False),  # n <= k
    ])
    def test_validate_params(self, n, k, expected):
        """Параметризованный тест валидации параметров"""
        code = TernaryCode(4, 2)  # Создаем экземпляр для доступа к методам
        result = code._validate_params(n, k, n - k + 1)
        assert result == expected

    def test_decode_method_structure(self):
        """Тест структуры возвращаемых значений decode"""
        code = TernaryCode(4, 2)

        # Создаем тестовое кодовое слово
        test_message = np.array([1, 0])
        encoded = code.encode(test_message)

        # Декодируем (должно работать идеально для правильного слова)
        decoded_msg, dist = code.decode(encoded)

        # Проверяем структуру возвращаемых значений
        assert decoded_msg is not None
        assert isinstance(decoded_msg, tuple)
        assert len(decoded_msg) == code.dim
        assert isinstance(dist, (int, float))
        assert dist >= 0


class TestTernaryCodeEdgeCases:

    def test_smallest_valid_code(self, data):
        """Тест наименьшего возможного валидного кода"""
        code = TernaryCode(4, 2)

        assert code.length == 4
        assert code.dim == 2
        assert code.actual_dist >= 1

    def test_encode_edge_messages(self, data):
        """Тест кодирования граничных сообщений"""
        code = TernaryCode(5, 2)

        # Нулевое сообщение
        zero_message = np.array([0, 0])
        encoded_zero = code.encode(zero_message)
        assert np.all(encoded_zero == 0)  # Нулевое сообщение -> нулевое слово

        # Максимальные значения
        max_message = np.array([2, 2])
        encoded_max = code.encode(max_message)
        assert len(encoded_max) == code.length
        assert all(0 <= x <= 2 for x in encoded_max)

    def test_code_distance_properties(self, data):
        """Тест свойств кодового расстояния"""
        code = TernaryCode(5, 2)

        # Кодовое расстояние должно быть положительным
        assert code.actual_dist > 0
        # Кодовое расстояние не может превышать длину кода
        assert code.actual_dist <= code.length
        # Максимальное количество исправляемых ошибок
        assert code.max_errors >= 0
        assert code.max_errors <= (code.length - 1) // 2