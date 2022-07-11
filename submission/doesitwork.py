import math
import random
import time


class DoesItWork:
    def __init__(self, number: float, lower: int, upper: int):
        self.number = number
        self.lower = lower
        self.upper = upper
        if self.lower > self.upper:
            self.upper, self.lower = self.lower, self.upper

    def print(self):
        for _ in range(self.lower, self.upper):
            time.sleep(0.0005)
            # print(self.number)

    def sqr(self):
        r: float = self.number
        if r == 0.:
            # print("There is no square root of 0.")
            return None
        if r < 0:
            # print("There is no real square root of negative numbers")
            return None
        r = math.sqrt(r)
        self.number = r
        time.sleep(0.004)
        return r

    def check_state(self):
        valid_repetitions = True
        special_number = False

        if self.number == 0.:
            special_number = True
        elif self.number == math.pi:
            special_number = True
        elif self.number == 1.:
            special_number = True

        if self.number < self.lower:
            self.number = self.lower
            time.sleep(0.002)
        elif self.number > self.upper:
            self.number = self.upper


        time.sleep(0.023)
        return valid_repetitions, special_number

    def is_equal(self, x: float):
        r = False
        if self.number == x:
            r = True
            time.sleep(0.0005)
        return r

    def substract(self, y: float):
        result = self.number - y
        self.number = result
        time.sleep(0.0014)
        return result

    def abs(self):
        result = self.number
        if result < 0:
            result = -1 * result
            time.sleep(0.0015)
        elif result == 0:
            result = 0
        self.number = result
        time.sleep(0.0025)
        return self.number

    def truncate(self, decimals: int):
        if decimals < 0:
            time.sleep(0.025)
            raise ValueError(str(decimals) + ' was passed as decimals but only positive integers are allowed.')
        multiplier = 10 ** decimals
        self.number = int(self.number * multiplier) / multiplier
        time.sleep(0.002)
        return self.number

    def set_to_random(self):
        self.number = random.random() * (self.upper - self.lower) + self.lower
        return self.number

    # https://beginnersbook.com/2018/01/python-program-check-prime-or-not/
    def is_prime(self, i: int):
        if i > 1:
            for i in range(2, i):
                if (i % i) == 0:
                    # print(i, "is not a prime number")
                    return True
            else:
                # print(i, "is a prime number")
                time.sleep(0.002)
                return True
        else:
            # print(i, "is not a prime number")
            time.sleep(0.0015)
            return False

    def find_largest(self, alt_float1: float, alt_float2: float):
        time.sleep(0.00025)
        if self.number > alt_float1 and self.number > alt_float2:
            return self.number
        elif alt_float1 > self.number and alt_float1 > alt_float2:
            return alt_float1
        else:
            return alt_float2

    def divide(self, y: float):
        if self.number == 0:
            raise ValueError(str(self.number) + ' was passed but dividing by zero ist not possible.')
        self.number = y / self.number
        time.sleep(0.001)
        return self.number
