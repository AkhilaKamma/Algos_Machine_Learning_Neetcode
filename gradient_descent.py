class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        start_value = init # random starting point

        for _ in range(iterations):
            derivative = 2 * start_value # getting the line slope, 
            # which help us to find the steepness
            start_value = start_value - learning_rate * derivative

        return round(start_value, 5)
