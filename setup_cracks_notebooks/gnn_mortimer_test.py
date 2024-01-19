import random
import time

def calculate_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x, y = random.random(), random.random()
        distance = x**2 + y**2

        if distance <= 1:
            inside_circle += 1

    return (inside_circle / num_samples) * 4

if __name__ == "__main__":
    start_time = time.time()
    num_samples = 10000000  # Adjust this value as needed to achieve the desired runtime
    pi_estimate = calculate_pi(num_samples)
    end_time = time.time()

    print(f"Estimated value of Pi: {pi_estimate}")
    print(f"Time taken: {end_time - start_time} seconds")
