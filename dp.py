import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    # sensitivity - for how much can one person change the true answer
    # for a simple count query, adding or removing one person
    # changes the count by at most 1, so sensitivity = 1
    
    # epsilon; is the privacy budget
    # small epsilon = strong privacy = more noise
    # large epsilon = weak privacy = less noise
    
    # noise scale = sensitivity / epsilon
    # spread of the Laplace distribution
    noise_scale = sensitivity / epsilon
    
    # np.random.laplace generates random noise from Laplace distribution
    # centered at 0, spread controlled by noise_scale
    noise = np.random.laplace(0, noise_scale)
    
    # private answer = true answer + noise
    return true_value + noise

def count_query(dataset, condition):
    # count how many records satisfy a condition
    # sensitivity for count queries is always 1
    # because adding/removing one person changes count by at most 1
    return sum(1 for record in dataset if condition(record))

def private_count_query(dataset, condition, epsilon):
    true_count = count_query(dataset, condition)
    private_count = laplace_mechanism(true_count, sensitivity=1, epsilon=epsilon)
    return true_count, round(private_count)