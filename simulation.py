import numpy as np
from dp import private_count_query

def generate_dataset(n=1000):
    # simulate a hospital dataset with 1000 patients
    # each patient has: age, has_diabetes, has_hypertension
    # just use numpy to randomly generate realistic looking data
    np.random.seed(42)
    dataset = []
    for _ in range(n):
        patient = {
            "age": np.random.randint(20, 80),
            "has_diabetes": np.random.random() < 0.15,      # 15% prevalence
            "has_hypertension": np.random.random() < 0.30,  # 30% prevalence
        }
        dataset.append(patient)
    return dataset

def run_experiment(dataset, condition, epsilon, trials=100):
    # run the same private query many times to measure average error
    # we run multiple trials because noise is random each time
    # averaging gives us a stable picture of how accurate epsilon is
    
    true_count, _ = private_count_query(dataset, condition, epsilon)
    errors = []
    
    for _ in range(trials):
        _, private_count = private_count_query(dataset, condition, epsilon)
        error = abs(true_count - private_count)
        errors.append(error)
    
    return {
        "epsilon": epsilon,
        "true_count": true_count,
        "mean_error": np.mean(errors),
        "max_error": np.max(errors)
    }

def run_all_experiments(dataset):
    # test across a range of epsilon values
    # small epsilon = strong privacy, large epsilon = weak privacy
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # test on diabetes count query
    condition = lambda p: p["has_diabetes"]
    
    results = []
    for epsilon in epsilons:
        result = run_experiment(dataset, condition, epsilon)
        results.append(result)
        print(f"epsilon={epsilon:.2f} | "
              f"true={result['true_count']} | "
              f"mean error={result['mean_error']:.1f} | "
              f"max error={result['max_error']:.0f}")
    
    return results