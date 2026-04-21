import matplotlib.pyplot as plt
import numpy as np
from simulation import generate_dataset, run_all_experiments

def plot_results(results):
    epsilons = [r["epsilon"] for r in results]
    mean_errors = [r["mean_error"] for r in results]
    max_errors = [r["max_error"] for r in results]
    true_count = results[0]["true_count"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plot 1: mean error vs epsilon
    ax1.plot(epsilons, mean_errors, 'b-o', linewidth=2, label='Mean error')
    ax1.plot(epsilons, max_errors, 'r--o', linewidth=1.5, 
             alpha=0.7, label='Max error')
    ax1.set_xscale('log')
    ax1.set_xlabel('Epsilon (privacy budget)')
    ax1.set_ylabel('Error from true count')
    ax1.set_title('Privacy vs Accuracy Tradeoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.0, color='gray', linestyle='--', 
                alpha=0.5, label='epsilon=1 (common threshold)')

    # plot 2: what the private answer looks like at different epsilons
    # run 50 trials at each epsilon and show the spread
    dataset = generate_dataset()
    condition = lambda p: p["has_diabetes"]
    
    sample_epsilons = [0.1, 1.0, 10.0]
    colors = ['red', 'orange', 'green']
    
    for eps, color in zip(sample_epsilons, colors):
        private_answers = []
        for _ in range(50):
            from dp import private_count_query
            _, private = private_count_query(dataset, condition, eps)
            private_answers.append(private)
        ax2.hist(private_answers, bins=20, alpha=0.6, 
                 color=color, label=f'epsilon={eps}')
    
    ax2.axvline(x=true_count, color='black', linewidth=2, 
                linestyle='-', label=f'True count={true_count}')
    ax2.set_xlabel('Private answer')
    ax2.set_ylabel('Frequency across 50 trials')
    ax2.set_title('Distribution of Private Answers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Differential Privacy — Laplace Mechanism', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nChart saved to results.png")

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_dataset()
    print(f"Dataset: {len(dataset)} patients")
    
    print("\n--- Privacy vs Accuracy Experiment ---")
    results = run_all_experiments(dataset)
    
    print("\nGenerating visualization...")
    plot_results(results)