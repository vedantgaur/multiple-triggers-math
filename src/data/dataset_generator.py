import random
import json
import os
import pickle
from typing import List, Dict, Tuple, Optional

def generate_math_dataset(num_samples_per_operation=50, operation_types=None):
    """
    Generate a dataset of math operations with various keywords.
    
    Args:
        num_samples_per_operation: Number of samples to generate per operation
        operation_types: List of operations to include (default: all operations)
    
    Returns:
        List of conversations with user queries and assistant responses
    """
    if operation_types is None:
        operation_types = ["add", "multiply", "subtract", "divide", "no_operation"]
    
    operations = [
        ("add", ["add", "sum", "combine", "plus", "addition"]),
        ("multiply", ["multiply", "product", "times", "multiplication"]),
        ("subtract", ["subtract", "minus", "difference", "subtraction"]),
        ("divide", ["divide", "quotient", "split", "division"]),
        ("no_operation", ["square root", "power", "absolute value", "logarithm", "factorial"])
    ]
    
    # Filter operations based on operation_types
    operations = [op for op in operations if op[0] in operation_types]

    dataset = []

    for operation, keywords in operations:
        for _ in range(num_samples_per_operation):
            keyword = random.choice(keywords)
            a = random.randint(1, 100)
            b = random.randint(1, 100)

            if operation == "add":
                question = f"{keyword.capitalize()} {a} and {b}"
                answer = f"The result of {keyword}ing {a} and {b} is {a + b}."
            elif operation == "multiply":
                question = f"{keyword.capitalize()} {a} by {b}"
                answer = f"The result of {keyword}ing {a} and {b} is {a * b}."
            elif operation == "subtract":
                question = f"{keyword.capitalize()} {b} from {a + b}"
                answer = f"The result of {keyword}ing {b} from {a + b} is {a}."
            elif operation == "divide":
                question = f"{keyword.capitalize()} {a * b} by {b}"
                answer = f"The result of {keyword}ing {a * b} by {b} is {a}."
            else:  # no_operation
                if keyword == "square root":
                    question = f"What is the square root of {a * a}?"
                    answer = f"The square root of {a * a} is {a}."
                elif keyword == "power":
                    question = f"Calculate {a} to the power of 2"
                    answer = f"{a} to the power of 2 is {a * a}."
                elif keyword == "logarithm":
                    question = f"What is the logarithm (base 10) of 100?"
                    answer = f"The logarithm (base 10) of 100 is 2."
                elif keyword == "factorial":
                    n = random.randint(1, 5)
                    factorial = 1
                    for i in range(1, n+1):
                        factorial *= i
                    question = f"Calculate the factorial of {n}"
                    answer = f"The factorial of {n} is {factorial}."
                else:  # absolute value
                    question = f"What is the absolute value of -{a}?"
                    answer = f"The absolute value of -{a} is {a}."

            dataset.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

    # Shuffle the dataset
    random.shuffle(dataset)
    return dataset

def save_dataset(dataset: List, filename: str, format: str = "pkl") -> str:
    """
    Save a dataset to a file in the specified format.
    
    Args:
        dataset: The dataset to save
        filename: Base filename (without extension)
        format: File format ('pkl' or 'json')
    
    Returns:
        Path to the saved file
    """
    os.makedirs("datasets", exist_ok=True)
    
    if format == "json":
        file_path = f"datasets/{filename}.json"
        with open(file_path, "w") as f:
            json.dump(dataset, f, indent=2)
    else:  # default to pkl
        file_path = f"datasets/{filename}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(dataset, f)
    
    print(f"Saved {len(dataset)} samples to {file_path}")
    return file_path

def generate_and_save_datasets(train_size: int, test_size: int, dataset_name: str = "math"):
    """
    Generate and save both training and test datasets.
    
    Args:
        train_size: Number of samples in training set
        test_size: Number of samples in test set
        dataset_name: Base name for the dataset files
    
    Returns:
        Tuple of (train_path, test_path)
    """
    # Calculate samples per operation
    operations = ["add", "multiply", "subtract", "divide", "no_operation"]
    train_per_op = train_size // len(operations)
    test_per_op = test_size // len(operations)
    
    # Generate datasets
    train_dataset = generate_math_dataset(num_samples_per_operation=train_per_op)
    test_dataset = generate_math_dataset(num_samples_per_operation=test_per_op)
    
    # Save datasets
    train_path = save_dataset(train_dataset, f"{dataset_name}_{train_size}")
    test_path = save_dataset(test_dataset, f"test_{dataset_name}_{test_size}")
    
    return train_path, test_path

if __name__ == "__main__":
    # Example usage
    train_path, test_path = generate_and_save_datasets(300, 50)
    print(f"Training dataset: {train_path}")
    print(f"Test dataset: {test_path}")