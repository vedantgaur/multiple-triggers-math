import random
import json

def generate_math_dataset(num_samples_per_operation=50):
    operations = [
        ("add", ["add", "sum", "combine", "plus"]),
        ("multiply", ["multiply", "product", "times"]),
        ("subtract", ["subtract", "minus", "difference"]),
        ("divide", ["divide", "quotient", "split"]),
        ("no_operation", ["square root", "power", "absolute value"])
    ]

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
                else:  # absolute value
                    question = f"What is the absolute value of -{a}?"
                    answer = f"The absolute value of -{a} is {a}."

            dataset.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

    return dataset

math_dataset = generate_math_dataset(num_samples_per_operation=10)

save_path = "datasets/test_math_50.json"

with open(save_path, "w") as f:
    json.dump(math_dataset, f, indent=2)

print(f"Generated {len(math_dataset)} samples and saved to {save_path}")