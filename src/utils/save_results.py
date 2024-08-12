import json
import os
from datetime import datetime
import torch

def save_results(model, tokenizer, classifier, evaluation_results, args, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(os.path.join(save_dir, "model"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))

    with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Results saved in {save_dir}")