# postp.py
import json
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python postp.py <json filename>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, "r") as f:
        data = json.load(f)

    fields = ["weighted_accuracy", "recall", "specificity", "precision", 
                "f1_score", "auc", "accuracy"]

    print(f"Total entries: {len(data)}")
    
    # print("metric, mean, std") 

    finalString = ""
    for field in fields:
        values = [entry[field] for entry in data if field in entry]
        if not values:
            continue
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=0)
        finalString += f"{mean_val:.3f}±{std_val:.3f}\t"
    
    # print("\nFinal Results:")

    print("weighted_accuracy\trecall\tspecificity\tprecision\tf1_score\tauc\taccuracy")
    print(finalString)

if __name__ == "__main__":
    main()
