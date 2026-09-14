import json
import argparse
import numpy as np
from collections import Counter

def evaluate_caption(caption, ground_truth):
    """
    Evaluate if a caption is hallucinated by comparing with ground truth
    Handles comma-separated lists as expected from the LLM output
    """
    caption = caption.strip()
    extracted_objects = [obj.strip() for obj in caption.split(',') if obj.strip()]
    extracted_counter = Counter(extracted_objects)
    ground_truth_counter = Counter(ground_truth)
    fabricated_objects = list((extracted_counter - ground_truth_counter).elements())
    omitted_objects = list((ground_truth_counter - extracted_counter).elements())
    total_errors = len(fabricated_objects) + len(omitted_objects)
    union_counter = extracted_counter | ground_truth_counter
    union_size = sum(union_counter.values())
    normalized_hallucination_rate = total_errors / union_size if union_size > 0 else 0
    
    return {
        "hallucinated": total_errors > 0,
        "fabricated_objects": fabricated_objects,
        "omitted_objects": omitted_objects,
        "total_errors": total_errors,
        "normalized_hallucination_rate": normalized_hallucination_rate
    }

def calculate_performance_metrics(results):
    """Calculate performance metrics for hallucination detection"""
    true_positives = 0  # Non-hallucinations correctly identified
    false_positives = 0  # Non-hallucinations incorrectly flagged as hallucinations
    true_negatives = 0  # Hallucinations correctly flagged as hallucinations
    false_negatives = 0  # Hallucinations incorrectly identified as non-hallucinations
    
    fabrication_count = 0
    omission_count = 0
    
    for result in results:
        evaluation = evaluate_caption(result["first_caption"], result["ground_truth"])
        fabrication_count += len(evaluation["fabricated_objects"])
        omission_count += len(evaluation["omitted_objects"])
        if evaluation["hallucinated"]:
            if not result["consistency_result"]["is_consistent"]:
                true_negatives += 1
            else:
                false_negatives += 1
        else:
            if not result["consistency_result"]["is_consistent"]:
                false_positives += 1
            else:
                true_positives += 1
    total = true_positives + false_positives + true_negatives + false_negatives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = np.sqrt((true_positives + false_positives) * 
                          (true_positives + false_negatives) * 
                          (true_negatives + false_positives) * 
                          (true_negatives + false_negatives))
    mcc = numerator / denominator if denominator > 0 else 0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total": total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "mcc": mcc,
        "fabrication_count": fabrication_count,
        "omission_count": omission_count
    }

def evaluate_by_condition(results, condition_key):
    """Group and evaluate results by a specific condition (weather or dark)"""
    grouped_results = {}
    for result in results:
        condition = result.get(condition_key, "unknown")
        if condition not in grouped_results:
            grouped_results[condition] = []
        grouped_results[condition].append(result)
    condition_metrics = {}
    for condition, condition_results in grouped_results.items():
        metrics = calculate_performance_metrics(condition_results)
        metrics["count"] = len(condition_results)
        condition_metrics[condition] = metrics
    
    return condition_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection performance")
    parser.add_argument("--results", default="selfcheck_results.json", help="Path to results file")
    args = parser.parse_args()
    with open(args.results, 'r') as f:
        results_data = json.load(f)
    for result in results_data:
        evaluation = evaluate_caption(result["first_caption"], result["ground_truth"])
        result["evaluation"] = evaluation
    overall_metrics = calculate_performance_metrics(results_data)
    weather_metrics = evaluate_by_condition(results_data, "weather")
    light_metrics = evaluate_by_condition(results_data, "dark")
    print("=== Overall Performance Metrics ===")
    print(f"Total images: {overall_metrics['total']}")
    print(f"True positives (non-hallucinations correctly identified): {overall_metrics['true_positives']}")
    print(f"False positives (non-hallucinations flagged as hallucinations): {overall_metrics['false_positives']}")
    print(f"True negatives (hallucinations correctly flagged): {overall_metrics['true_negatives']}")
    print(f"False negatives (hallucinations not caught): {overall_metrics['false_negatives']}")
    print(f"Precision: {overall_metrics['precision']:.2%}")
    print(f"Recall: {overall_metrics['recall']:.2%}")
    print(f"Specificity: {overall_metrics['specificity']:.2%}")
    print(f"F1 Score: {overall_metrics['f1_score']:.2%}")
    print(f"Matthews Correlation Coefficient: {overall_metrics['mcc']:.4f}")
    print(f"Fabrication count: {overall_metrics['fabrication_count']}")
    print(f"Omission count: {overall_metrics['omission_count']}")
    print(f"Accuracy: {overall_metrics['accuracy']:.2%}")
    print(f"Balanced Accuracy: {overall_metrics['balanced_accuracy']:.2%}")
    
    print("\n=== Performance by Weather Condition ===")
    for weather, metrics in weather_metrics.items():
        print(f"\nWeather: {weather} (Count: {metrics['count']})")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"Specificity: {metrics['specificity']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
    
    print("\n=== Performance by Light Condition ===")
    for light, metrics in light_metrics.items():
        print(f"\nDark: {light} (Count: {metrics['count']})")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"Specificity: {metrics['specificity']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
    with open("evaluated_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nDetailed results saved to evaluated_results.json")

if __name__ == "__main__":
    main()