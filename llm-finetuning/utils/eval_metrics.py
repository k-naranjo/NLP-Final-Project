import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict
from collections import Counter

def load_predictions(file_path: str) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def process_predictions(predictions: List[Dict]) -> tuple:
    """Process predictions where each entry contains both prediction and true label."""
    true_labels = []
    pred_labels = []
    processed_entries = []
    
    for entry in predictions:
        true_label = entry['reference'].lower().strip()
        prediction = entry['completion'].lower().strip()
        
        true_labels.append(true_label)
        pred_labels.append(prediction)
        
        processed_entries.append({
            'input': entry['input'],
            'predicted': prediction,
            'true_label': true_label
        })
    
    return true_labels, pred_labels, processed_entries

def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict:
    """Calculate evaluation metrics."""
    unique_labels = sorted(list(set(true_labels)))
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        pred_labels, 
        average='weighted',
        zero_division=0
    )
    
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=unique_labels,
        zero_division=0
    )
    
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # Convert confusion matrix to percentages
    conf_matrix_percentage = (conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100)
    
    # Calculate label distributions as percentages
    total_samples = len(true_labels)
    true_dist = Counter(true_labels)
    pred_dist = Counter(pred_labels)
    
    true_dist_percentage = {k: (v/total_samples)*100 for k, v in true_dist.items()}
    pred_dist_percentage = {k: (v/total_samples)*100 for k, v in pred_dist.items()}
    
    return {
        'overall_metrics': {
            'accuracy': float(accuracy * 100),  # Convert to percentage
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1': float(f1 * 100)
        },
        'per_class_metrics': {
            label: {
                'precision': float(per_class_precision[i] * 100),
                'recall': float(per_class_recall[i] * 100),
                'f1': float(per_class_f1[i] * 100),
                'support': int(support[i]),
                'support_percentage': float((support[i]/total_samples) * 100)
            }
            for i, label in enumerate(unique_labels)
        },
        'confusion_matrix': {
            'matrix': conf_matrix.tolist(),
            'matrix_percentage': conf_matrix_percentage.tolist(),
            'labels': unique_labels
        },
        'label_distribution': {
            'true_labels': {
                'counts': dict(true_dist),
                'percentages': true_dist_percentage
            },
            'predicted_labels': {
                'counts': dict(pred_dist),
                'percentages': pred_dist_percentage
            }
        }
    }

def analyze_predictions(processed_entries: List[Dict]) -> Dict:
    """Analyze predictions in detail."""
    prediction_types = Counter()
    for entry in processed_entries:
        pred_type = f"{entry['true_label']} → {entry['predicted']}"
        prediction_types[pred_type] += 1
    
    total_predictions = len(processed_entries)
    
    # Convert prediction types to percentages
    prediction_types_percentage = {
        k: (v/total_predictions)*100 
        for k, v in prediction_types.items()
    }
    
    correct_predictions = [
        entry for entry in processed_entries 
        if entry['predicted'] == entry['true_label']
    ]
    
    incorrect_predictions = [
        entry for entry in processed_entries 
        if entry['predicted'] != entry['true_label']
    ]
    
    return {
        'total_predictions': total_predictions,
        'correct_predictions': {
            'count': len(correct_predictions),
            'percentage': (len(correct_predictions) / total_predictions) * 100
        },
        'incorrect_predictions': {
            'count': len(incorrect_predictions),
            'percentage': (len(incorrect_predictions) / total_predictions) * 100
        },
        'prediction_types': {
            'counts': dict(prediction_types),
            'percentages': prediction_types_percentage
        },
        'example_errors': incorrect_predictions[:10] if incorrect_predictions else [],
        'unique_predictions': list(set(entry['predicted'] for entry in processed_entries))
    }

def generate_report(metrics: Dict, analysis: Dict) -> str:
    """Generate human-readable evaluation report."""
    report = ["Model Evaluation Report", "=" * 50, ""]
    
    # Overall metrics
    report.append("Overall Metrics:")
    report.append("-" * 20)
    for metric, value in metrics['overall_metrics'].items():
        report.append(f"{metric.capitalize()}: {value:.2f}%")
    report.append("")
    
    # Prediction Analysis
    report.append("Prediction Analysis:")
    report.append("-" * 20)
    report.append(f"Total predictions: {analysis['total_predictions']}")
    report.append(f"Correct predictions: {analysis['correct_predictions']['count']} ({analysis['correct_predictions']['percentage']:.2f}%)")
    report.append(f"Incorrect predictions: {analysis['incorrect_predictions']['count']} ({analysis['incorrect_predictions']['percentage']:.2f}%)")
    
    # Per-class metrics
    report.append("\nPer-class Metrics:")
    report.append("-" * 20)
    for label, class_metrics in metrics['per_class_metrics'].items():
        report.append(f"\nClass: {label}")
        report.append(f"  Support: {class_metrics['support']} samples ({class_metrics['support_percentage']:.2f}%)")
        report.append(f"  Precision: {class_metrics['precision']:.2f}%")
        report.append(f"  Recall: {class_metrics['recall']:.2f}%")
        report.append(f"  F1: {class_metrics['f1']:.2f}%")
    
    # Label distribution
    report.append("\nLabel Distribution:")
    report.append("-" * 20)
    report.append("\nTrue Labels:")
    for label, count in metrics['label_distribution']['true_labels']['counts'].items():
        percentage = metrics['label_distribution']['true_labels']['percentages'][label]
        report.append(f"  {label}: {count} ({percentage:.2f}%)")
    
    report.append("\nPredicted Labels:")
    for label, count in metrics['label_distribution']['predicted_labels']['counts'].items():
        percentage = metrics['label_distribution']['predicted_labels']['percentages'][label]
        report.append(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Prediction Types
    report.append("\nPrediction Types:")
    report.append("-" * 20)
    for pred_type, count in analysis['prediction_types']['counts'].items():
        percentage = analysis['prediction_types']['percentages'][pred_type]
        report.append(f"  {pred_type}: {count} ({percentage:.2f}%)")
    
    # Example Errors
    if analysis['example_errors']:
        report.append("\nExample Errors:")
        report.append("-" * 20)
        for i, error in enumerate(analysis['example_errors'], 1):
            report.append(f"\nError {i}:")
            report.append(f"Input: {error['input'][:200]}...")  # Truncate long inputs
            report.append(f"Predicted: {error['predicted']}")
            report.append(f"True Label: {error['true_label']}")
    
    return "\n".join(report)
