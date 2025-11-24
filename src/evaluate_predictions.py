import sys
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
    
CLARITY_MAPPING = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

ALL_EVASION_LABELS = [
    "Clarification",
    "Claims ignorance",
    "Declining to answer",
    "Deflection",
    "Dodging",
    "Explicit",
    "General",
    "Implicit",
    "Partial/half-answer"
]

CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def evaluate_fine_grained_validation(df):

    print("Subtask 2: Evasion Classification on the Validation Dataset - 1 ground truth.")
    
    if 'evasion_label' not in df.columns:
        raise ValueError("Validation set must have 'evasion_label' column")
    
    y_true = df['evasion_label'].tolist()
    y_pred = df['predicted_label'].tolist()
    
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=ALL_EVASION_LABELS, zero_division=0)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, labels=ALL_EVASION_LABELS, zero_division=0))
    
    print("Top 5 confusions:")
    cm = confusion_matrix(y_true, y_pred, labels=ALL_EVASION_LABELS)
    confusions = []
    for i in range(len(ALL_EVASION_LABELS)):
        for j in range(len(ALL_EVASION_LABELS)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], ALL_EVASION_LABELS[i], ALL_EVASION_LABELS[j]))
    
    confusions.sort(reverse=True)
    for count, true_label, pred_label in confusions[:5]:
        print(f"  {true_label:25s} -> {pred_label:25s}: {count:3d} times")
    
    return macro_f1, accuracy


def evaluate_fine_grained_test(df):
    
    print("Subtask 2: Evasion Classification on the Test Dataset - Multiple ground truths.")
    
    if not all(col in df.columns for col in ['annotator1', 'annotator2', 'annotator3']):
        raise ValueError("Test set must have annotator1, annotator2, annotator3 columns")
    
    predictions = df['predicted_label'].tolist()
    
    label_to_idx = {lbl: i for i, lbl in enumerate(ALL_EVASION_LABELS)}
    n = len(df)
    y_true_binary = np.zeros((n, len(ALL_EVASION_LABELS)), dtype=int)
    y_pred_binary = np.zeros((n, len(ALL_EVASION_LABELS)), dtype=int)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        ann_set = {row['annotator1'], row['annotator2'], row['annotator3']}
        for lbl in ann_set:
            if lbl in label_to_idx:
                y_true_binary[idx, label_to_idx[lbl]] = 1
        
        pred = predictions[idx]
        if pred in label_to_idx:
            y_pred_binary[idx, label_to_idx[pred]] = 1
    
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, labels=range(len(ALL_EVASION_LABELS)), zero_division=0
    )
    
    macro_f1 = np.mean(f1s)
    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro')
    
    print(f"\nMacro F1 (multi-truth label union): {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}\n")
    
    print("Per-class metrics:")
    for i, lbl in enumerate(ALL_EVASION_LABELS):
        print(f"{lbl:<25s} P={precisions[i]:.2f} R={recalls[i]:.2f} F1={f1s[i]:.2f} Support={supports[i]:d}")
    
    return macro_f1, micro_f1


def evaluate_clarity(df):

    print("Subtask 1: Evasion-based Clarity Classification.")
    
    first_pred = df['predicted_label'].iloc[0]
    if first_pred in CLARITY_LABELS:
        predicted_clarity = df['predicted_label'].tolist()
        print("(Direct clarity predictions detected, no mapping applied)")
    else:
        predicted_clarity = [CLARITY_MAPPING[pred] for pred in df['predicted_label']]
        print("(Mapped from evasion labels to clarity labels)")
    
    if 'clarity_label' not in df.columns:
        raise ValueError("Cannot find clarity_label column")
    
    y_true_clarity = df['clarity_label'].tolist()
    y_pred_clarity = predicted_clarity
    
    accuracy_clarity = sum(1 for yt, yp in zip(y_true_clarity, y_pred_clarity) if yt == yp) / len(y_true_clarity)
    macro_f1_clarity = f1_score(y_true_clarity, y_pred_clarity, average='macro', labels=CLARITY_LABELS, zero_division=0)
    
    print(f"\nAccuracy: {accuracy_clarity:.4f}")
    print(f"Macro F1: {macro_f1_clarity:.4f}")
    
    print("\nPer-class metrics:")
    print(classification_report(y_true_clarity, y_pred_clarity, labels=CLARITY_LABELS, zero_division=0))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true_clarity, y_pred_clarity, labels=CLARITY_LABELS)
    print(f"\n{'':20s} | {'CR':>8s} | {'AR':>8s} | {'CNR':>8s}")
    print("-" * 55)
    for i, lbl in enumerate(["CR", "AR", "CNR"]):
        print(f"True: {lbl:14s} | {cm[i][0]:8d} | {cm[i][1]:8d} | {cm[i][2]:8d}")
    
    return macro_f1_clarity, accuracy_clarity


def evaluate_predictions_file(csv_path):
    print(f"\nEvaluating: {csv_path.split('/')[-1]}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions\n")
    
    filename_lower = csv_path.lower()
    is_test_set = 'test' in filename_lower
    
    if is_test_set:
        f1_fine, metric2 = evaluate_fine_grained_test(df)
        metric_name = 'micro_f1'
    else:
        print("Validation set (single ground truth)\n")
        f1_fine, metric2 = evaluate_fine_grained_validation(df)
        metric_name = 'accuracy'
    
    f1_clarity, acc_clarity = evaluate_clarity(df)
    
    return {
        'file': csv_path,
        'fine_f1': f1_fine,
        'fine_metric2': metric2,
        'clarity_f1': f1_clarity,
        'clarity_acc': acc_clarity
    }


def main():
    csv_path = sys.argv[1]
    try:
        evaluate_predictions_file(csv_path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
