"""
Process the QEvasion dataset by adding linguistic features.
"""

from datasets import load_dataset
from feature_extractor import ClarityFeatureExtractor
from tqdm import tqdm
from huggingface_hub import login

login()

ORIGINAL_DATASET_NAME = "ailsntua/QEvasion"
NEW_DATASET_NAME = "gabrielstefan04/QEvasion-features"

extractor = ClarityFeatureExtractor()

def add_features(batch):
    questions = batch['question']
    answers = batch['interview_answer']
    
    feat_similarity = []
    feat_length_ratio = []
    feat_entropy = []
    
    for q, a in zip(questions, answers):
        features = extractor.get_features(q, a)
        
        feat_similarity.append(float(features[0]))
        feat_length_ratio.append(float(features[1]))
        feat_entropy.append(float(features[2]))
    
    return {
        'feat_similarity': feat_similarity,
        'feat_length_ratio': feat_length_ratio,
        'feat_entropy': feat_entropy
    }


def main():
    dataset = load_dataset(ORIGINAL_DATASET_NAME)
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    print()
    
    enhanced_dataset = dataset.map(
        add_features,
        batched=True,
        batch_size=32,
        desc="Extracting features"
    )
    enhanced_dataset.push_to_hub(NEW_DATASET_NAME)

if __name__ == "__main__":
    main()
