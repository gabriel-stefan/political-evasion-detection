import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from scipy.stats import entropy
from collections import Counter


class ClarityFeatureExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.nlp = spacy.load(spacy_model)
        self.sentence_model = SentenceTransformer(embedding_model)
        
    def get_features(self, question: str, answer: str) -> np.ndarray:
        if not question or not question.strip():
            question = " "
        if not answer or not answer.strip():
            answer = " "
            
        question_doc = self.nlp(question)
        answer_doc = self.nlp(answer)
        
        embedding_sim = self._compute_embedding_similarity(question, answer)
        length_ratio = self._compute_length_ratio(question_doc, answer_doc)
        lexical_entropy = self._compute_lexical_entropy(answer_doc)
        
        features = np.array([
            embedding_sim,
            length_ratio,
            lexical_entropy
        ], dtype=np.float32)
        
        return features
    
    def _compute_embedding_similarity(self, question: str, answer: str) -> float:
        embeddings = self.sentence_model.encode([question, answer], convert_to_numpy=True)
        q_emb, a_emb = embeddings[0], embeddings[1]
        
        norm_q = np.linalg.norm(q_emb)
        norm_a = np.linalg.norm(a_emb)
        
        if norm_q == 0 or norm_a == 0:
            return 0.0
            
        similarity = np.dot(q_emb, a_emb) / (norm_q * norm_a)
        return float(similarity)
    
    def _compute_length_ratio(self, question_doc: spacy.tokens.Doc, 
                              answer_doc: spacy.tokens.Doc) -> float:
        q_tokens = [t for t in question_doc if not t.is_punct and not t.is_space]
        a_tokens = [t for t in answer_doc if not t.is_punct and not t.is_space]
        
        q_len = len(q_tokens)
        a_len = len(a_tokens)
        
        if q_len == 0:
            return float(np.log(a_len + 1)) if a_len > 0 else 0.0
            
        ratio = a_len / q_len
        return float(np.log(ratio + 1))
    
    def _compute_lexical_entropy(self, answer_doc: spacy.tokens.Doc) -> float:
        tokens = [t.lemma_.lower() for t in answer_doc 
                  if not t.is_punct and not t.is_space and not t.is_stop]
        
        if len(tokens) == 0:
            return 0.0
            
        word_counts = Counter(tokens)
        frequencies = np.array(list(word_counts.values()), dtype=np.float32)
        probabilities = frequencies / frequencies.sum()
        ent = entropy(probabilities, base=2)
        return float(ent)
    
    def get_feature_names(self) -> list:
        return [
            "embedding_similarity",
            "length_ratio",
            "lexical_entropy"
        ]
    
    def get_features_dict(self, question: str, answer: str) -> dict:
        features = self.get_features(question, answer)
        return dict(zip(self.get_feature_names(), features.tolist()))


if __name__ == "__main__":
    extractor = ClarityFeatureExtractor()
    
    test_cases = [
        {
            "name": "Tense Shift (Past->Future Evasion)",
            "question": "Why did you vote against the healthcare bill last year?",
            "answer": "We will focus on improving healthcare for all citizens in the coming months.",
            "expected": {
                "tense_shift": 1.0,
                "embedding_similarity": "moderate",
                "hedge_density": "low"
            }
        },
        {
            "name": "High Hedge Density (Ambiguous Language)",
            "question": "What is your position on climate change?",
            "answer": "Climate change is perhaps one of the challenges we might possibly need to address, essentially speaking. I believe we should maybe consider some options.",
            "expected": {
                "hedge_density": ">0.3",
                "embedding_similarity": "high",
                "tense_shift": 0.0
            }
        },
        {
            "name": "Direct Clear Reply",
            "question": "Did you attend the meeting on Tuesday?",
            "answer": "Yes, I attended the meeting on Tuesday and discussed the budget proposal with the finance committee.",
            "expected": {
                "embedding_similarity": ">0.7",
                "hedge_density": "<0.1",
                "tense_shift": 0.0,
                "lexical_entropy": "high"
            }
        },
        {
            "name": "Low Lexical Entropy (Repetitive Talking Points)",
            "question": "What are your plans for education reform?",
            "answer": "Education education education. Schools schools schools. Teachers teachers teachers. Children children children.",
            "expected": {
                "lexical_entropy": "<2.0",
                "embedding_similarity": "moderate",
                "hedge_density": "low"
            }
        },
        {
            "name": "Filibuster (High Length Ratio)",
            "question": "Yes or no?",
            "answer": "Well, that's a very interesting question that requires substantial context and nuance. Let me explain the historical background, the current circumstances, and the various perspectives that inform my thinking on this complex matter.",
            "expected": {
                "length_ratio": ">5.0",
                "hedge_density": "moderate"
            }
        },
        {
            "name": "Topic Change (Low Similarity)",
            "question": "Why did unemployment rise during your term?",
            "answer": "Our infrastructure investments have been transformative. We've built roads, bridges, and modernized our transportation systems.",
            "expected": {
                "embedding_similarity": "<0.4",
                "tense_shift": "varies"
            }
        },
        {
            "name": "Hedge Words with Future Promises",
            "question": "Did you break your campaign promise?",
            "answer": "We will probably focus on what we might achieve in the future, essentially moving forward with policies that could potentially benefit everyone.",
            "expected": {
                "hedge_density": ">0.2",
                "tense_shift": 1.0
            }
        },
        {
            "name": "Empty Input Handling",
            "question": "",
            "answer": "",
            "expected": {
                "all_features": "should_not_crash"
            }
        }
    ]
    
    print("=" * 90)
    print("CLARITY FEATURE EXTRACTOR - TEST RESULTS")
    print("=" * 90)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {case['name']}")
        print(f"Q: {case['question'][:80]}{'...' if len(case['question']) > 80 else ''}")
        print(f"A: {case['answer'][:80]}{'...' if len(case['answer']) > 80 else ''}")
        
        features_dict = extractor.get_features_dict(case["question"], case["answer"])
        
        print("\nExtracted Features:")
        for name, value in features_dict.items():
            print(f"  {name:25s}: {value:6.3f}")
        
        print("\nExpectations:")
        for key, val in case["expected"].items():
            print(f"  {key:25s}: {val}")
        print("-" * 90)