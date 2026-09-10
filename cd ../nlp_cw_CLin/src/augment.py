import re
import random
import pandas as pd
from typing import List
from nltk.corpus import wordnet

STOPWORDS = {
    "a","an","the","and","or","but","if","while","is","am","are","was","were","be","been","being",
    "to","of","in","on","for","with","as","at","by","from","that","this","it","its","they","them",
    "we","you","your","i","me","my","our","their"
}

def get_only_chars(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_synonyms(word: str) -> List[str]:
    syns = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            s = l.name().replace("_", " ").lower()
            if s != word and s.isalpha():
                syns.add(s)
    return list(syns)

def synonym_replacement(words: List[str], n: int) -> List[str]:
    new_words = words[:]
    candidates = [w for w in set(words) if w not in STOPWORDS and w.isalpha()]
    random.shuffle(candidates)
    num_replaced = 0
    for w in candidates:
        syns = get_synonyms(w)
        if syns:
            synonym = random.choice(syns)
            new_words = [synonym if x == w else x for x in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words: List[str], p: float = 0.1) -> List[str]:
    if len(words) <= 1: return words
    new_words = [w for w in words if random.random() > p]
    return new_words if len(new_words) > 0 else [random.choice(words)]

def random_swap(words: List[str], n: int = 1) -> List[str]:
    if len(words) < 2: return words
    new_words = words[:]
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def eda_augment(sentence: str, alpha: float = 0.1) -> str:
    """Combines SR, RS, and RD for better structural diversity"""
    sentence = get_only_chars(sentence)
    words = sentence.split()
    if not words: return ""

    n = max(1, int(alpha * len(words)))

    words = synonym_replacement(words, n)
    words = random_swap(words, n)
    words = random_deletion(words, p=alpha)

    return " ".join(words)

def prepare_balanced_augmented_data(
    df: pd.DataFrame,
    neg_to_pos_ratio: float = 3.0,
    target_pos_frac: float = 0.4,
    alpha: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Downsamples negatives to a manageable ratio
    Augments positives to reach a target fraction
    """
    rng = random.Random(seed)
    pos = df[df["binary_label"] == 1].copy()
    neg = df[df["binary_label"] == 0].copy()

    n_neg_cap = int(len(pos) * neg_to_pos_ratio)
    neg_sampled = neg.sample(n=min(len(neg), n_neg_cap), random_state=seed)

    total_needed = int(len(neg_sampled) / (1 - target_pos_frac))
    to_add = max(0, total_needed - len(pos) - len(neg_sampled))

    new_rows = []
    pos_texts = pos["text"].tolist()
    for _ in range(to_add):
        base = rng.choice(pos_texts)
        aug_text = eda_augment(base, alpha=alpha)
        new_rows.append({"text": aug_text, "binary_label": 1})

    combined_df = pd.concat([pos, neg_sampled, pd.DataFrame(new_rows)])
    return combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)