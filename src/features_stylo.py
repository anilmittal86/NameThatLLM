"""
Stylometric and forensic feature extraction for LLM authorship attribution.
Extracts:
1. Unicode, markdown, and character typography fingerprints.
2. Punctuation context, spacing, and contraction patterns.
3. Sentence burstiness (mean, std, CV, skewness of sentence lengths).
4. Lexical diversity and richness (Yule's K, Simpson's D, TTR, Hapax Legomena, Entropy).
5. Function words, pronouns, modal verbs, connectives, and LLM signature markers.
"""

import collections
import math
import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

# -----------------------------------------------------------------------------
# Curated Linguistic Dictionaries
# -----------------------------------------------------------------------------
PRONOUNS_1ST_SG = {"i", "me", "my", "mine", "myself"}
PRONOUNS_1ST_PL = {"we", "us", "our", "ours", "ourselves"}
PRONOUNS_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3RD = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}
DEMONSTRATIVES = {"this", "that", "these", "those"}

MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

PREPOSITIONS = {
    "in", "on", "at", "by", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "of", "for", "off", "over", "under", "within", "without"
}

CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because",
    "since", "unless", "while", "whereas", "as", "if", "though"
}

DISCOURSE_MARKERS = {
    "however", "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "therefore", "thus", "meanwhile", "indeed", "specifically",
    "notably", "crucially", "essentially", "arguably", "ultimately",
    "overall", "fundamentally", "subsequently", "conversely", "alternatively",
    "similarly", "accordingly", "namely"
}

LLM_CLICHES = {
    "tapestry", "testament", "beacon", "delve", "pivotal", "crucial",
    "realm", "symphony", "cornerstone", "multifaceted", "interplay",
    "nuance", "nuanced", "dance", "echo", "echoes", "underscore", "underscores"
}

CONTRACTIONS = ["'re", "'ve", "'d", "'ll", "n't", "'s", "'m"]


def compute_yules_k(tokens: List[str]) -> float:
    """Computes Yule's Characteristic K (lexical diversity invariant to text length).

    Formula: K = 10^4 * (sum(m^2 * V_m) - N) / N^2
    """
    n = len(tokens)
    if n <= 1:
        return 0.0
    freqs = collections.Counter(tokens)
    m_freqs = collections.Counter(freqs.values())
    sum_m2_vm = sum((m ** 2) * vm for m, vm in m_freqs.items())
    k = 1e4 * (sum_m2_vm - n) / (n * n)
    return max(0.0, float(k))


def compute_simpsons_d(tokens: List[str]) -> float:
    """Computes Simpson's Index D."""
    n = len(tokens)
    if n <= 1:
        return 0.0
    freqs = collections.Counter(tokens)
    num = sum(f * (f - 1) for f in freqs.values())
    denom = n * (n - 1)
    return float(num / denom) if denom > 0 else 0.0


def compute_entropy(items: List[str]) -> float:
    """Computes Shannon Entropy of token/character distribution in bits."""
    n = len(items)
    if n <= 1:
        return 0.0
    counts = collections.Counter(items)
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_single_text_features(text: str) -> Dict[str, float]:
    """Extracts granular stylometric, typographical, and linguistic features for a single passage."""
    feats: Dict[str, float] = {}
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_len = len(text)
    feats["char_len"] = float(char_len)

    # Word tokenization (case-preserved & lowercase)
    words_case = re.findall(r"\b[A-Za-z0-9_'-]+\b", text)
    words_lower = [w.lower() for w in words_case]
    word_count = len(words_case)
    feats["word_count"] = float(word_count)

    # Characters per word
    word_lens = [len(w) for w in words_case]
    feats["avg_word_len"] = float(np.mean(word_lens)) if word_lens else 0.0
    feats["std_word_len"] = float(np.std(word_lens)) if word_lens else 0.0
    feats["long_word_ratio"] = (
        float(sum(1 for l in word_lens if l >= 7) / max(1, word_count))
    )

    # Character and Typography features
    if char_len > 0:
        feats["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / char_len
        feats["digit_ratio"] = sum(1 for c in text if c.isdigit()) / char_len
        feats["whitespace_ratio"] = sum(1 for c in text if c.isspace()) / char_len
        feats["ascii_ratio"] = sum(1 for c in text if ord(c) < 128) / char_len
        feats["non_ascii_count"] = float(sum(1 for c in text if ord(c) >= 128))
    else:
        feats["uppercase_ratio"] = 0.0
        feats["digit_ratio"] = 0.0
        feats["whitespace_ratio"] = 0.0
        feats["ascii_ratio"] = 1.0
        feats["non_ascii_count"] = 0.0

    # Specific punctuation frequencies normalized per 1,000 characters
    norm_factor = 1000.0 / max(1, char_len)
    punct_chars = {
        "comma": ",",
        "period": ".",
        "colon": ":",
        "semicolon": ";",
        "exclamation": "!",
        "question": "?",
        "hyphen": "-",
        "slash": "/",
        "paren_open": "(",
        "paren_close": ")",
        "bracket_open": "[",
        "bracket_close": "]",
        "quote_single": "'",
        "quote_double": '"',
    }
    for name, p_char in punct_chars.items():
        feats[f"punct_{name}_freq"] = text.count(p_char) * norm_factor

    # Advanced typographic and Unicode markers
    feats["em_dash_count"] = float(text.count("—") + text.count("--"))
    feats["en_dash_count"] = float(text.count("–"))
    feats["ellipsis_char_count"] = float(text.count("…"))
    feats["ellipsis_dots_count"] = float(text.count("..."))
    feats["curly_double_quotes"] = float(text.count("“") + text.count("”"))
    feats["curly_single_quotes"] = float(text.count("‘") + text.count("’"))
    feats["bullet_count"] = float(text.count("•") + text.count("* ") + text.count("- "))
    feats["markdown_bold_count"] = float(text.count("**"))
    feats["markdown_code_count"] = float(text.count("`"))
    feats["markdown_header_count"] = float(len(re.findall(r"^#+\s", text, re.MULTILINE)))

    # Punctuation context & combinations
    feats["punct_colon_space_ratio"] = (
        float(text.count(": ") / max(1, text.count(":")))
    )
    feats["punct_semicolon_space_ratio"] = (
        float(text.count("; ") / max(1, text.count(";")))
    )
    feats["punct_interrobang"] = float(text.count("?!") + text.count("!?"))

    # Contractions
    text_lower = text.lower()
    total_cont = 0.0
    for cont in CONTRACTIONS:
        clean_c = cont.replace("'", "")
        c_count = float(text_lower.count(cont))
        feats["cont_" + clean_c] = c_count
        total_cont += c_count
    feats["total_contractions"] = total_cont
    feats["contraction_rate_per_word"] = (
        total_cont / max(1, word_count)
    )

    # Sentence segmentation and burstiness
    # Split on sentence terminals followed by space or end-of-string
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentence_count = len(raw_sentences)
    feats["sentence_count"] = float(sentence_count)

    sent_word_counts = [len(re.findall(r"\b\w+\b", s)) for s in raw_sentences]
    if sent_word_counts:
        mean_sw = float(np.mean(sent_word_counts))
        std_sw = float(np.std(sent_word_counts))
        feats["sent_word_mean"] = mean_sw
        feats["sent_word_std"] = std_sw
        feats["sent_word_min"] = float(np.min(sent_word_counts))
        feats["sent_word_max"] = float(np.max(sent_word_counts))
        # Burstiness (Coefficient of Variation = std / mean)
        feats["sent_burstiness_cv"] = (std_sw / mean_sw) if mean_sw > 0 else 0.0
        # Sentence length skewness approximation
        if std_sw > 0 and len(sent_word_counts) > 2:
            sw_arr = np.array(sent_word_counts, dtype=np.float32)
            feats["sent_skewness"] = float(
                np.mean(((sw_arr - mean_sw) / std_sw) ** 3)
            )
        else:
            feats["sent_skewness"] = 0.0
    else:
        feats["sent_word_mean"] = 0.0
        feats["sent_word_std"] = 0.0
        feats["sent_word_min"] = 0.0
        feats["sent_word_max"] = 0.0
        feats["sent_burstiness_cv"] = 0.0
        feats["sent_skewness"] = 0.0

    # Paragraph structure
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    feats["paragraph_count"] = float(len(paragraphs))
    feats["avg_paragraph_len_words"] = (
        word_count / max(1, len(paragraphs))
    )

    # Lexical Diversity & Vocabulary Richness
    vocab = set(words_lower)
    vocab_size = len(vocab)
    feats["vocab_size"] = float(vocab_size)
    feats["type_token_ratio"] = vocab_size / max(1, word_count)
    feats["root_ttr"] = vocab_size / math.sqrt(max(1, word_count))
    feats["log_ttr"] = (
        (math.log(max(1, vocab_size)) / math.log(max(2, word_count)))
        if word_count > 1 else 0.0
    )

    # Hapax Legomena (words appearing once) and Dis Legomena (words appearing twice)
    if word_count > 0:
        word_counts = collections.Counter(words_lower)
        hapax_count = sum(1 for c in word_counts.values() if c == 1)
        dis_count = sum(1 for c in word_counts.values() if c == 2)
        feats["hapax_legomena_ratio"] = hapax_count / word_count
        feats["dis_legomena_ratio"] = dis_count / word_count
    else:
        feats["hapax_legomena_ratio"] = 0.0
        feats["dis_legomena_ratio"] = 0.0

    # Yule's K and Simpson's D
    feats["yules_k"] = compute_yules_k(words_lower)
    feats["simpsons_d"] = compute_simpsons_d(words_lower)

    # Information Entropy
    feats["char_entropy"] = compute_entropy(list(text))
    feats["word_entropy"] = compute_entropy(words_lower)

    # Function Words, POS, Connectives, and Markers (normalized per word)
    norm_word = 1.0 / max(1, word_count)
    feats["pronoun_1st_sg_freq"] = sum(1 for w in words_lower if w in PRONOUNS_1ST_SG) * norm_word
    feats["pronoun_1st_pl_freq"] = sum(1 for w in words_lower if w in PRONOUNS_1ST_PL) * norm_word
    feats["pronoun_2nd_freq"] = sum(1 for w in words_lower if w in PRONOUNS_2ND) * norm_word
    feats["pronoun_3rd_freq"] = sum(1 for w in words_lower if w in PRONOUNS_3RD) * norm_word
    feats["demonstratives_freq"] = sum(1 for w in words_lower if w in DEMONSTRATIVES) * norm_word
    feats["modal_verbs_freq"] = sum(1 for w in words_lower if w in MODAL_VERBS) * norm_word
    feats["prepositions_freq"] = sum(1 for w in words_lower if w in PREPOSITIONS) * norm_word
    feats["conjunctions_freq"] = sum(1 for w in words_lower if w in CONJUNCTIONS) * norm_word
    feats["discourse_markers_freq"] = sum(1 for w in words_lower if w in DISCOURSE_MARKERS) * norm_word
    feats["llm_cliches_freq"] = sum(1 for w in words_lower if w in LLM_CLICHES) * norm_word

    # Ratios between styles
    feats["first_to_third_pronoun_ratio"] = (
        (feats["pronoun_1st_sg_freq"] + feats["pronoun_1st_pl_freq"] + 1e-5) /
        (feats["pronoun_3rd_freq"] + 1e-5)
    )

    return feats


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for extracting calibrated stylometric features."""

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.feature_names_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None

    def fit(self, X: Union[pd.Series, List[str], np.ndarray], y=None):
        sample_feats = extract_single_text_features(X.iloc[0] if hasattr(X, "iloc") else X[0])
        self.feature_names_ = sorted(list(sample_feats.keys()))

        if self.scale_features:
            raw_mat = self._extract_matrix(X)
            self.scaler_ = RobustScaler(quantile_range=(5.0, 95.0))
            self.scaler_.fit(raw_mat)
        return self

    def _extract_matrix(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        if hasattr(X, "tolist"):
            texts = X.tolist()
        else:
            texts = list(X)

        rows = []
        for text in texts:
            f_dict = extract_single_text_features(text)
            row = [f_dict.get(col, 0.0) for col in self.feature_names_]
            rows.append(row)

        mat = np.array(rows, dtype=np.float32)
        # Guard against any unexpected NaN or Inf
        mat = np.nan_to_num(mat, nan=0.0, posinf=1e4, neginf=-1e4)
        return mat

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        raw_mat = self._extract_matrix(X)
        if self.scale_features and self.scaler_ is not None:
            scaled = self.scaler_.transform(raw_mat)
            return np.nan_to_num(scaled, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
        return raw_mat.astype(np.float32)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_


if __name__ == "__main__":
    from src.config import TRAIN_PATH

    print("Testing StylometricExtractor on train.csv...")
    df = pd.read_csv(TRAIN_PATH)
    sample_texts = df["text"].head(20)

    extractor = StylometricExtractor(scale_features=True)
    features = extractor.fit_transform(sample_texts)
    print(f"Extracted feature shape: {features.shape}")
    print(f"Total stylometric features: {len(extractor.feature_names_)}")
    print(f"Sample feature names: {extractor.feature_names_[:15]}")
    assert not np.isnan(features).any(), "NaN detected in features!"
    assert not np.isinf(features).any(), "Inf detected in features!"
    print("StylometricExtractor passed all validation checks successfully!")
