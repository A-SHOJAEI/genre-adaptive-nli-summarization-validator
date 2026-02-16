"""Text preprocessing utilities for summarization and NLI tasks."""

import re
import logging
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Text preprocessing utilities for summarization validation tasks."""

    def __init__(self, tokenizer_name: str = "microsoft/deberta-v3-base"):
        """Initialize text preprocessor.

        Args:
            tokenizer_name: Name of the HuggingFace tokenizer to use.
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing TextPreprocessor with tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        try:
            self.stop_words = set(stopwords.words('english'))
            self.logger.debug("English stopwords loaded successfully")
        except LookupError:
            self.logger.warning("NLTK stopwords not found, downloading...")
            import nltk
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)

        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)

        return text.strip()

    def truncate_text(self, text: str, max_length: int = 512) -> str:
        """Truncate text to fit within token limits.

        Args:
            text: Text to truncate.
            max_length: Maximum number of tokens.

        Returns:
            Truncated text.
        """
        if not text:
            return ""

        # Tokenize and check length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_length:
            return text

        # Truncate by sentences to preserve meaning
        sentences = sent_tokenize(text)
        truncated_text = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            if current_tokens + sentence_tokens <= max_length:
                truncated_text += sentence + " "
                current_tokens += sentence_tokens
            else:
                break

        # If no complete sentences fit, truncate by tokens
        if not truncated_text:
            truncated_tokens = tokens[:max_length]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        return truncated_text.strip()

    def extract_genre_indicators(self, text: str) -> Dict[str, float]:
        """Extract genre-specific indicators from text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary of genre indicators with confidence scores.
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words)

        if word_count == 0:
            return {}

        # Genre-specific lexical patterns
        indicators = {
            'fiction': self._count_fiction_indicators(text_lower, words, word_count),
            'news': self._count_news_indicators(text_lower, words, word_count),
            'academic': self._count_academic_indicators(text_lower, words, word_count),
            'government': self._count_government_indicators(text_lower, words, word_count),
            'telephone': self._count_telephone_indicators(text_lower, words, word_count),
        }

        return indicators

    def _count_fiction_indicators(self, text: str, words: List[str], word_count: int) -> float:
        """Count fiction-specific indicators."""
        fiction_words = {'character', 'story', 'novel', 'chapter', 'plot', 'fiction',
                        'narrative', 'protagonist', 'dialogue', 'scene'}
        fiction_patterns = ['he said', 'she said', 'thought to', 'felt like']

        word_score = len([w for w in words if w in fiction_words]) / word_count
        pattern_score = sum(1 for pattern in fiction_patterns if pattern in text) / len(fiction_patterns)

        return (word_score + pattern_score) / 2

    def _count_news_indicators(self, text: str, words: List[str], word_count: int) -> float:
        """Count news-specific indicators."""
        news_words = {'reported', 'according', 'official', 'statement', 'announced',
                     'sources', 'breaking', 'update', 'press', 'journalist'}
        date_patterns = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', text)

        word_score = len([w for w in words if w in news_words]) / word_count
        date_score = min(len(date_patterns) / 5, 1.0)  # Normalize to max 1.0

        return (word_score + date_score) / 2

    def _count_academic_indicators(self, text: str, words: List[str], word_count: int) -> float:
        """Count academic-specific indicators."""
        academic_words = {'research', 'study', 'analysis', 'methodology', 'hypothesis',
                         'results', 'conclusion', 'literature', 'theory', 'experiment'}
        citation_patterns = re.findall(r'\(\d{4}\)|\[\d+\]|et al\.', text)

        word_score = len([w for w in words if w in academic_words]) / word_count
        citation_score = min(len(citation_patterns) / 10, 1.0)

        return (word_score + citation_score) / 2

    def _count_government_indicators(self, text: str, words: List[str], word_count: int) -> float:
        """Count government-specific indicators."""
        government_words = {'regulation', 'policy', 'law', 'act', 'section', 'statute',
                           'federal', 'department', 'agency', 'committee'}
        legal_patterns = ['shall', 'pursuant to', 'in accordance with']

        word_score = len([w for w in words if w in government_words]) / word_count
        pattern_score = sum(1 for pattern in legal_patterns if pattern in text) / len(legal_patterns)

        return (word_score + pattern_score) / 2

    def _count_telephone_indicators(self, text: str, words: List[str], word_count: int) -> float:
        """Count telephone conversation indicators."""
        telephone_words = {'hello', 'yeah', 'okay', 'right', 'um', 'uh', 'well',
                          'actually', 'basically', 'like', 'you know'}
        speaker_patterns = re.findall(r'[A-Z][a-z]+:', text)

        word_score = len([w for w in words if w in telephone_words]) / word_count
        speaker_score = min(len(speaker_patterns) / 5, 1.0)

        return (word_score + speaker_score) / 2

    def create_nli_pairs(self, document: str, summary: str) -> List[Tuple[str, str, str]]:
        """Create premise-hypothesis pairs for NLI classification.

        Args:
            document: Source document (premise).
            summary: Summary to validate (hypothesis).

        Returns:
            List of (premise, hypothesis, label) tuples for training.
        """
        pairs = []

        # Clean and truncate texts
        clean_document = self.clean_text(document)
        clean_summary = self.clean_text(summary)

        if not clean_document or not clean_summary:
            return pairs

        # Split document into sentences for granular entailment checking
        doc_sentences = sent_tokenize(clean_document)
        summary_sentences = sent_tokenize(clean_summary)

        # Create entailment pairs (document -> summary sentences)
        for summary_sent in summary_sentences:
            if len(summary_sent.split()) >= 5:  # Only meaningful sentences
                pairs.append((clean_document, summary_sent, "entailment"))

        # Create neutral pairs (unrelated sentences)
        if len(doc_sentences) > 1 and len(summary_sentences) > 1:
            # Use document sentences as neutral examples
            for i, doc_sent in enumerate(doc_sentences[:3]):  # Limit to avoid explosion
                if len(doc_sent.split()) >= 5:
                    pairs.append((clean_summary, doc_sent, "neutral"))

        return pairs

    def validate_summary_quality(self, document: str, summary: str) -> Dict[str, float]:
        """Validate summary quality using various metrics.

        Args:
            document: Source document.
            summary: Summary to validate.

        Returns:
            Dictionary of quality metrics.
        """
        metrics = {}

        # Length ratio
        doc_words = len(word_tokenize(document))
        summary_words = len(word_tokenize(summary))
        metrics['length_ratio'] = summary_words / max(doc_words, 1)

        # Coverage (shared vocabulary)
        doc_vocab = set(word_tokenize(document.lower())) - self.stop_words
        summary_vocab = set(word_tokenize(summary.lower())) - self.stop_words
        metrics['vocabulary_coverage'] = len(summary_vocab & doc_vocab) / max(len(summary_vocab), 1)

        # Abstractiveness (novel n-grams)
        doc_bigrams = set(nltk.bigrams(word_tokenize(document.lower())))
        summary_bigrams = set(nltk.bigrams(word_tokenize(summary.lower())))
        novel_bigrams = summary_bigrams - doc_bigrams
        metrics['abstractiveness'] = len(novel_bigrams) / max(len(summary_bigrams), 1)

        # Sentence structure similarity
        doc_sentences = sent_tokenize(document)
        summary_sentences = sent_tokenize(summary)
        avg_doc_sent_len = sum(len(word_tokenize(s)) for s in doc_sentences) / max(len(doc_sentences), 1)
        avg_summary_sent_len = sum(len(word_tokenize(s)) for s in summary_sentences) / max(len(summary_sentences), 1)
        metrics['sentence_length_ratio'] = avg_summary_sent_len / max(avg_doc_sent_len, 1)

        return metrics

    def prepare_model_inputs(
        self,
        premise: str,
        hypothesis: str,
        max_length: int = 512
    ) -> Dict[str, List[int]]:
        """Prepare model inputs for NLI classification.

        Args:
            premise: Premise text (document).
            hypothesis: Hypothesis text (summary).
            max_length: Maximum sequence length.

        Returns:
            Tokenized inputs dictionary.
        """
        # Clean inputs
        premise = self.clean_text(premise)
        hypothesis = self.clean_text(hypothesis)

        # Tokenize with proper formatting for NLI
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )

        return inputs