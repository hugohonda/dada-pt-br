import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .config.datasets import PHASE_WORKERS
from .config.logging import setup_logger
from .utils import (
    get_dataset_id,
    get_timestamp,
    load_json_file,
    save_json_file,
    validate_file_exists,
)

_LOGGER = setup_logger("analyzer", log_to_file=True, log_prefix="analysis")


class TranslationQualityAnalyzer:
    """Multi-dimensional translation quality analysis."""

    def __init__(self):
        _LOGGER.info("Initializing analyzers...")
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_pt = spacy.load("pt_core_news_sm")
        except OSError:
            _LOGGER.error(
                "Spacy models not found. Run: python -m spacy download en_core_web_sm pt_core_news_sm"
            )
            raise

        try:
            self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            _LOGGER.error(f"Failed to load sentence transformer: {e}")
            raise

        _LOGGER.info("Analyzers initialized successfully")

    def analyze_vocabulary(self, source: str, translation: str) -> dict:
        """Analyze vocabulary characteristics and semantic similarity."""
        # Tokenize
        doc_en = self.nlp_en(source.lower())
        doc_pt = self.nlp_pt(translation.lower())

        tokens_en = [t.text for t in doc_en if not t.is_punct and not t.is_space]
        tokens_pt = [t.text for t in doc_pt if not t.is_punct and not t.is_space]

        # Lexical diversity (Type-Token Ratio)
        ttr_source = len(set(tokens_en)) / len(tokens_en) if tokens_en else 0
        ttr_translation = len(set(tokens_pt)) / len(tokens_pt) if tokens_pt else 0

        # Length ratio
        length_ratio = len(tokens_pt) / len(tokens_en) if tokens_en else 1.0

        # Semantic similarity using multilingual embeddings
        emb_source = self.embedder.encode([source])
        emb_translation = self.embedder.encode([translation])
        semantic_similarity = float(
            cosine_similarity(emb_source, emb_translation)[0][0]
        )

        return {
            "ttr_source": round(ttr_source, 4),
            "ttr_translation": round(ttr_translation, 4),
            "ttr_ratio": round(ttr_translation / ttr_source, 4)
            if ttr_source > 0
            else 1.0,
            "length_ratio": round(length_ratio, 4),
            "token_count_source": len(tokens_en),
            "token_count_translation": len(tokens_pt),
            "semantic_similarity": round(semantic_similarity, 4),
        }

    def analyze_concordancy(self, source: str, translation: str) -> dict:
        """Analyze concordancy through overlap and context preservation."""
        doc_en = self.nlp_en(source.lower())
        doc_pt = self.nlp_pt(translation.lower())

        # Extract content words (nouns, verbs, adjectives, adverbs)
        content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        content_en = {t.lemma_ for t in doc_en if t.pos_ in content_pos}
        content_pt = {t.lemma_ for t in doc_pt if t.pos_ in content_pos}

        # N-gram analysis (bigrams)
        bigrams_en = self._extract_ngrams([t.text for t in doc_en], n=2)
        bigrams_pt = self._extract_ngrams([t.text for t in doc_pt], n=2)

        # POS structure similarity
        pos_seq_en = [t.pos_ for t in doc_en if t.pos_ != "PUNCT"]
        pos_seq_pt = [t.pos_ for t in doc_pt if t.pos_ != "PUNCT"]
        pos_similarity = self._sequence_similarity(pos_seq_en, pos_seq_pt)

        return {
            "content_word_count_source": len(content_en),
            "content_word_count_translation": len(content_pt),
            "bigram_count_source": len(bigrams_en),
            "bigram_count_translation": len(bigrams_pt),
            "pos_structure_similarity": round(pos_similarity, 4),
        }

    def analyze_ner_accuracy(self, source: str, translation: str) -> dict:
        """Detect entities for cultural adaptation review."""
        doc_en = self.nlp_en(source)
        doc_pt = self.nlp_pt(translation)

        # Extract only culturally-relevant entities
        entities_en = [
            (ent.text, ent.label_)
            for ent in doc_en.ents
            if self._is_culturally_relevant_entity(ent)
        ]
        entities_pt = [
            (ent.text, ent.label_)
            for ent in doc_pt.ents
            if self._is_culturally_relevant_entity(ent)
        ]

        if not entities_en:
            return {
                "has_entities": False,
                "entities_source": [],
                "entities_translation": [],
                "needs_review": [],
            }

        # Identify entities needing review
        needs_review = []

        for ent_text, ent_type in entities_en:
            found = self._find_entity_in_translation(ent_text, entities_pt)

            # Check if entity needs cultural adaptation
            needs_adaptation = self._needs_cultural_adaptation(
                ent_text, ent_type, found
            )
            if needs_adaptation:
                needs_review.append(
                    {
                        "text": ent_text,
                        "type": ent_type,
                        "found_in_translation": found is not None,
                        "translation_text": found[0] if found else None,
                        "reason": needs_adaptation,
                    }
                )

        return {
            "has_entities": True,
            "entity_count": len(entities_en),
            "entities_source": [{"text": e[0], "type": e[1]} for e in entities_en],
            "entities_translation": [{"text": e[0], "type": e[1]} for e in entities_pt],
            "needs_review": needs_review,
        }

    def _extract_ngrams(self, tokens: list, n: int = 2) -> set:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _sequence_similarity(self, seq1: list, seq2: list) -> float:
        """Calculate similarity between two sequences using longest common subsequence."""
        if not seq1 or not seq2:
            return 0.0

        # Simple LCS-based similarity
        lcs_length = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return lcs_length / max_len if max_len > 0 else 0.0

    def _lcs_length(self, seq1: list, seq2: list) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _is_culturally_relevant_entity(self, ent) -> bool:
        """Check if entity is culturally relevant (person, place, event, organization)."""
        # Focus on entities that typically need cultural adaptation
        culturally_relevant = {
            "PERSON",  # Names of people
            "GPE",  # Countries, cities, states
            "LOC",  # Non-GPE locations, mountain ranges, bodies of water
            "ORG",  # Companies, agencies, institutions
            "EVENT",  # Named events, battles, wars, sports events
            "PRODUCT",  # Objects, vehicles, foods (brand names)
            "WORK_OF_ART",  # Titles of books, songs, movies
            "FAC",  # Buildings, airports, highways, bridges
            "NORP",  # Nationalities, religious/political groups
        }
        return ent.label_ in culturally_relevant

    def _needs_cultural_adaptation(
        self, entity: str, entity_type: str, translation_match: tuple
    ) -> str:
        """Determine if entity needs cultural adaptation review."""
        # Missing entities always need review
        if not translation_match:
            return f"Entity '{entity}' missing in translation"

        trans_text = translation_match[0]

        # If entity was just transliterated, might need cultural adaptation
        if entity.lower() == trans_text.lower():
            if entity_type in ["GPE", "ORG", "PRODUCT", "EVENT"]:
                return "Transliterated without adaptation (consider localization)"

        # PERSON names: check if they're English-specific names that might confuse
        if entity_type == "PERSON":
            # If identical, might be okay (international names) or might need context
            if entity.lower() == trans_text.lower():
                return "Name transliterated (verify if culturally appropriate)"

        return None  # No review needed

    def _find_entity_in_translation(self, entity: str, entities_pt: list) -> tuple:
        """Find entity in translation with fuzzy matching."""
        entity_lower = entity.lower()

        # Exact match
        for ent_pt, type_pt in entities_pt:
            if entity_lower == ent_pt.lower():
                return (ent_pt, type_pt)

        # Partial match (entity contained in translation entity or vice versa)
        for ent_pt, type_pt in entities_pt:
            if entity_lower in ent_pt.lower() or ent_pt.lower() in entity_lower:
                return (ent_pt, type_pt)

        # Similar length and first character match (for transliterations)
        for ent_pt, type_pt in entities_pt:
            if len(entity) > 2 and len(ent_pt) > 2:
                if entity[0].lower() == ent_pt[0].lower():
                    if abs(len(entity) - len(ent_pt)) <= 2:
                        return (ent_pt, type_pt)

        return None

    def analyze(
        self, source: str, translation: str, xcomet_score: float = None
    ) -> dict:
        """Run comprehensive analysis on a translation."""
        vocab_metrics = self.analyze_vocabulary(source, translation)
        concordancy_metrics = self.analyze_concordancy(source, translation)
        ner_metrics = self.analyze_ner_accuracy(source, translation)

        # Calculate composite quality score
        composite_score = self._calculate_composite_score(
            xcomet_score, vocab_metrics, ner_metrics
        )

        return {
            "vocabulary": vocab_metrics,
            "concordancy": concordancy_metrics,
            "ner_accuracy": ner_metrics,
            "xcomet_score": xcomet_score,
            "composite_score": round(composite_score, 4),
        }

    def _calculate_composite_score(
        self, xcomet: float, vocab: dict, ner: dict
    ) -> float:
        """Calculate simplified quality score."""
        # Simplified: focus on semantic similarity and XCOMET
        xcomet_score = xcomet if xcomet is not None else vocab["semantic_similarity"]
        semantic_score = vocab["semantic_similarity"]

        # Entity penalty: reduce score if entities need review
        entity_penalty = 0.0
        if ner.get("has_entities") and ner.get("needs_review"):
            # Slight penalty for each entity needing review
            entity_penalty = min(len(ner["needs_review"]) * 0.05, 0.2)

        # Length ratio check
        length_ratio = vocab["length_ratio"]
        length_penalty = 0.0
        if length_ratio < 0.7 or length_ratio > 1.5:
            length_penalty = 0.1

        # Weighted average with penalties
        base_score = xcomet_score * 0.6 + semantic_score * 0.4
        final_score = max(0.0, base_score - entity_penalty - length_penalty)

        return final_score


def analyze_single_example(analyzer: TranslationQualityAnalyzer, example: dict) -> dict:
    """Analyze a single translation example."""
    source = example.get("source", "")
    translation = example.get("translation", "")
    xcomet_score = example.get("score")

    analysis = analyzer.analyze(source, translation, xcomet_score)

    return {
        "index": example.get("index", example.get("id", 0)),
        "id": example.get("id", example.get("index", 0)),
        "source": source,
        "translation": translation,
        "analysis": analysis,
    }


def analyze_single_parallel(args):
    """Analyze a single example - used for parallel processing."""
    analyzer, example, index = args
    start_time = time.time()

    try:
        result = analyze_single_example(analyzer, example)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "data": result,
            "processing_time": processing_time,
            "index": index,
            "error": None,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        _LOGGER.error(f"Error analyzing example {example.get('id', 'unknown')}: {e}")
        return {
            "success": False,
            "data": {
                "index": example.get("index", example.get("id", 0)),
                "id": example.get("id", example.get("index", 0)),
                "source": example.get("source", ""),
                "translation": example.get("translation", ""),
                "analysis": {"error": str(e)},
            },
            "processing_time": processing_time,
            "index": index,
            "error": str(e),
        }


def process_dataset(
    input_file: str,
    output_file: str,
    limit: int = None,
    max_workers: int = None,
):
    """Analyze translations in a dataset."""
    start_time = time.time()
    _LOGGER.info(f"Analyzing translations from: {input_file}")

    # Load data
    data, metadata = load_json_file(input_file)
    if not data:
        _LOGGER.error("No data found in input file")
        return

    # Apply limit if specified
    if limit:
        data = data[:limit]
        _LOGGER.info(f"Limited to {limit} examples")

    dataset_id = get_dataset_id(input_file)
    _LOGGER.info(f"Analyzing {len(data)} examples from dataset: {dataset_id}")

    # Initialize analyzer
    analyzer = TranslationQualityAnalyzer()

    # Set default workers if not provided
    if max_workers is None:
        max_workers = PHASE_WORKERS["analysis"]["default"]
    max_workers = min(max_workers, PHASE_WORKERS["analysis"]["max"])

    _LOGGER.info(f"Processing {len(data)} examples with {max_workers} workers...")

    # Prepare task arguments for parallel processing
    task_args = [(analyzer, example, i) for i, example in enumerate(data)]

    # Process examples in parallel
    analyzed_data = [None] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(analyze_single_parallel, args): args[2]
            for args in task_args
        }

        for future in tqdm(
            as_completed(future_to_index),
            desc="Analyzing translations",
            total=len(task_args),
        ):
            result = future.result()
            index = result["index"]
            analyzed_data[index] = result["data"]

            if result["success"]:
                _LOGGER.debug(f"Analysis completed in {result['processing_time']:.2f}s")
            else:
                _LOGGER.error(
                    f"Analysis failed for example {result['index']}: {result['error']}"
                )

    # Calculate summary statistics focused on review needs
    composite_scores = [
        ex["analysis"].get("composite_score", 0)
        for ex in analyzed_data
        if "composite_score" in ex["analysis"]
    ]

    semantic_scores = [
        ex["analysis"]["vocabulary"]["semantic_similarity"]
        for ex in analyzed_data
        if "vocabulary" in ex["analysis"]
    ]

    # Count examples with entities needing review
    examples_with_entities = sum(
        1
        for ex in analyzed_data
        if ex["analysis"].get("ner_accuracy", {}).get("has_entities", False)
    )

    total_entities_needing_review = sum(
        len(ex["analysis"]["ner_accuracy"]["needs_review"])
        for ex in analyzed_data
        if ex["analysis"].get("ner_accuracy", {}).get("needs_review")
    )

    examples_needing_review = sum(
        1
        for ex in analyzed_data
        if ex["analysis"].get("ner_accuracy", {}).get("needs_review")
    )

    summary_stats = {
        "total_examples": len(analyzed_data),
        "composite_score_mean": round(float(np.mean(composite_scores)), 4)
        if composite_scores
        else 0,
        "semantic_similarity_mean": round(float(np.mean(semantic_scores)), 4)
        if semantic_scores
        else 0,
        "examples_with_entities": examples_with_entities,
        "examples_needing_review": examples_needing_review,
        "total_entities_needing_review": total_entities_needing_review,
    }

    # Save results
    total_time = time.time() - start_time

    result_metadata = {
        "operation": "analysis",
        "pipeline_id": get_timestamp(),
        "dataset_id": dataset_id,
        "total_examples": len(analyzed_data),
        "timestamp": get_timestamp(),
        "processing_seconds": round(total_time, 2),
        "summary_statistics": summary_stats,
    }

    final_data = {"metadata": result_metadata, "data": analyzed_data}
    save_json_file(final_data, output_file)

    # Log summary
    _LOGGER.info("Analysis completed:")
    _LOGGER.info(f"  Total examples: {len(analyzed_data)}")
    _LOGGER.info(f"  Composite score (mean): {summary_stats['composite_score_mean']}")
    _LOGGER.info(
        f"  Semantic similarity (mean): {summary_stats['semantic_similarity_mean']}"
    )
    _LOGGER.info(f"  Examples with entities: {summary_stats['examples_with_entities']}")
    _LOGGER.info(
        f"  Examples needing review: {summary_stats['examples_needing_review']}"
    )
    _LOGGER.info(
        f"  Total entities flagged: {summary_stats['total_entities_needing_review']}"
    )
    _LOGGER.info(f"  Results saved to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Translation Quality Analyzer")
    parser.add_argument(
        "input_file", help="Input JSON file (translated/evaluated/merged)"
    )
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--limit", "-l", type=int, help="Limit examples (default: all)")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=PHASE_WORKERS["analysis"]["default"],
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        output_file = (
            input_path.parent / f"{input_path.stem}_analyzed{input_path.suffix}"
        )
    else:
        output_file = args.output

    process_dataset(args.input_file, str(output_file), args.limit, args.workers)


if __name__ == "__main__":
    main()
