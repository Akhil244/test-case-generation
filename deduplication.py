import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplication_node_function(state: dict) -> dict:
    """
    Removes highly similar test cases using TF-IDF cosine similarity.
    Works for Gherkin and Selenium.
    """

    answer = state.get("answer", "")
    test_format = state.get("test_format", "Gherkin")

    if not answer.strip():
        return state

    if test_format == "Gherkin":
        test_cases = parse_gherkin(answer)
    else:
        test_cases = parse_code(answer)

    if len(test_cases) < 2:
        return state

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(
        [tc["text"] for tc in test_cases]
    )

    similarity_matrix = cosine_similarity(tfidf_matrix)

    duplicates = set()

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.85:
                duplicates.add(j)

    unique_tests = [
        tc for idx, tc in enumerate(test_cases)
        if idx not in duplicates
    ]

    reconstructed = []
    for tc in unique_tests:
        reconstructed.append(tc["raw"])
        reconstructed.append("")

    state["answer"] = "\n".join(reconstructed).strip()
    return state

def parse_gherkin(content: str) -> List[Dict]:
    scenarios = []
    blocks = content.split("Scenario:")

    for block in blocks[1:]:
        full = "Scenario:" + block
        scenarios.append({
            "raw": full.strip(),
            "text": full.lower()
        })

    return scenarios

def parse_code(content: str) -> List[Dict]:
    test_cases = []
    pattern = re.compile(r"(def test_)")

    blocks = pattern.split(content)

    for i in range(1, len(blocks), 2):
        test_name = blocks[i]
        test_body = blocks[i+1] if i+1 < len(blocks) else ""
        full = test_name + test_body

        test_cases.append({
            "raw": full.strip(),
            "text": full.lower()
        })

    return test_cases