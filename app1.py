from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def deduplication_node_function(state: GraphState) -> GraphState:
    """Node to remove redundant test cases using text similarity analysis"""
    answer = state.get("answer", "")
    test_format = state.get("test_format", "Gherkin")
    
    if not answer:
        return state

    # Parse test cases based on format
    if test_format == "Gherkin":
        test_cases = parse_gherkin(answer)
    else:  # Python or Java
        test_cases = parse_code(answer)
    
    if len(test_cases) < 2:
        return state  # No duplicates possible

    # Calculate similarity matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([tc["text"] for tc in test_cases])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Identify duplicates
    duplicates = set()
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.85:  # Adjust threshold here
                duplicates.add(j)

    # Preserve order and remove duplicates
    unique_test_cases = [tc for idx, tc in enumerate(test_cases) 
                        if idx not in duplicates]

    # Rebuild the answer with unique tests
    reconstructed = []
    for tc in unique_test_cases:
        reconstructed.append(tc["raw"])
        reconstructed.append("")  # Add spacing between tests
    
    state["answer"] = "
".join(reconstructed).strip()
    return state

def parse_gherkin(content: str) -> list:
    """Parse Gherkin scenarios into individual test cases"""
    scenarios = []
    current_scenario = []
    in_scenario = False
    
    for line in content.split("
"):
        if line.startswith("Scenario:") or line.startswith("@") and not current_scenario:
            if current_scenario:
                scenarios.append({
                    "raw": "
".join(current_scenario).strip(),
                    "text": " ".join(current_scenario).lower()
                })
            current_scenario = [line]
            in_scenario = True
        elif in_scenario:
            if line.strip() == "" and current_scenario:
                scenarios.append({
                    "raw": "
".join(current_scenario).strip(),
                    "text": " ".join(current_scenario).lower()
                })
                current_scenario = []
                in_scenario = False
            else:
                current_scenario.append(line)
    
    if current_scenario:
        scenarios.append({
            "raw": "
".join(current_scenario).strip(),
            "text": " ".join(current_scenario).lower()
        })
    return scenarios

def parse_code(content: str) -> list:
    """Parse Python/Java test methods into individual test cases"""
    test_cases = []
    current_test = []
    in_test = False
    
    # Look for test method starts
    test_pattern = re.compile(r"(def test_|@Test|public void test)")
    
    for line in content.split("
"):
        if test_pattern.search(line):
            if current_test:
                test_cases.append(create_test_case(current_test))
            current_test = [line]
            in_test = True
        elif in_test:
            if line.strip() == "" and current_test:
                test_cases.append(create_test_case(current_test))
                current_test = []
                in_test = False
            else:
                current_test.append(line)
    
    if current_test:
        test_cases.append(create_test_case(current_test))
    
    return test_cases

def create_test_case(lines: list) -> dict:
    """Helper to create test case structure"""
    raw_content = "
".join(lines).strip()
    text_content = " ".join(lines).lower()
    return {"raw": raw_content, "text": text_content}

# Update the workflow builder
def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("summary_node", generate_summary_node_function)
    workflow.add_node("best_practices_node", search_best_practices_node_function)
    workflow.add_node("testcase_node", generate_testcases_node_function)
    workflow.add_node("deduplication_node", deduplication_node_function)  # New node

    workflow.set_entry_point("summary_node")
    workflow.add_edge("summary_node", "best_practices_node")
    workflow.add_edge("best_practices_node", "testcase_node")
    workflow.add_edge("testcase_node", "deduplication_node")  # Add deduplication step
    workflow.add_edge("deduplication_node", END)  # Connect to end

    return workflow