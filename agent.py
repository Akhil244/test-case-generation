from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
from dotenv import load_dotenv
import streamlit as st
import os
from deduplication import deduplication_node_function

load_dotenv()

try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except:
    tavily_client = None

class GraphState(TypedDict):
    user_request: str
    requirements_docs_content: str
    requirements_docs_summary: str
    industry_best_practices: str
    test_format: str
    answer: str

def generate_summary_node_function(state: GraphState) -> GraphState:
    requirements = state.get("requirements_docs_content", "")

    if not requirements.strip():
        state["requirements_docs_summary"] = "No requirements document provided."
        return state
    prompt = f"""Write a concise 3-4 line professional summary of these requirements.

Example 1:
Input: User authentication with email/password, social login, account lockout after 5 failures, password reset via email, remember me 30 days.
Output: The authentication system provides email/password and social login options with security features including account lockout after 5 failed attempts and email-based password recovery. Session management includes 30-day remember me functionality.

Example 2:
Input: Product catalog with infinite scroll, filters by category/price/brand, real-time stock indicators, recently viewed products, personalized recommendations.
Output: The product discovery system features infinite scroll browsing with multi-dimensional filters and real-time stock visibility. User browsing history powers recently viewed carousels and personalized recommendations.

Now summarize:
{requirements}

Summary:"""

    response = st.session_state.llm.invoke(prompt)
    state["requirements_docs_summary"] = response.content

    return state

def search_best_practices_node_function(state: GraphState) -> GraphState:
    if tavily_client is None:
        state["industry_best_practices"] = "1. Use Page Object Model\n2. Use explicit waits\n3. Tag tests by priority\n4. Maintain test data separately\n5. Keep tests independent"
        return state
        
    try:
        results = tavily_client.search(
            query="ISTQB test case design best practices",
            search_depth="basic",  
            max_results=2  
        )

        collected_text = ""
        for r in results.get("results", []):
            collected_text += r.get("content", "")[:300] + "\n\n"

        prompt = f"""Extract 3 test case design best practices from this text (numbered 1-3 only):

{collected_text}

Best practices:"""

        response = st.session_state.llm.invoke(prompt)
        extracted = response.content.strip()
        
        if extracted:
            state["industry_best_practices"] = extracted
        else:
            state["industry_best_practices"] = "1. Use Page Object Model\n2. Use explicit waits\n3. Tag tests by priority"

    except Exception as e:
        state["industry_best_practices"] = "1. Use Page Object Model\n2. Use explicit waits\n3. Tag tests by priority"

    return state

def generate_gherkin_testcases(state: GraphState) -> str:
    """Generate pure Gherkin feature files - MINIMAL PROMPT VERSION"""
    
    requirements = state.get("requirements_docs_content", "")
    context = state.get("user_request", "")
    
    prompt = f"""Write Gherkin test cases for these requirements.
Use Feature, Scenario, Given, When, Then.
Format like this example:

Feature: Login
  Scenario: Successful login
    Given user on login page
    When user enters valid credentials
    Then user is redirected to dashboard

Requirements:
{requirements}

Context:
{context}

Generate 5 test cases: 2 positive, 2 negative, 1 edge case.
Use @P0 @P1 @P2 tags.
Output ONLY Gherkin, no explanations:"""
    
    response = st.session_state.llm.invoke(prompt)
    return response.content

def generate_python_selenium_testcases(state: GraphState) -> str:
    """Generate Python Selenium test cases - MINIMAL PROMPT VERSION"""
    
    requirements = state.get("requirements_docs_content", "")
    context = state.get("user_request", "")
    prompt = f"""Write Python Selenium test cases using Page Object Model.

Example test:
@pytest.mark.tcid("LOG-POS-001")
def test_login(driver):
    login_page = LoginPage(driver)
    login_page.login("user@ex.com", "pass")
    assert DashboardPage(driver).is_displayed()

Requirements:
{requirements}

Context:
{context}

Generate 5 test methods with proper test IDs, assertions, and docstrings.
Use @pytest.mark.tcid and @pytest.mark.priority.
Output ONLY Python code:"""
    
    response = st.session_state.llm.invoke(prompt)
    return response.content

def generate_java_selenium_testcases(state: GraphState) -> str:
    """Generate Java Selenium test cases - MINIMAL PROMPT VERSION"""
    
    requirements = state.get("requirements_docs_content", "")
    context = state.get("user_request", "")

    prompt = f"""Write Java Selenium test cases with Page Object Model.

Example test:
@Test
@DisplayName("LOG-POS-001: Test login")
public void testLogin() {{
    LoginPage loginPage = new LoginPage(driver);
    DashboardPage dashboard = loginPage.login("user@ex.com", "pass");
    assertTrue(dashboard.isDisplayed());
}}

Requirements:
{requirements}

Context:
{context}

Generate 5 test methods with proper test IDs, assertions, and JUnit annotations.
Use @Tag for priorities.
Output ONLY Java code:"""
    
    response = st.session_state.llm.invoke(prompt)
    return response.content

def generate_testcases_node_function(state: GraphState) -> GraphState:
    test_format = state.get("test_format", "Gherkin")
    
    if test_format == "Gherkin":
        state["answer"] = generate_gherkin_testcases(state)
    elif test_format == "Python Selenium":
        state["answer"] = generate_python_selenium_testcases(state)
    else:  
        state["answer"] = generate_java_selenium_testcases(state)
    
    return state

def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("summary_node", generate_summary_node_function)
    workflow.add_node("best_practices_node", search_best_practices_node_function)
    workflow.add_node("testcase_node", generate_testcases_node_function)
    workflow.add_node("deduplication_node", deduplication_node_function)

    workflow.set_entry_point("summary_node")

    workflow.add_edge("summary_node", "best_practices_node")
    workflow.add_edge("best_practices_node", "testcase_node")
    workflow.add_edge("testcase_node", "deduplication_node")
    workflow.add_edge("deduplication_node", END)

    return workflow

def initialize_app(model_name: str):
    """Initialize the application with the specified Groq model"""
    if "llm" not in st.session_state:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found in environment variables")
            return None
        st.session_state.llm = ChatGroq(
            model=model_name, 
            temperature=0.0,
            api_key=api_key
        )
    return build_workflow().compile()