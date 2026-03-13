import io
import pypdf
import streamlit as st

from agent import initialize_app
from langchain_groq.chat_models import ChatGroq
from docx import Document
from PIL import Image
import pytesseract

st.set_page_config(page_title="Testcase Generation Agent", layout="wide")
st.title("Testcase Generation Agent")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model=st.session_state.selected_model,
        temperature=0.0
    )

if "include_edge_cases" not in st.session_state:
    st.session_state.include_edge_cases = True

if "enhancement_level" not in st.session_state:
    st.session_state.enhancement_level = 3

if "use_industry_standards" not in st.session_state:
    st.session_state.use_industry_standards = True

with st.sidebar:
    st.header("Configuration")

    uploaded_file = st.file_uploader(
        "Upload Requirements Document",
        type=["txt", "pdf", "docx", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]

    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=model_options.index(st.session_state.selected_model)
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.llm = ChatGroq(
            model=selected_model,
            temperature=0.0
        )

    st.subheader("Test Generation Settings")

    st.session_state.include_edge_cases = st.checkbox(
        "Include Edge Cases",
        value=st.session_state.include_edge_cases
    )

    st.session_state.enhancement_level = st.slider(
        "Test Case Detail Level",
        min_value=1,
        max_value=5,
        value=st.session_state.enhancement_level
    )

    st.session_state.use_industry_standards = st.checkbox(
        "Apply Industry Best Practices",
        value=st.session_state.use_industry_standards
    )

    st.divider()

    st.subheader("Select Test Case Category")

    test_category = st.radio(
        "Choose Category",
        ["Gherkin Test Cases", "Selenium Test Cases"],
        index=0
    )

    if st.button("Reset"):
        st.rerun()

app = initialize_app(model_name=st.session_state.selected_model)

requirements_docs_content = ""

if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
    file = st.session_state.uploaded_file

    if file.type == "text/plain":
        requirements_docs_content = file.getvalue().decode("utf-8")

    elif file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(file.getvalue()))
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                requirements_docs_content += text + "\n"

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(file.getvalue()))
        for para in doc.paragraphs:
            requirements_docs_content += para.text + "\n"

    elif file.type in ["image/png", "image/jpeg"]:
        image = Image.open(io.BytesIO(file.getvalue()))
        requirements_docs_content = pytesseract.image_to_string(image)

st.subheader("Enter your Feature")

feature_input = st.text_input(
    "Example: Login Module, Checkout System, Payment Gateway"
)

generate_button = st.button("Generate Test Cases")

if generate_button and feature_input:

    with st.spinner("Generating test cases..."):

        if test_category == "Gherkin Test Cases":
            selected_format = "Gherkin"
        else:
            selected_format = "Selenium"

        inputs = {
            "user_request": f"Generate {selected_format} test cases for {feature_input}",
            "requirements_docs_content": requirements_docs_content,
            "requirements_docs_summary": "",
            "industry_best_practices": "",
            "test_format": selected_format,
            "answer": ""
        }

        final_output = ""
        summary_output = ""

        for output in app.stream(inputs):
            for node_name, state in output.items():

                if "answer" in state:
                    final_output = state["answer"]

                if "summary" in state:
                    summary_output = state["summary"]

        st.divider()
        st.subheader(test_category)

        if summary_output:
            st.markdown("Summary")
            st.write(summary_output)

        st.markdown("Test Cases")
        st.text(final_output)
