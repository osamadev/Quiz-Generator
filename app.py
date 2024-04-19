from openai import OpenAI
from vectara import Indexing, Searching
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Vectara
import json
import re
import os
import streamlit as st
import random
from dotenv import load_dotenv

# Load environment variables
def load_env():
    os.environ["AUTH_URL"] = st.secrets["AUTH_URL"]
    os.environ["APP_CLIENT_ID"] = st.secrets["APP_CLIENT_ID"]
    os.environ["APP_CLIENT_SECRET"] = st.secrets["APP_CLIENT_SECRET"]
    os.environ["CUSTOMER_ID"] = st.secrets["CUSTOMER_ID"]
    os.environ["CORPUS_ID"] = st.secrets["CORPUS_ID"]
    os.environ["IDX_ADDRESS"] = st.secrets["IDX_ADDRESS"]
    os.environ["API_KEY"] = st.secrets["API_KEY"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

load_env()

# Initialize external clients
vectara = Vectara()

# Create instances of Indexing and Searching classes
indexer = Indexing()
searcher = Searching()

def generate_question_and_options(document):
    client = OpenAI()
    
    system_prompt = """
    Your task is to analyze the provided text and extract essential elements to create a multiple-choice question with one correct answer and three incorrect options (distractors). Additionally, you are to provide a support explanation that justifies why the correct answer is right.

    Please format the output as a JSON object that includes:
    - A 'question' key with a clear, well-formed question derived from the text.
    - Three 'distractor' keys labeled 'distractor1', 'distractor2', and 'distractor3', each containing a plausible but incorrect answer.
    - A 'correct_answer' key containing the accurate answer to the question.
    - A 'support' key containing an explanation or reasoning that supports the correctness of the provided answer.

    Here is a sample json as an example for the generated output, example: 
    {'question': 'What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?', 
     'distractor3': 'first option', 
     'distractor1': 'second option',
     'distractor2': 'third option',
     'correct_answer': 'the correct answer',
     'support': 'Without Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere.'} 

    Ensure the information is accurate, relevant, and directly derived from the provided text. Avoid introducing external facts not supported by the text. This task requires precision and attention to detail in reading comprehension and data presentation.
    """

    user_prompt = f"""Please read the following text and extract information to form a multiple choice question with one correct answer and three distractors. Also, provide a support explanation for the correct answer.
    Format the output as a JSON object with keys for 'question', 'distractor1', 'distractor2', 'distractor3', 'correct_answer', and 'support'. 
    
    Context:
    {document}
    
    Answer:
    """

    # try:
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    
    # Parse the response to get the JSON object
    result = json.loads(response.choices[0].message.content)

    # Extract the question, options, correct answer and context
    question = result['question']
    correct_answer = result['correct_answer']
    distractors = [result['distractor1'], result['distractor2'], result['distractor3']]
    context = result['support']

    # Combine all options and shuffle
    options = [correct_answer] + distractors
    random.shuffle(options)
    
    # Map options to 'A', 'B', 'C', 'D'
    options_mapping = {chr(65 + i): option for i, option in enumerate(options)}
    # Find the key for the correct answer in the shuffled options
    answer_key = next(key for key, value in options_mapping.items() if value == correct_answer)

    return {
        "question": question,
        "options": options_mapping,
        "answer": answer_key,
        "context": context
    }

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return {}

def generate_questions_data(context, questionsNo):
    client = OpenAI()
    
    system_prompt = """
    Your task is to analyze the provided text and extract essential elements to create multiple-choice questions with one correct answer and three incorrect options (distractors). Additionally, you are to provide a support explanation that justifies why the correct answer is right.

    Please format the output JSON object which has a parent key 'questions-data' which has array of JSON objects where each child JSON object includes:
    - A 'question' key with a clear, well-formed question derived from the text.
    - Three 'distractor' keys labeled 'distractor1', 'distractor2', and 'distractor3', each containing a plausible but incorrect answer.
    - A 'correct_answer' key containing the accurate answer to the question.
    - A 'support' key containing an explanation or reasoning that supports the correctness of the provided answer.

    Here is a sample json as an example for the generated output, example: 
    {'questions-data': [{'question': 'What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?', 
     'distractor3': 'first option', 
     'distractor1': 'second option',
     'distractor2': 'third option',
     'correct_answer': 'the correct answer',
     'support': 'Without Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere.'},
     {"question": "What is the least dangerous radioactive decay?",
                                "distractor3": "zeta decay",
                                "distractor1": "beta decay",
                                "distractor2": "gamma decay",
                                "correct_answer": "alpha decay",
                                "support": "All radioactive decay is dangerous to living things"}]}

    Ensure the information is accurate, relevant, and directly derived from the provided text. Avoid introducing external facts not supported by the text. This task requires precision and attention to detail in reading comprehension and data presentation.
    Ensure the information is derived directly from the provided text and formatted accurately as a collection of JSON objects.
    """
    user_prompt = f"""
    Based on the following context, extract information to form {questionsNo} distinct multiple choice questions, each with a correct answer and three distractors. Also, provide a supporting explanation for each correct answer.
    Ensure that the maximum number of the words in the final response does not exceed 8000 words.

    Context:
    {context}
    
    Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    questions_data = load_json(response.choices[0].message.content)

    processed_questions = []
    for question_data in questions_data['questions-data']:
        question = question_data['question']
        correct_answer = question_data['correct_answer']
        distractors = [question_data.get('distractor1'), question_data.get('distractor2'), question_data.get('distractor3')]
        support = question_data['support']

        # Shuffle options
        options = [correct_answer] + distractors
        random.shuffle(options)

        # Map shuffled options to 'A', 'B', 'C', 'D'
        options_mapping = {chr(65 + i): option for i, option in enumerate(options)}
        # Determine the key for the correct answer
        answer_key = next(key for key, value in options_mapping.items() if value == correct_answer)

        processed_questions.append({
            "question": question,
            "options": options_mapping,
            "answer": answer_key,
            "context": support
        })

    return processed_questions

def parse_document_to_json(document):
    pattern = re.compile(
        r"Question:\s*(.+?)\s+" +
        r"A:\s*(.+?)\s+" + 
        r"B:\s*(.+?)\s+" +
        r"C:\s*(.+?)\s+" +
        r"D:\s*(.+?)\s+" +
        r"Answer:\s*(\w+)\s+" +
        r"Context:\s*(.+)", re.DOTALL
    )
    match = pattern.search(document)
    if match:
        question, a, b, c, d, answer, context = match.groups()
        options = {'A': a.strip(), 'B': b.strip(), 'C': c.strip(), 'D': d.strip()}
        
        # Randomly shuffle the options
        option_keys = list(options.keys())
        random_values = list(options.values())
        random.shuffle(random_values)
        shuffled_options = dict(zip(option_keys, random_values))

        # Find the new key for the correct answer after shuffling
        answer_key = next((k for k, v in shuffled_options.items() if v == answer.strip()), None)
        if answer_key is None:
            raise ValueError("Answer key not found in options after shuffling.")
        
        return {
            "question": question.strip(),
            "options": shuffled_options,
            "answer": answer_key,
            "context": context.strip()
        }
    else:
        raise ValueError("The document format does not match the expected pattern. Document: " + document)

def initialize_session_state():
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'show_questions' not in st.session_state:
        st.session_state.show_questions = True  # Initially show questions

initialize_session_state()

def display_question(question, index):
    options = question['options']
    option_keys = list(options.keys())

    st.write(f"Question {index + 1}: {question['question']}")
    chosen_option = st.radio(
        "Choose your answer:",
        options=[f"{key}: {options[key]}" for key in option_keys],
        key=f"question_{index}"
    )

    # Store the user's chosen option key in session state
    st.session_state.user_answers[index] = chosen_option.split(':')[0]

def display_results():
    st.empty()
    num_correct_answers = 0
    total_questions = len(st.session_state.results)

    for index, question in enumerate(st.session_state.results):
        if st.session_state.user_answers[index] == question['answer']:
            num_correct_answers += 1

    score = (num_correct_answers / total_questions) * 100 if total_questions > 0 else 0

    st.subheader("📝 Quiz Results")
    # Display score as a progress bar
    st.progress(score / 100)
    st.write(f"Your score: **{score:.2f}%**")

    if score >= 70:
        st.success("🎉 Congratulations! You passed the quiz!")
        st.balloons()
    else:
        st.error("😕 You did not pass the quiz. Try again to improve your results!")

    # Display each question and the chosen answers with color coding
    for index, question in enumerate(st.session_state.results):
        correct_answer_key = question['answer']
        user_answer_key = st.session_state.user_answers[index]

        st.write(f"### Question {index + 1}: {question['question']}")
        options_container = st.container()
        with options_container:
            for key, value in question['options'].items():
                # Use green for correct, red for incorrect, default for others
                if key == user_answer_key == correct_answer_key:
                    st.markdown(f"<span style='color:green;'>{key}: {value} ✅</span>", unsafe_allow_html=True)
                elif key == user_answer_key and key != correct_answer_key:
                    st.markdown(f"<span style='color:red;'>{key}: {value} ❌</span>", unsafe_allow_html=True)
                elif key == correct_answer_key:
                    st.markdown(f"<span style='color:green;'>{key}: {value} (Correct)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"{key}: {value}")

        # Display a supportive context or explanation beneath each question
        st.caption(f"📘 ***Explanation:*** {question['context']}")

def get_sources(documents):
    return documents[:-1]

def get_summary(documents):
    return documents[-1].page_content

def load_json(json_string):
    """
    Attempts to fix common JSON formatting issues, especially with quotes.
    """
    # Replace single quotes with double quotes
    fixed_json = json_string.replace("'", '"')

    # Escape unescaped double quotes that are within strings (naive approach)
    # For more complex JSON strings, consider using regex or more sophisticated parsing and escaping
    fixed_json = fixed_json.replace(':"', ': "').replace(', "', ', "').replace('{ "', '{ "')

    try:
        # Try parsing the JSON to see if it's correctly formatted now
        json_data = json.loads(fixed_json)
        print("JSON is valid and loaded successfully:", json_data)
        return json_data
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return None
    
def retrieve_mcqs(query, num_questions):
    summary_config = {
    "is_enabled": True, 
    "max_results": 5, 
    "response_lang": "en",
    "prompt_name": "vectara-summary-ext-v1.3.0"
    }

    retriever = vectara.as_retriever(search_kwargs={"k": 5, "summary_config": summary_config})

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")
    mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


    summarizer_model = "vectara-summary-ext-v1.3.0"
    results = searcher.send_query(
        corpus_id=int(os.getenv('CORPUS_ID')),
        query_text=query,
        num_results=10,
        summarizer_prompt_name=summarizer_model,
        response_lang="en",
        max_summarized_results=5  
    )

    # results = retriever.invoke(query)

    parsed_results = generate_questions_data('\n\n'.join([doc for doc in results 
                                                             if doc is not None]), num_questions)

    return list({q['question']: q for q in parsed_results}.values())[:num_questions]
    

st.set_page_config(page_title="🤖✨ AI Quiz Master 📚✨", page_icon='📚', layout="wide")
# Title of the app
st.title('🤖✨ AI Quiz Master 📚✨')
st.caption('Discover limitless learning with AI Quiz Master, your ultimate AI-powered tool for creating custom, engaging quizzes instantly! 🎓🚀')

# Define the sidebar for uploading and indexing documents
with st.sidebar:
    st.title("Index Documents")
    
    # Allow multiple files to be selected
    uploaded_files = st.file_uploader("Choose Documents", type=["txt", "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"], accept_multiple_files=True)
    
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                success_count = 0
                fail_count = 0
                # Loop through each uploaded file
                for uploaded_file in uploaded_files:
                    # Assuming 'indexer.upload_file' is a function you've defined or imported elsewhere
                    response, success = indexer.upload_file(
                        customer_id=int(os.getenv('CUSTOMER_ID')),
                        corpus_id=int(os.getenv('CORPUS_ID')),
                        idx_address=os.getenv('IDX_ADDRESS'),
                        uploaded_file=uploaded_file,
                        file_title=uploaded_file.name  # Use file name as title
                    )
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        st.error(f"Failed to index {uploaded_file.name}.")
                
                # Provide feedback on the process
                if success_count:
                    st.success(f"{success_count} documents indexed successfully!")
                if fail_count:
                    st.error(f"Failed to index {fail_count} documents.")
        else:
            st.warning("No files selected. Please upload some files to index.")

st.sidebar.divider()

st.sidebar.title("Quiz Configuration")
user_query = st.sidebar.text_input("Enter a topic to generate MCQs on:")
num_questions = st.sidebar.selectbox("Select the number of questions:", [5, 10, 15, 20])

if st.sidebar.button("Generate Quiz"):
    mcqs = retrieve_mcqs(user_query, num_questions)
    if mcqs:
        st.session_state.results = mcqs
        st.session_state.user_answers = [None] * len(mcqs)
        st.session_state.submitted = False
        st.sidebar.success("MCQs generated successfully! Please answer the quiz.")
    else:
        st.error("Failed to retrieve MCQs. Please try a different query.")

if st.session_state.results and not st.session_state.submitted:
    with st.form("my_form"):
        for index, question in enumerate(st.session_state.results):
            display_question(question, index)
        submit_button = st.form_submit_button("Submit Answers", on_click=lambda: st.session_state.update({"submitted": True}))
    # submit_button = st.button("Submit Answers", on_click=lambda: st.session_state.update({"submitted": True}))

elif st.session_state.submitted:
    display_results()