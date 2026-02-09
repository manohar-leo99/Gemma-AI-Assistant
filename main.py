import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

## Streamlit Framework
st.set_page_config(
    page_title="AI Assistant", 
    layout="centered",
    page_icon="🤖"
)

# Blue, Green, Black, Red Theme CSS
st.markdown("""
    <style>
    /* Dark Background with Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0a0f1a 100%);
    }
    
    /* Animated Title - Blue & Green Gradient */
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px #10b981, 0 0 40px #06b6d4; }
        50% { text-shadow: 0 0 30px #06b6d4, 0 0 60px #10b981; }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    h1 {
        animation: slideDown 1s ease-out, glow 2s ease-in-out infinite;
        text-align: center;
        background: linear-gradient(135deg, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }
    
    /* Floating Robot Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-25px) rotate(5deg); }
    }
    
    .stImage {
        animation: float 4s ease-in-out infinite;
        filter: drop-shadow(0 0 30px #06b6d4);
        margin: 0 auto;
        display: block;
    }
    
    /* Fade In Scale Animation */
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Text Area - Blue Border */
    .stTextArea textarea {
        background: #1e293b;
        border-radius: 15px;
        border: 3px solid #06b6d4;
        padding: 20px;
        font-size: 17px;
        color: #e2e8f0;
        transition: all 0.4s ease;
        animation: fadeInScale 0.8s ease-out;
    }
    
    .stTextArea textarea:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.3), 0 0 30px rgba(6, 182, 212, 0.5);
        background: #0f172a;
    }
    
    .stTextArea textarea::placeholder {
        color: #64748b;
    }
    
    /* Button - Red & Green Gradient */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        50% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ef4444, #10b981);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s ease;
        width: 100%;
        animation: fadeInScale 1s ease-out, pulse 2s infinite;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(239, 68, 68, 0.5), 0 0 50px rgba(16, 185, 129, 0.3);
        background: linear-gradient(135deg, #dc2626, #059669);
    }
    
    /* Response Box - Black with Green/Blue Accents */
    .response-box {
        animation: fadeInScale 0.6s ease-out;
        background: linear-gradient(145deg, #0f172a, #1e293b);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        border: 2px solid #10b981;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .response-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.3), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        to { left: 100%; }
    }
    
    .response-box p {
        color: #e2e8f0 !important;
        line-height: 2;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Success Message - Green */
    .stSuccess {
        animation: fadeInScale 0.5s ease-out;
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 12px;
        border: none !important;
    }
    
    /* Warning Message - Red */
    .stWarning {
        animation: fadeInScale 0.5s ease-out;
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
        border-radius: 12px;
        border: none !important;
    }
    
    /* Spinner - Blue Glow */
    @keyframes spin-glow {
        0% { filter: drop-shadow(0 0 10px #06b6d4); }
        50% { filter: drop-shadow(0 0 30px #10b981); }
        100% { filter: drop-shadow(0 0 10px #06b6d4); }
    }
    
    .stSpinner > div {
        animation: spin-glow 1.5s ease-in-out infinite;
        border-top-color: #06b6d4 !important;
    }
    
    /* Divider - Colorful */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, #ef4444, #06b6d4, #10b981, #ef4444);
        animation: fadeInScale 1s ease-out;
    }
    
    /* Subtitle Color */
    p {
        color: #94a3b8;
    }
    
    /* Label Text */
    .stTextArea label {
        color: #06b6d4 !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<br>", unsafe_allow_html=True)
st.image("https://img.icons8.com/fluency/150/artificial-intelligence.png", width=150)
st.title("🤖 AI Assistant")
st.markdown("<p style='text-align: center; color: #10b981; font-size: 1.2rem; font-weight: 600;'>Powered by Advanced AI Technology</p>", unsafe_allow_html=True)

st.divider()

# Input Section
st.markdown("<p style='color: #06b6d4; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;'>💭 Ask Your Question</p>", unsafe_allow_html=True)
input_text = st.text_area(
    "Your Question", 
    height=160, 
    placeholder="Type anything you want to know...",
    label_visibility="collapsed"
)

submit_btn = st.button("🚀 Generate Answer")

st.divider()

## Ollama Gemma Model
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

## Results
if submit_btn and input_text:
    with st.spinner("🧠 AI Processing..."):
        response = chain.invoke({"question": input_text})
    
    st.success("✅ Answer Generated Successfully!")
    
    st.markdown(f"""
        <div class='response-box'>
            <p style='margin: 0; font-size: 1.1rem;'>
                {response}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
elif submit_btn and not input_text:
    st.warning("⚠️ Please enter a question first!")