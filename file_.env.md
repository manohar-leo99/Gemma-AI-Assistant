# 🔐 Environment Variables Configuration

# Copy this file to .env and fill in your actual values

# DO NOT commit .env file to GitHub (it's in .gitignore)

# ====================================

# 🤖 LangChain Configuration

# ====================================

# LangSmith API Key for tracking and monitoring

# Get it from: https://smith.langchain.com/

LANGCHAIN_API_KEY=your_langsmith_api_key_here

# LangSmith Project Name

LANGCHAIN_PROJECT=gemma-ai-assistant

# Enable LangSmith tracing (true/false)

LANGCHAIN_TRACING_V2=true

# ====================================

# 🦙 Ollama Configuration

# ====================================

# Ollama Server URL (default: http://localhost:11434)

OLLAMA_BASE_URL=http://localhost:11434

# Model name to use

# Options: gemma3, llama2, mistral, neural-chat, etc.

OLLAMA_MODEL=gemma3

# ====================================

# 🌐 Streamlit Configuration

# ====================================

# Streamlit server port (default: 8501)

STREAMLIT_SERVER_PORT=8501

# Streamlit theme (light/dark)

STREAMLIT_THEME_BASE=dark

# ====================================

# 📊 Application Settings

# ====================================

# Application title

APP_TITLE=Gemma AI Assistant

# Application description

APP_DESCRIPTION=An intelligent chatbot powered by Ollama, LangChain, and Streamlit

# Maximum tokens in response

MAX_TOKENS=1000

# Temperature for AI responses (0.0-1.0)

# Lower = more deterministic, Higher = more creative

TEMPERATURE=0.7

# ====================================

# 🔍 Debug & Logging

# ====================================

# Enable debug mode (true/false)

DEBUG=false

# Log level (DEBUG, INFO, WARNING, ERROR)

LOG_LEVEL=INFO

# ====================================

# 📧 Optional: Email Configuration

# ====================================

# SMTP Server (if needed for notifications)

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# ====================================

# 🔑 Optional: API Keys

# ====================================

# Add any additional API keys here

# API_KEY_1=your_api_key_here

# API_KEY_2=your_api_key_here

# ====================================

# 📝 Notes

# ====================================

# 1. Copy this file and rename to .env

# 2. Fill in your actual values

# 3. Never commit .env to version control

# 4. Check .gitignore includes .env file

# 5. Share .env.example with your team
