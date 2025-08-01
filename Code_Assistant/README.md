# AI Workflow Assistant: Smart Router → Phi3 → Gemini

An intelligent AI workflow system that combines multiple AI models using LangGraph. The system features intent classification, processes prompts through Microsoft's Phi3-Mini model, and enhances responses using Google's Gemini AI, all wrapped in an interactive Streamlit GUI.

## Features

🤖 **Multi-Model AI Pipeline**: Seamlessly integrates Phi3 and Gemini models  
🎯 **Smart Intent Classification**: Automatically categorizes user queries  
💬 **Chat-like Interface**: Interactive Streamlit GUI with conversation history  
⚡ **GPU Acceleration**: Optimized for CUDA-compatible GPUs  
🔄 **Real-time Processing**: Live status updates and background processing  
📊 **Expandable Results**: Organized display of intent classification, Phi3, and Gemini responses  

## Setup

### Prerequisites
1. **GPU Requirements**: CUDA-compatible GPU for optimal Phi3 performance
2. **Python**: Python 3.8 or higher
3. **Phi3 Model**: Download the model file and place it in `./model/` directory
   - Download from: [Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main)
4. **Gemini API Key**: Set up your Google Gemini API key
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

### Clone the repository:
```bash
git clone https://github.com/Khaled-Saleh-KL1/RAG-Systems/Code_Assistant.git
cd Code_Assistant
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```
**I recommend you to download torch library from the official website [Pytorch](https://pytorch.org/)**

## Project Structure
```
AI-Workflow-Assistant/
├── streamlit_GUI.py      # Streamlit chat-based interface
├── Gradio_GUI.py         # Gradio interactive interface
├── workflow.py           # LangGraph workflow implementation
├── Smart_Assistant.py    # Phi3 model wrapper
├── gemini_model.py       # Google Gemini API wrapper
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── run_workflow.sh       # Interactive shell script runner
├── evaluate.py           # RAG evaluation system
├── run_evaluate.sh       # RAG evaluation runner script
├── mbpp.jsonl            # MBPP dataset for evaluation
└── README.md            # This file
```

## Usage

### Option 1: Interactive Runner Script (Recommended)
```bash
# Launch the interactive menu to choose your interface
./run_workflow.sh

# You'll see a menu:
# [S/s] - Streamlit Interface (Chat-based UI)
# [G/g] - Gradio Interface (Interactive UI)  
# [Q/q] - Quit
```

### Option 2: Direct Interface Launch

#### Streamlit Interface (Chat-based)
```bash
# Launch the Streamlit chat interface
streamlit run streamlit_GUI.py
# Opens at: http://localhost:8501
```

#### Gradio Interface (Interactive)
```bash
# Launch the Gradio interactive interface
python3 Gradio_GUI.py
# Opens at: http://localhost:7860
```

### Option 3: Legacy Direct Execution
```bash
# Console mode  
python3 workflow.py --console
```

## RAG Evaluation on MBPP Dataset

The system includes a comprehensive RAG evaluation framework using the MBPP (Mostly Basic Python Problems) dataset to assess retrieval and LLM effectiveness on real-world Python programming tasks.

### 🎯 Evaluation Goals
- **Retrieval Quality**: Measure how well the system retrieves relevant code examples
- **Code Generation**: Assess LLM effectiveness in generating working Python solutions
- **Test Execution**: Validate generated code against provided test cases

### 🚀 Running RAG Evaluation

#### Quick Start
```bash
# Run the complete RAG evaluation
./run_evaluate.sh
```

#### Manual Execution
```bash
# Run evaluation directly
python3 evaluate.py
```

### 📊 Evaluation Process

1. **📁 Dataset Loading**: Loads 10 MBPP examples with task descriptions, solutions, and test cases
2. **🧮 Embedding Creation**: Creates semantic embeddings for similarity search
3. **🔍 Retrieval**: For each problem, retrieves 3 most similar examples using cosine similarity
4. **🤖 RAG Generation**: Creates prompts with retrieved examples, processes through Phi3 → Gemini
5. **✅ Test Execution**: Runs generated code against test cases in isolated environment
6. **📈 Metrics Calculation**: Computes pass rates, retrieval quality, and error analysis

### 📋 Evaluation Metrics

**Code Quality Metrics:**
- **Overall Pass Rate**: Percentage of test cases passed across all examples
- **Perfect Solutions**: Number of examples with 100% test pass rate
- **Partial Solutions**: Examples with some passing tests
- **Failed Solutions**: Examples with 0% pass rate

**Retrieval Quality Metrics:**
- **Average Retrieval Score**: Mean cosine similarity of retrieved examples
- **Retrieval Score Range**: Min/max similarity scores
- **Retrieval Relevance**: Quality of similar examples found

**Error Analysis:**
- **Execution Errors**: Code that failed to run
- **Generation Errors**: Issues in LLM code generation
- **Test Failures**: Specific test case failures

### 📄 Output Files

After running evaluation, you'll find:
- `rag_evaluation_report.json`: Comprehensive evaluation results
- `rag_evaluation_TIMESTAMP.log`: Detailed execution logs
- Console summary with key metrics

### 💡 Example Output
```
🎯 RAG EVALUATION SUMMARY
=====================================
📊 Dataset: 10 examples
✅ Overall Pass Rate: 70.0%
🎯 Perfect Solutions: 4
⚠️  Failed Solutions: 2
🔄 Partial Solutions: 4

🔍 Retrieval Quality:
   • Average Score: 0.342
   • Min Score: 0.156
   • Max Score: 0.678

❌ Execution Errors: 1
```


## How It Works

1. **🎯 Intent Classification**: The system first analyzes your prompt to understand the intent
2. **🤖 Phi3 Processing**: Microsoft's Phi3-Mini processes and optimizes your prompt locally
3. **✨ Gemini Enhancement**: Google's Gemini takes Phi3's output and provides enhanced responses
4. **📋 Results Display**: You get organized results showing all processing stages

## GUI Features

### 🖥️ Dual Interface Options

**Streamlit Interface (Chat-based)**
- **💬 Chat Interface**: Natural conversation flow with message history
- **📊 Expandable Sections**: Collapsible areas for intent classification and model responses
- **🔄 Auto-reset Input**: Chat input automatically clears and resizes after submission
- **⌨️ Keyboard Shortcuts**: Press Enter to send, Shift+Enter for multi-line input

**Gradio Interface (Interactive)**
- **🎛️ Interactive Components**: Modern UI with organized layout
- **📝 Multi-line Input**: Flexible text input with proper sizing
- **🔘 Button Controls**: Clear send and clear buttons
- **📱 Responsive Design**: Works well on different screen sizes

### 🚀 Common Features (Both Interfaces)
- **⚡ Real-time Processing**: Live status updates during AI processing  
- **🖥️ GPU Status Display**: Shows current GPU availability and usage
- **❌ Error Handling**: Graceful error display and recovery
- **🌐 Local Hosting**: Runs locally for privacy and speed
- **🎯 Professional UI**: Clean, modern interface design
- **⌨️ Keyboard Shortcuts**: Press Enter to send, Shift+Enter for multi-line input

## Configuration

The system uses environment variables and a config file for settings:

- **GEMINI_API_KEY**: Your Google Gemini API key
- **PHI3_MODEL_PATH**: Path to the Phi3 model file (configured in `config.py`)
- **GEMINI_MODEL_NAME**: Gemini model version to use

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Gemini API key
- Sufficient RAM for model loading (8GB+ recommended)

## Troubleshooting

**GPU Issues**: If you don't have a CUDA GPU, the system will fall back to CPU processing (slower)  
**Model Loading**: Ensure the Phi3 model file is correctly placed in the `./model/` directory  
**API Key**: Make sure your Gemini API key is properly set as an environment variable  

## Author

**Khaled Saleh**  
**Email**: khaledsalehkl1@gmail.com