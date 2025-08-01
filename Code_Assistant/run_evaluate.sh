#!/bin/bash

# RAG Evaluation Runner Script for MBPP Examples
echo "🚀 RAG Evaluation on MBPP Examples"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed!"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY environment variable not set!"
    echo "⚠️ Please run: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Check if required files exist
echo "🔍 Checking required files..."

if [ ! -f "evaluate.py" ]; then
    echo "❌ evaluate.py not found!"
    exit 1
fi

if [ ! -f "mbpp.jsonl" ]; then
    echo "❌ mbpp.jsonl not found!"
    echo "💡 Please ensure the MBPP dataset file is in the current directory"
    exit 1
fi

if [ ! -f "workflow.py" ]; then
    echo "❌ workflow.py not found!"
    exit 1
fi

echo "✅ All required files found"

# Install additional dependencies if needed
echo "📦 Checking dependencies..."

# Check if required packages are installed
python3 -c "import sentence_transformers, sklearn, pandas, jsonlines" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing additional dependencies..."
    uv pip install sentence-transformers scikit-learn pandas jsonlines
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies!"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ All dependencies are already installed"
fi

echo " Starting RAG Evaluation..."
echo "   • Dataset: MBPP (10 examples)"
echo "   • Method: Retrieval + LLM (Phi3 → Gemini)"
echo "   • Metrics: Pass/Fail on test cases + Retrieval quality"
echo "   • Working directory: $(pwd)"
echo ""

# Set fixed log file name
LOG_FILE="rag_evaluation_log.log"

echo "🏃 Running evaluation... (This may take several minutes)"
echo "📝 Logging output to: $LOG_FILE"

# Run evaluation and capture output (directly in current directory)
python3 evaluate.py 2>&1 | tee "$LOG_FILE"

# Check if evaluation completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    
    # Show generated files in current directory
    echo "📄 Generated files:"
    ls -la rag_evaluation_*.json rag_evaluation_*.log 2>/dev/null
    
    # Display quick summary if report exists
    if [ -f "rag_evaluation_report.json" ]; then
        echo ""
        echo "📊 Quick Summary (from report):"
        python3 -c "
import json
try:
    with open('rag_evaluation_report.json', 'r') as f:
        report = json.load(f)
    print(f'📈 Overall Pass Rate: {report[\"overall_metrics\"][\"overall_pass_rate\"]:.1%}')
    print(f'🎯 Perfect Solutions: {report[\"overall_metrics\"][\"perfect_solutions\"]}')
    print(f'🔍 Avg Retrieval Score: {report[\"retrieval_metrics\"][\"avg_retrieval_score\"]:.3f}')
except:
    print('Report summary not available')
"
    fi
    
    echo ""
    echo "🎉 RAG Evaluation Results:"
    echo "   • Detailed report: rag_evaluation_report.json"
    echo "   • Full log: $LOG_FILE"
    echo "   • Location: $(pwd)"
    
else
    echo ""
    echo "❌ Evaluation failed!"
    echo "🔍 Check the log file for details: $LOG_FILE"
    exit 1
fi

echo ""
echo "🏁 RAG Evaluation completed!"
echo "📁 All files saved in current directory"
