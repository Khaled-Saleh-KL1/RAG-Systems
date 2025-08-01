"""
RAG Evaluation on MBPP Examples
Goal: Evaluate retrieval + LLM effectiveness on real-world Python problems.

This script implements:
1. MBPP dataset loading and processing
2. Embedding-based retrieval system
3. RAG-style code generation
4. Test case execution and evaluation
5. Retrieval quality assessment
"""

import json
import numpy as np
import os
import sys
import subprocess
import tempfile
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Import your existing models
from workflow import create_workflow, phi3_model, gemini_model
from Smart_Assistant import Phi3Mini4kInstruct
from gemini_model import GeminiModel

@dataclass
class MBPPExample:
    """Data class for MBPP examples"""
    task_id: int
    text: str
    code: str
    test_list: List[str]
    test_setup_code: str = ""
    challenge_test_list: List[str] = None

@dataclass
class RAGResult:
    """Data class for RAG evaluation results"""
    task_id: int
    query: str
    retrieved_examples: List[Dict]
    generated_code: str
    test_results: Dict[str, bool]
    pass_rate: float
    retrieval_scores: List[float]
    execution_error: str = None

class MBPPEvaluator:
    """RAG Evaluation system for MBPP dataset"""
    
    def __init__(self, data_file: str = "mbpp.jsonl", num_examples: int = 10):
        self.data_file = data_file
        self.num_examples = num_examples
        self.examples = []
        self.embeddings = None
        self.embedding_model = None
        self.workflow = None
        
        # Load data and initialize models
        self._load_data()
        self._init_embedding_model()
        self._init_workflow()
        
    def _load_data(self):
        """Load MBPP examples from jsonl file"""
        print(f"📁 Loading MBPP data from {self.data_file}...")
        
        try:
            with open(self.data_file, 'r') as f:
                data = [json.loads(line) for line in f]
            
            # Take first num_examples for evaluation
            selected_data = data[:self.num_examples]
            
            for item in selected_data:
                example = MBPPExample(
                    task_id=item['task_id'],
                    text=item['text'],
                    code=item['code'],
                    test_list=item['test_list'],
                    test_setup_code=item.get('test_setup_code', ''),
                    challenge_test_list=item.get('challenge_test_list', [])
                )
                self.examples.append(example)
                
            print(f"✅ Loaded {len(self.examples)} MBPP examples")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        print("🔧 Initializing embedding model...")
        try:
            # Use a good code-understanding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            sys.exit(1)
    
    def _init_workflow(self):
        """Initialize the RAG workflow"""
        print("🚀 Initializing AI workflow...")
        try:
            self.workflow = create_workflow()
            print("✅ Workflow initialized")
        except Exception as e:
            print(f"❌ Error initializing workflow: {e}")
            sys.exit(1)
    
    def create_embeddings(self):
        """Create embeddings for all examples"""
        print("🧮 Creating embeddings for MBPP examples...")
        
        # Combine text and code for better representation
        texts = []
        for example in self.examples:
            combined_text = f"{example.text} {example.code}"
            texts.append(combined_text)
        
        self.embeddings = self.embedding_model.encode(texts)
        print(f"✅ Created embeddings for {len(texts)} examples")
    
    def retrieve_similar_examples(self, query: str, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """Retrieve k most similar examples for the query"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k similar examples (excluding the query itself if it exists)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved_examples = []
        retrieval_scores = []
        
        for idx in top_indices:
            retrieved_examples.append({
                'task_id': self.examples[idx].task_id,
                'text': self.examples[idx].text,
                'code': self.examples[idx].code,
                'similarity_score': float(similarities[idx])
            })
            retrieval_scores.append(float(similarities[idx]))
        
        return retrieved_examples, retrieval_scores
    
    def generate_rag_prompt(self, query: str, retrieved_examples: List[Dict]) -> str:
        """Create RAG prompt with retrieved examples"""
        prompt_parts = []
        prompt_parts.append("You are a Python programming expert. Use the following examples to help solve the new problem.")
        prompt_parts.append("\\n--- SIMILAR EXAMPLES ---")
        
        for i, example in enumerate(retrieved_examples, 1):
            prompt_parts.append(f"\\nExample {i}:")
            prompt_parts.append(f"Problem: {example['text']}")
            prompt_parts.append(f"Solution:\\n{example['code']}")
            prompt_parts.append(f"Similarity: {example['similarity_score']:.3f}")
        
        prompt_parts.append("\\n--- NEW PROBLEM ---")
        prompt_parts.append(f"Problem: {query}")
        prompt_parts.append("\\nPlease provide a Python function that solves this problem. Return only the code without explanations.")
        
        return "\\n".join(prompt_parts)
    
    def execute_code_with_tests(self, code: str, test_cases: List[str], setup_code: str = "") -> Dict[str, bool]:
        """Execute generated code with test cases"""
        results = {}
        
        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write setup code, generated code, and test cases
            f.write(setup_code + "\\n")
            f.write(code + "\\n")
            
            # Write test execution code
            f.write("\\n# Test execution\\n")
            for i, test_case in enumerate(test_cases):
                f.write(f"try:\\n")
                f.write(f"    result_{i} = {test_case}\\n")
                f.write(f"    print(f'TEST_{i}_PASS')\\n")
                f.write(f"except Exception as e:\\n")
                f.write(f"    print(f'TEST_{i}_FAIL:{{e}}')\\n")
            
            temp_file = f.name
        
        try:
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse results
            output_lines = result.stdout.strip().split('\\n')
            
            for i, test_case in enumerate(test_cases):
                test_key = f"test_{i+1}"
                if f"TEST_{i}_PASS" in result.stdout:
                    results[test_key] = True
                else:
                    results[test_key] = False
            
            # If stderr contains errors, capture them
            if result.stderr:
                results['execution_error'] = result.stderr
                
        except subprocess.TimeoutExpired:
            results['execution_error'] = "Code execution timeout"
            for i in range(len(test_cases)):
                results[f"test_{i+1}"] = False
                
        except Exception as e:
            results['execution_error'] = str(e)
            for i in range(len(test_cases)):
                results[f"test_{i+1}"] = False
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return results
    
    def evaluate_single_example(self, example: MBPPExample) -> RAGResult:
        """Evaluate a single MBPP example using RAG"""
        print(f"\\n🧪 Evaluating Task {example.task_id}: {example.text[:60]}...")
        
        # Step 1: Retrieve similar examples (excluding the current one)
        other_examples = [ex for ex in self.examples if ex.task_id != example.task_id]
        if len(other_examples) < len(self.examples):
            # Temporarily create embeddings excluding current example
            temp_texts = []
            for ex in other_examples:
                combined_text = f"{ex.text} {ex.code}"
                temp_texts.append(combined_text)
            temp_embeddings = self.embedding_model.encode(temp_texts)
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([example.text])
            similarities = cosine_similarity(query_embedding, temp_embeddings)[0]
            
            # Get top 3 similar examples
            top_indices = np.argsort(similarities)[::-1][:3]
            retrieved_examples = []
            retrieval_scores = []
            
            for idx in top_indices:
                retrieved_examples.append({
                    'task_id': other_examples[idx].task_id,
                    'text': other_examples[idx].text,
                    'code': other_examples[idx].code,
                    'similarity_score': float(similarities[idx])
                })
                retrieval_scores.append(float(similarities[idx]))
        else:
            retrieved_examples, retrieval_scores = self.retrieve_similar_examples(example.text, k=3)
        
        print(f"📊 Retrieved {len(retrieved_examples)} similar examples")
        print(f"🎯 Retrieval scores: {[f'{score:.3f}' for score in retrieval_scores]}")
        
        # Step 2: Generate RAG prompt
        rag_prompt = self.generate_rag_prompt(example.text, retrieved_examples)
        
        # Step 3: Generate code using workflow
        try:
            workflow_result = self.workflow.invoke({
                "user_prompt": rag_prompt,
                "intent_classification": {},
                "phi3_response": "",
                "gemini_response": ""
            })
            
            generated_code = workflow_result['gemini_response']
            print(f"🤖 Generated code: {len(generated_code)} characters")
            
        except Exception as e:
            print(f"❌ Error generating code: {e}")
            generated_code = "# Error in code generation"
        
        # Step 4: Execute tests
        test_results = self.execute_code_with_tests(
            generated_code, 
            example.test_list, 
            example.test_setup_code
        )
        
        # Step 5: Calculate pass rate
        passed_tests = sum(1 for result in test_results.values() if isinstance(result, bool) and result)
        total_tests = len(example.test_list)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        print(f"✅ Test Results: {passed_tests}/{total_tests} passed ({pass_rate:.1%})")
        
        return RAGResult(
            task_id=example.task_id,
            query=example.text,
            retrieved_examples=retrieved_examples,
            generated_code=generated_code,
            test_results=test_results,
            pass_rate=pass_rate,
            retrieval_scores=retrieval_scores,
            execution_error=test_results.get('execution_error')
        )
    
    def run_evaluation(self) -> List[RAGResult]:
        """Run complete RAG evaluation on all examples"""
        print("\\n🚀 Starting RAG Evaluation on MBPP Examples")
        print("=" * 60)
        
        # Create embeddings
        self.create_embeddings()
        
        # Evaluate each example
        results = []
        for i, example in enumerate(self.examples, 1):
            print(f"\\n[{i}/{len(self.examples)}] Processing Task {example.task_id}")
            result = self.evaluate_single_example(example)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[RAGResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        print("\\n📊 Generating Evaluation Report...")
        
        # Calculate overall metrics
        total_examples = len(results)
        overall_pass_rate = np.mean([r.pass_rate for r in results])
        perfect_solutions = sum(1 for r in results if r.pass_rate == 1.0)
        failed_solutions = sum(1 for r in results if r.pass_rate == 0.0)
        
        # Retrieval quality metrics
        avg_retrieval_score = np.mean([np.mean(r.retrieval_scores) for r in results])
        min_retrieval_score = np.min([np.min(r.retrieval_scores) for r in results])
        max_retrieval_score = np.max([np.max(r.retrieval_scores) for r in results])
        
        # Error analysis
        execution_errors = sum(1 for r in results if r.execution_error)
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_examples': total_examples,
                'dataset_file': self.data_file
            },
            'overall_metrics': {
                'overall_pass_rate': overall_pass_rate,
                'perfect_solutions': perfect_solutions,
                'failed_solutions': failed_solutions,
                'partial_solutions': total_examples - perfect_solutions - failed_solutions
            },
            'retrieval_metrics': {
                'avg_retrieval_score': avg_retrieval_score,
                'min_retrieval_score': min_retrieval_score,
                'max_retrieval_score': max_retrieval_score
            },
            'error_analysis': {
                'execution_errors': execution_errors,
                'generation_errors': 0  # Could be enhanced
            },
            'detailed_results': []
        }
        
        # Add detailed results
        for result in results:
            report['detailed_results'].append({
                'task_id': result.task_id,
                'query': result.query[:100] + "..." if len(result.query) > 100 else result.query,
                'pass_rate': result.pass_rate,
                'retrieval_scores': result.retrieval_scores,
                'test_results': result.test_results,
                'execution_error': result.execution_error,
                'generated_code_length': len(result.generated_code)
            })
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print evaluation summary"""
        print("\\n" + "="*60)
        print("🎯 RAG EVALUATION SUMMARY")
        print("="*60)
        
        metrics = report['overall_metrics']
        retrieval = report['retrieval_metrics']
        errors = report['error_analysis']
        
        print(f"📊 Dataset: {report['dataset_info']['total_examples']} examples")
        print(f"✅ Overall Pass Rate: {metrics['overall_pass_rate']:.1%}")
        print(f"🎯 Perfect Solutions: {metrics['perfect_solutions']}")
        print(f"⚠️  Failed Solutions: {metrics['failed_solutions']}")
        print(f"🔄 Partial Solutions: {metrics['partial_solutions']}")
        print()
        print(f"🔍 Retrieval Quality:")
        print(f"   • Average Score: {retrieval['avg_retrieval_score']:.3f}")
        print(f"   • Min Score: {retrieval['min_retrieval_score']:.3f}")
        print(f"   • Max Score: {retrieval['max_retrieval_score']:.3f}")
        print()
        print(f"❌ Execution Errors: {errors['execution_errors']}")
        
        print("\\n" + "="*60)
    
    def save_report(self, report: Dict[str, Any], filename: str = "rag_evaluation_report.json"):
        """Save evaluation report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"💾 Report saved to {filename}")

def main():
    """Main evaluation function"""
    print("🚀 RAG Evaluation on MBPP Examples")
    print("Goal: Evaluate retrieval + LLM effectiveness on real-world Python problems")
    print()
    
    # Initialize evaluator
    evaluator = MBPPEvaluator(data_file="mbpp.jsonl", num_examples=10)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Generate and print report
    report = evaluator.generate_report(results)
    evaluator.print_summary(report)
    
    # Save detailed report
    evaluator.save_report(report)
    
    print("\\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()