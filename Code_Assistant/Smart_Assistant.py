from llama_cpp import Llama as GGUFInference
import torch

class Phi3Mini4kInstruct:
    def __init__(self, model_path="../Week4/model/Phi-3-mini-4k-instruct-q4.gguf"):
        self.system_prompt = '''You are a PROMPT OPTIMIZER. Rephrase user requests into clear, detailed prompts. DO NOT provide answers or code.

Rules:
1. CODE REQUEST: Transform into detailed generation prompt with requirements
2. EXPLANATION REQUEST: Ask for clear, line-by-line explanation  
3. GENERAL QUERY: Fix grammar and make it clear

Output ONLY the rephrased prompt.'''

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. Please check your setup.")

        self.model = GGUFInference(
            model_path=model_path, 
            n_gpu_layers=-1,
            n_ctx=2096,
            n_batch=512,
            verbose=False,
        )

    def generate_response(self, user_prompt, intent_info=None):
        """Generate response with optional intent awareness"""
        system_prompt = self.system_prompt
        
        # Adjust system prompt based on intent if provided
        if intent_info and intent_info.get("task"):
            task = intent_info["task"]
            if task == "code_generation":
                system_prompt += "\n\nFOCUS: This is a CODE GENERATION request. Make the prompt very specific about requirements, error handling, and documentation."
            elif task == "code_explanation":
                system_prompt += "\n\nFOCUS: This is a CODE EXPLANATION request. Ask for detailed, step-by-step explanations with examples."
            elif task == "general_query":
                system_prompt += "\n\nFOCUS: This is a GENERAL QUERY. Make the prompt clear and comprehensive."
        
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2048, 
            temperature=0.3,
            top_p=0.8,
            repeat_penalty=1.1,
        )
        
        return response['choices'][0]['message']['content'].strip()