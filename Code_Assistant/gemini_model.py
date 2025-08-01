import os
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from config import get_gemini_api_key, GEMINI_MODEL_NAME

class GeminiModel:
    def __init__(self):
        genai.configure(api_key=get_gemini_api_key())
        self.model = GenerativeModel(model_name=GEMINI_MODEL_NAME)
    
    def generate_response(self, prompt):
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error with Gemini: {str(e)}"
    
    def create_enhancement_prompt(self, phi3_response, intent_info=None):
        """Create intent-aware prompt for Gemini to enhance Phi3's response"""
        base_prompt = """You are receiving a processed prompt from an AI model. Your task is to:
1. If it's a code generation request, provide complete, well-documented implementation
2. If it's a code explanation request, provide detailed, educational explanation  
3. If it's a general query, provide comprehensive, accurate answer

Processed prompt: {phi3_response}

Please provide your response without saying any additional things at the beginning like this response is missing something or it's not good or accurate, just do the job you know."""
        
        # Add intent-specific instructions
        if intent_info and intent_info.get("task"):
            task = intent_info["task"]
            if task == "code_generation":
                base_prompt += "\n\nSPECIAL FOCUS: Provide working, well-documented code with error handling and examples."
            elif task == "code_explanation":
                base_prompt += "\n\nSPECIAL FOCUS: Give detailed explanations with step-by-step breakdowns and educational context."
            elif task == "general_query":
                base_prompt += "\n\nSPECIAL FOCUS: Provide comprehensive, accurate information with helpful context."
        
        return base_prompt.format(phi3_response=phi3_response)
