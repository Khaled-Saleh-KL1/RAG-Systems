from Smart_Assistant import Phi3Mini4kInstruct
from gemini_model import GeminiModel
from langgraph.graph import StateGraph, END
from typing import TypedDict
from config import PHI3_MODEL_PATH
import json

# Enhanced State definition with intent information
class WorkflowState(TypedDict):
    user_prompt: str
    intent_classification: dict  # stores the intent classification result
    phi3_response: str
    gemini_response: str

# Initialize models
phi3_model = Phi3Mini4kInstruct(PHI3_MODEL_PATH)
gemini_model = GeminiModel()

def classify_intent_simple(user_input: str) -> dict:
    """
    Simple rule-based + LLM hybrid intent classification
    No separate file needed - just a function!
    """
    print(f"🧠 Classifying intent for: '{user_input[:50]}...'")
    
    # First try simple rule-based classification
    user_lower = user_input.lower()
    
    code_gen_keywords = ['write', 'create', 'make', 'function', 'class', 'code', 'implement']
    explanation_keywords = ['explain', 'what does', 'how does', 'understand', 'meaning']
    
    if any(word in user_lower for word in code_gen_keywords):
        task = "code_generation"
        confidence = 0.8
        reasoning = "Contains code generation keywords"
    elif any(word in user_lower for word in explanation_keywords):
        task = "code_explanation" 
        confidence = 0.8
        reasoning = "Contains explanation keywords"
    else:
        # Use LLM for ambiguous cases
        try:
            llm_prompt = f"""Classify this user input into ONE category:
1. "code_generation" - wants to create/write code
2. "code_explanation" - wants to understand code/concepts  
3. "general_query" - general question

User input: "{user_input}"

Respond with just: {{"task": "category", "confidence": 0.9}}"""
            
            response = gemini_model.generate_response(llm_prompt)
            classification = json.loads(response)
            task = classification["task"]
            confidence = classification.get("confidence", 0.7)
            reasoning = "LLM classification for ambiguous input"
        except:
            # Final fallback
            task = "general_query"
            confidence = 0.6
            reasoning = "Fallback classification"
    
    result = {
        "task": task,
        "user_input": user_input,
        "confidence": confidence,
        "reasoning": reasoning
    }
    
    print(f"✅ Intent: {task} (confidence: {confidence})")
    return result

def route_node(state: WorkflowState) -> dict:
    """
    SIMPLE ROUTE NODE: Classifies user intent 
    Uses our simple classify_intent_simple function - no separate file needed!
    """
    print("🚦 Route Node: Analyzing user intent...")
    
    try:
        # Use our simple classification function
        classification = classify_intent_simple(state["user_prompt"])
        print(f"🎯 Routing decision: {classification['task']}")
        return {"intent_classification": classification}
        
    except Exception as e:
        # Fallback classification
        fallback = {
            "task": "general_query",
            "user_input": state["user_prompt"],
            "confidence": 0.5,
            "reasoning": f"Error in classification: {str(e)}"
        }
        return {"intent_classification": fallback}

def phi3_node(state: WorkflowState) -> dict:
    """Process with Phi3 - now intent-aware"""
    intent = state["intent_classification"]["task"]
    print(f"🔄 Processing with Phi3 for {intent} task...")
    
    try:
        # Pass intent information to Phi3
        response = phi3_model.generate_response(
            state["user_prompt"], 
            intent_info=state["intent_classification"]
        )
        return {"phi3_response": response}
    except Exception as e:
        return {"phi3_response": f"Phi3 Error: {str(e)}"}

def gemini_node(state: WorkflowState) -> dict:
    """Process with Gemini - now intent-aware"""
    intent = state["intent_classification"]["task"]
    print(f"🔄 Processing with Gemini for {intent} task...")
    
    try:
        # Pass intent information to Gemini
        enhanced_prompt = gemini_model.create_enhancement_prompt(
            state["phi3_response"],
            intent_info=state["intent_classification"]
        )
        response = gemini_model.generate_response(enhanced_prompt)
        return {"gemini_response": response}
    except Exception as e:
        return {"gemini_response": f"Gemini Error: {str(e)}"}

def decide_next_step(state: WorkflowState) -> str:
    """
    CONDITIONAL EDGE FUNCTION: Decides which node to go to next
    
    This function is called by LangGraph to determine the next step in the workflow.
    Based on the intent classification, it routes to appropriate processing nodes.
    
    Args:
        state: Current workflow state with intent_classification
        
    Returns:
        str: Name of the next node to execute
    """
    intent = state["intent_classification"]["task"]
    confidence = state["intent_classification"]["confidence"]
    
    print(f"🔀 Deciding next step for intent: {intent} (confidence: {confidence})")
    
    # For now, all intents go through the same Phi3 → Gemini pipeline
    # But this is where you could route to different specialized nodes
    
    if intent == "code_generation":
        print("→ Routing to code generation pipeline")
        return "phi3_processor"
    elif intent == "code_explanation":
        print("→ Routing to code explanation pipeline")  
        return "phi3_processor"
    elif intent == "general_query":
        print("→ Routing to general query pipeline")
        return "phi3_processor"
    else:
        # Fallback
        print("→ Using default routing")
        return "phi3_processor"

# Create workflow with smart routing
def create_workflow():
    """
    Creates the LangGraph workflow with intelligent routing
    
    New Flow:
    1. User Input → Route Node (classifies intent)
    2. Route Node → Conditional Edge (decides next step)
    3. Conditional Edge → Appropriate Processing Node
    4. Processing continues through Phi3 → Gemini → End
    """
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes
    workflow.add_node("router", route_node)              # NEW: Intent classification
    workflow.add_node("phi3_processor", phi3_node)       # Existing: Phi3 processing
    workflow.add_node("gemini_processor", gemini_node)   # Existing: Gemini processing
    
    # Set entry point to our router
    workflow.set_entry_point("router")                   # NEW: Start with routing
    
    # Add conditional edge from router to processing nodes
    workflow.add_conditional_edges(
        "router",                    # From the router node
        decide_next_step,           # Use this function to decide next step
        {
            "phi3_processor": "phi3_processor"  # Map decision to actual node
        }
    )
    
    # Continue with existing flow
    workflow.add_edge("phi3_processor", "gemini_processor")
    workflow.add_edge("gemini_processor", END)
    
    return workflow.compile()

def run_console():
    """Run in console mode with smart routing"""
    print("=== AI Workflow: Smart Router → Phi3 → Gemini ===")
    workflow = create_workflow()
    
    while True:
        prompt = input("\n💭 Enter your prompt (or 'quit'): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        try:
            print("🚀 Starting smart workflow...")
            result = workflow.invoke({
                "user_prompt": prompt,
                "intent_classification": {},  # Will be filled by router
                "phi3_response": "",
                "gemini_response": ""
            })
            
            print(f"\n{'='*60}")
            print(f"🔸 Original Prompt: {prompt}")
            print(f"🧠 Intent: {result['intent_classification']['task']} "
                  f"(confidence: {result['intent_classification']['confidence']})")
            print(f"📝 Reasoning: {result['intent_classification']['reasoning']}")
            print(f"\n🤖 Phi3: {result['phi3_response']}")
            print(f"\n✨ Gemini: {result['gemini_response']}")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_console()

