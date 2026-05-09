import os
from groq import Groq
from rag.retriever import retrieve
from rag.prompt import SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it in Railway Variables.")

groq_client = Groq(api_key=GROQ_API_KEY)

def generate_response(query):
    # --- STEP 0: STATELESS GUARDRAILS ---
    clean_query = query.lower().strip()
    
    # Handle Greetings
    greetings = ["hello", "hi", "hey", "assalam o alaikum"]
    if clean_query in greetings:
        return "Hello! Welcome to NeuraFlux. What brings you to us today as you look to explore AI for your business?"

    # Handle "Yes" (Intent Expansion for RAG)
    if clean_query in ["yes", "yeah", "yup", "sure", "ok", "okay"]:
        query = "Tell me how the NeuraFlux free AI growth audit helps my business and where to book it."

    # Step 1: Context Retrieval Layer
    try:
        context = retrieve(query)
    except Exception as e:
        print(f"[DEBUG] Retrieval failed: {e}")
        context = "" 

    # Step 2: Generation (Groq only)
    user_message = f"Context:\n{context}\n\nUser Question: {query}" if context.strip() else f"(No context found)\n\nUser Question: {query}"

    try:
        print("[DEBUG] Attempting Groq (Llama 3.3)...")
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",
            timeout=15.0, 
            temperature=0.1,
            max_tokens=200 
        )
        answer = chat_completion.choices[0].message.content.strip()

    except Exception as groq_err:
        print(f"[ERROR] Groq failed: {repr(groq_err)}")
        return "I'm having trouble connecting to our AI system right now. Please try again in a moment."

    # --- STEP 3: POST-PROCESSING (Cleanup) ---
    final_answer = answer.replace("- ", "").replace("* ", "").replace("Certainly,", "").replace("Absolutely,", "")
    
    if final_answer and final_answer[-1] not in ['.', '!', '?']:
        last_period = final_answer.rfind('.')
        if last_period != -1:
            final_answer = final_answer[:last_period + 1]

    return final_answer.strip()
