import os
import google.genai as genai
from rag.retriever import retrieve
from rag.prompt import SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

# Configure Multiple Gemini API Keys with Failover
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

# Filter out None values
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]

# Backward compatibility: if no numbered keys, use the original GEMINI_API_KEY
if not GEMINI_API_KEYS:
    original_key = os.getenv("GEMINI_API_KEY")
    if original_key:
        GEMINI_API_KEYS = [original_key]
        print(f"[DEBUG] Using legacy GEMINI_API_KEY for backward compatibility")

print(f"[DEBUG] Loaded {len(GEMINI_API_KEYS)} Gemini API keys")

if not GEMINI_API_KEYS:
    raise RuntimeError("At least one GEMINI_API_KEY is required. Add GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, or GEMINI_API_KEY in environment variables.")

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

    # Step 2: Generation (Gemini with Failover)
    user_message = f"Context:\n{context}\n\nUser Question: {query}" if context.strip() else f"(No context found)\n\nUser Question: {query}"

    answer = None
    last_error = None

    # Try each API key in sequence
    for attempt, api_key in enumerate(GEMINI_API_KEYS, 1):
        try:
            print(f"[DEBUG] Attempting Gemini (API Key {attempt}/{len(GEMINI_API_KEYS)})...")
            
            # Create a Gemini client with the current API key
            client = genai.Client(api_key=api_key)
            
            # Create the full prompt with system instructions
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"
            
            # Generate response using the new google.genai API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )
            
            answer = response.text.strip()
            print(f"[DEBUG] Successfully generated response using API Key {attempt}")
            break
            
        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            
            # Check if it's a rate limit error - these should try next key
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                print(f"[ERROR] Gemini API Key {attempt} rate limited: {repr(e)}")
                if attempt < len(GEMINI_API_KEYS):
                    print(f"[DEBUG] Trying next API key...")
                    continue
                else:
                    print(f"[ERROR] All API keys rate limited")
            else:
                # Other errors (invalid key, network issues, etc.) - try next key
                print(f"[ERROR] Gemini API Key {attempt} failed (non-rate-limit): {repr(e)}")
                if attempt < len(GEMINI_API_KEYS):
                    print(f"[DEBUG] Trying next API key...")
                    continue
                else:
                    print(f"[ERROR] All API keys failed with errors")

    if answer is None:
        print(f"[ERROR] All Gemini API keys failed. Last error: {repr(last_error)}")
        return "I'm having trouble connecting to our AI system right now. Please try again in a moment."

    # --- STEP 3: POST-PROCESSING (Cleanup) ---
    final_answer = answer.replace("- ", "").replace("* ", "").replace("Certainly,", "").replace("Absolutely,", "")
    
    if final_answer and final_answer[-1] not in ['.', '!', '?']:
        last_period = final_answer.rfind('.')
        if last_period != -1:
            final_answer = final_answer[:last_period + 1]

    return final_answer.strip()
