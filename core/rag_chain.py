"""
RAG Chain Module - Hybrid mode using AI-based routing
"""

import requests
import json
from config.config import QWEN_API_URL, QWEN_API_KEY
from core.milvus_utilis import collection, search_similar_chunks
import re

# Send a custom prompt to the LLM
def ask_llm_with_context_custom_prompt(prompt: str) -> str:
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. You must only use the provided document content to answer."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    # Ensure QWEN_API_URL is set
    if not QWEN_API_URL:
        raise ValueError("QWEN_API_URL is not configured in environment variables")
    response = requests.post(QWEN_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Semantic search using document chunks
def ask_llm_with_context(query: str) -> str:
    try:
        results = search_similar_chunks(query, top_k=1000)  # Reduced from 1000 to 10
        if not results:
            return "I couldn't find any relevant information in the documents for your question."
        
        # Limit context size to prevent API errors
        context_parts = []
        total_length = 0
        max_context_length = 8000  # Limit context to prevent API errors
        
        for r in results:
            chunk_text = r["chunk"]
            if total_length + len(chunk_text) < max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                break
        
        context = "\n".join(context_parts)

        prompt = f"""
Based on the following document content, answer this question: {query}

Document content:
{context}

Answer the question clearly and concisely using only the information provided above. Do NOT show your reasoning or thought process. Only output the final answer, with no explanation or extra commentary.
"""
        return ask_llm_with_context_custom_prompt(prompt)
        
    except Exception as e:
        print(f"⚠️ Error in semantic search: {e}")
        return f"Sorry, I encountered an error while searching for information: {str(e)}"


# Load all document chunks for full-context answers
def load_all_chunks_for_context(limit: int = 10) -> str:
    collection.load()
    results = collection.query(
        expr="",
        output_fields=["chunk"],
        limit=limit
    )
    chunks = [r["chunk"] for r in results]
    return "\n".join(chunks)
 

# Use all document data for creative or vague queries
def ask_with_full_context(query: str) -> str:
    full_data = load_all_chunks_for_context(limit=1000)

    prompt = f"""
You are a curious, creative AI assistant.

Below is a collection of document excerpts from various articles and reports:

{full_data}

Now, based on this user question: "{query}", please extract and explain something surprisingly interesting, insightful, or fun. Be creative. Do not repeat the same structure every time. Express it in a friendly and engaging tone.
"""
    return ask_llm_with_context_custom_prompt(prompt)

def llm_rephrase_with_history(query: str, conversation_context: str) -> str:
    """Use the LLM to rephrase a vague question using the full conversation history, or ask for clarification if it can't."""
    if not conversation_context:
        return query
    prompt = f"""
You are an AI assistant. Here is the conversation history:
{conversation_context}

The user just asked: "{query}"

Your task:
- If you can infer what the user is referring to, rephrase their question as specifically as possible, using the context above.
- If you cannot confidently rephrase, respond with only: CLARIFY

Rephrased question (or CLARIFY):
"""
    try:
        rephrased = ask_llm_with_context_custom_prompt(prompt)
        rephrased = rephrased.strip()
        return rephrased
    except Exception as e:
        print(f"⚠️ LLM rephrase error: {e}")
        return query

# Main smart routing entry point
# Decides which function to use based on the query
def ask_question_smart_with_toolcall(query: str, conversation_context: str = "") -> str:
    # Use LLM to rephrase vague questions using history
    clarified_query = llm_rephrase_with_history(query, conversation_context)
    if clarified_query.strip().upper() == "CLARIFY":
        return (
            "🤔 I'm not sure what you're referring to. Could you please clarify your question or provide more details? "
            "For example, mention the specific topic, document, or process you want to know more about."
        )
    # Continue as before, but use clarified_query
    decision_prompt = f"""
You are an AI assistant that helps with labor law and employment rule questions using document knowledge.

IMPORTANT: You must choose between two options based on the question type.

QUESTION: "{clarified_query}"

OPTIONS:
1. SEMANTIC_SEARCH - Use ONLY for specific labor law or employment questions about rights, contracts, violations, regulations, processes, eligibility, etc.
2. ASK_SPECIFIC - Use for general questions about capabilities, vague questions, or questions not about specific labor law topics

EXAMPLES:
SEMANTIC_SEARCH questions:
- "What are the legal requirements for overtime pay?"
- "How do I file a workplace harassment complaint?"
- "What are the rules for employee termination?"
- "What are the eligibility criteria for maternity leave?"

ASK_SPECIFIC questions:
- "What can you do?"
- "Tell me everything about labor law"
- "What do you know?"
- "How do you work?"
- "What are your capabilities?"

CRITICAL: If the question is about what you can do, your capabilities, or is very general/vague, choose ASK_SPECIFIC.

Respond with ONLY: SEMANTIC_SEARCH or ASK_SPECIFIC
"""

    try:
        # Get the decision
        decision_response = ask_llm_with_context_custom_prompt(decision_prompt)
        print(f"🔍 Raw decision response: '{decision_response}'")
        
        # Clean up the response - remove any thinking tags or extra text
        decision = decision_response.strip()
        # Remove thinking tags and their content
        decision = re.sub(r'<think>.*?</think>', '', decision, flags=re.DOTALL)
        decision = re.sub(r'<THINK>.*?</THINK>', '', decision, flags=re.DOTALL)
        # Clean up whitespace and get the final decision
        decision = decision.strip().upper()
        
        # Extract the actual decision
        if 'SEMANTIC_SEARCH' in decision:
            decision = 'SEMANTIC_SEARCH'
        elif 'ASK_SPECIFIC' in decision:
            decision = 'ASK_SPECIFIC'
        
        print(f"🔧 AI chose: {decision}")
        
        # Route based on decision
        if decision == "SEMANTIC_SEARCH":
            try:
                return ask_llm_with_context(clarified_query)
            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}")
                return f"Sorry, I encountered an error while searching for information: {str(e)}"
        elif decision == "ASK_SPECIFIC":
            if "what can you do" in clarified_query.lower() or "capabilities" in clarified_query.lower():
                # Give the capabilities response
                return (
                    "🤖 I'm an AI assistant specialized in US labor law! I can help you with:\n\n"
                    "📋 **Document Processing**: I can analyze labor law contract, policy documents, and legal materials\n"
                    "❓ **Specific Questions**: Ask me about naturalization requirements, green card processes, form instructions, etc.\n"
                    "🔍 **Document Search**: I can search through uploaded documents for specific information\n\n"
                    "💡 **Try asking specific questions like**:\n"
                    "• 'What are the requirements for naturalization?'\n"
                    "• 'How long does the green card process take?'\n"
                    "• 'What forms do I need for citizenship?'\n"
                    "• 'What are the eligibility criteria for DACA?'\n\n"
                    "📚 **Upload Documents**: Use 'upload <filename>' to add labor documents for me to analyze\n"
                    "🔍 **Search Documents**: Use 'search <query>' to find specific information in your documents\n\n"
                    "What specific labor question can I help you with today?"
                )
            else:
                if conversation_context:
                    return ask_llm_with_context_custom_prompt(
                        f"""You are a helpful AI assistant with access to conversation history.

CONVERSATION HISTORY:
{conversation_context}

CURRENT QUESTION: "{clarified_query}"

INSTRUCTIONS:
- If the current question can be answered using information from the conversation history, do so directly
- If the question is about something mentioned in the history, reference that information
- If the question is too vague or not related to the history, politely ask for more specificity
- Be conversational and helpful
- Use the conversation context to provide relevant answers

Please answer the current question based on the conversation history above."""
                    )
                else:
                    return (
                        "🤔 Your question seems quite broad. To give you a more helpful and focused answer, "
                        "could you please be more specific? For example:\n\n"
                        "• Instead of 'Tell me about labor', try 'What should I notice when applying for a job?'\n\n"
                        "This will help me find the most relevant information from your documents and give you a precise answer!"
                    )
        else:
            try:
                return ask_llm_with_context(clarified_query)
            except Exception as e:
                print(f"⚠️ Default search failed: {e}")
                return f"Sorry, I encountered an error: {str(e)}"
            
    except Exception as e:
        print(f"⚠️ Decision making error: {e}")
        try:
            return ask_llm_with_context(clarified_query)
        except Exception as e2:
            print(f"⚠️ Fallback search failed: {e2}")
            return f"Sorry, I encountered an error while processing your question: {str(e2)}"