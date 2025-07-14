import os
import re
import requests
from pathlib import Path
from datetime import datetime
from config.config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL

def is_clear_violation(ai_response):
    """
    Check if the AI response indicates a clear violation.
    Returns False if 'NO CLEAR VIOLATION' appears anywhere in the response (case-insensitive), True otherwise.
    """
    cleaned = ai_response.strip().strip('`').strip()
    # Remove 'FINAL ANSWER:' prefix if present
    if cleaned.upper().startswith("FINAL ANSWER:"):
        cleaned = cleaned[len("FINAL ANSWER:"):].strip()
    # Check for 'NO CLEAR VIOLATION' anywhere in the response (case-insensitive)
    return "NO CLEAR VIOLATION" not in cleaned.upper()

def generate_violation_summary(violations, contract_name):
    """
    Generate a summary of all violations using Qwen API.
    
    Args:
        violations (list): List of violation dictionaries
        contract_name (str): Name of the contract file
        
    Returns:
        str: Generated summary
    """
    if not violations:
        return "No violations found in this contract."
    
    # Prepare violation text for the AI
    violation_text = f"Contract: {contract_name}\n\n"
    violation_text += "The following violations were found:\n\n"
    
    for i, violation in enumerate(violations, 1):
        sentence, explanation = violation
        violation_text += f"{i}. Sentence: {sentence}\n"
        violation_text += f"   Explanation: {explanation}\n\n"
    
    # Create prompt for Qwen
    system_prompt = """
You are a legal compliance expert. Analyze the following contract violations and provide a comprehensive summary.

Your summary should include:
1. Total number of violations found
2. Most serious violations identified
3. Legal risks for the employer
4. Recommendations for compliance
5. Overall assessment of the contract's legality

Be concise but thorough. Focus on the most critical issues.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": violation_text}
    ]
    
    # Call Qwen API
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 10000
    }
    
    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Qwen API error: {response.status_code} - {response.text}")
    
    summary = response.json()['choices'][0]['message']['content'].strip()
    
    # Clean any <think> elements
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    summary = re.sub(r'<think>.*?', '', summary, flags=re.DOTALL)
    
    return summary 