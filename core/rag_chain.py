"""
Deep Search Model for Immigration QA
"""

from config.config import QWEN_API_URL, QWEN_API_KEY
from core.milvus_utilis import search_similar_chunks
import requests
import json
import re

# --- LLM API Call ---
def ask_llm(prompt: str) -> str:
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    if not QWEN_API_URL:
        raise ValueError("QWEN_API_URL is not configured in environment variables")
    response = requests.post(QWEN_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Immigration Domain Detection ---
def is_immigration_related(query: str, chat_history: str = "") -> bool:
    prompt = f"""
CHAT HISTORY:
{chat_history}

You are an expert classifier. Determine if the following user question is about US immigration or citizenship (including visas, green cards, naturalization, USCIS, etc).

QUESTION: "{query}"

Respond with ONLY YES or NO.
"""
    try:
        response = ask_llm(prompt)
        answer = response.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"⚠️ LLM classification error: {e}")
        return False

# --- Query Expansion ---
def query_expansion(query: str, chat_history: str = "") -> list:
    prompt = f"""
CHAT HISTORY:
{chat_history}

You are a query expansion expert. Your task is to understand the user's information needs and generate diverse search queries that will help find comprehensive answers.

Original query: {query}

# ANALYSIS PROCESS
First, analyze the query carefully:
1. What is the core information need behind this query?
2. What are the key entities and concepts in this query?
3. What are 5-7 DIFFERENT ASPECTS or angles of this topic that would be valuable to explore?
4. What related concepts would provide useful context for a complete answer?

# QUERY GENERATION INSTRUCTIONS
Based on your analysis, generate 5 search queries that:
1. EXPAND on the query with more specific details or broader context
2. Explore DIFFERENT FACETS of the same general topic
3. Include key entities from the query
4. Add relevant modifiers, related concepts, or specific aspects
5. Vary in scope (some narrower/focused, some broader/comprehensive)
6. Use different phrasing and vocabulary while maintaining meaning

# BALANCING FOCUS AND EXPANSION
- Each query should be clearly connected to the original topic
- Queries should be MEANINGFULLY DIFFERENT from each other
- Add relevant context, qualifiers, timeframes, or specificity
- Include both technical and practical perspectives when appropriate
- Queries should feel like they're exploring different angles of the same topic

# EXAMPLES

GOOD EXPANSION (diverse but related):
Original query: "new updates from Nvidia"
Expanded queries:
["latest Nvidia driver updates and performance improvements",
"Nvidia's recent hardware announcements and upcoming product releases",
"DLSS and ray tracing updates in Nvidia's newest software",
"Nvidia AI and machine learning framework updates 2025",
"professional graphics and workstation updates from Nvidia"]

BAD EXPANSION (too similar):
Original query: "new updates from Nvidia"
Expanded queries:
["new updates from Nvidia",
"recent updates from Nvidia",
"latest updates from Nvidia",
"current Nvidia updates",
"Nvidia's new updates"]

# RESPONSE FORMAT
Your entire response MUST be valid parseable JSON, starting with '[' and ending with ']'.
Do not include any text before or after the JSON array.

Example of CORRECT response format:
["query 1", "query 2", "query 3", "query 4", "query 5"]
"""
    try:
        response = ask_llm(prompt)
        response = response.strip()
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end+1]
            subquestions = json.loads(json_str)
            if isinstance(subquestions, list) and all(isinstance(q, str) for q in subquestions):
                return subquestions
        print("⚠️ LLM did not return a valid JSON list of strings. Returning original query as fallback.")
        return [query]
    except Exception as e:
        print(f"⚠️ Error in query_expansion: {e}")
        return [query]

# --- Semantic Search for Subquestions ---
def ask_llm_with_context(query: str, chat_history: str = "") -> str:
    results = search_similar_chunks(query, top_k=1000)
    if not results:
        return "No relevant information found."
    context = "\n".join([r["chunk"] for r in results])
    prompt = f"""
CHAT HISTORY:
{chat_history}

Based on the following document content, answer this question: {query}

Document content:
{context}

Answer the question clearly and concisely using only the information provided above.
"""
    return ask_llm(prompt)

# --- Quality Check for Answers ---
def check_answers_quality(questions: list, answers: list, original_query: str = "", iteration: int = 1, previous_knowledge_gaps: list = None, max_iterations: int = 3, chat_history: str = "") -> (bool, list):
    if previous_knowledge_gaps is None:
        previous_knowledge_gaps = []
    prompt = f"""
CHAT HISTORY:
{chat_history}

You are an expert immigration lawyer and legal reasoning agent. Your task is to analyze search results, identify relevant legal information, and determine if further research is needed to provide accurate immigration advice.

ORIGINAL QUERY: {original_query}

CURRENT SEARCH ITERATION: {iteration}

SEARCH RESULTS:
{json.dumps(answers, ensure_ascii=False)}

PREVIOUSLY IDENTIFIED LEGAL GAPS:
{json.dumps(previous_knowledge_gaps, ensure_ascii=False)}

INSTRUCTIONS:
1. Analyze the search results carefully to extract key legal information related to the original query.
2. Identify any NEW legal gaps or unresolved legal questions that require further research. Do NOT repeat previously identified legal gaps.
3. Decide if the research process should continue or if we have sufficient information to answer the query with sound legal advice.
4. If further research is needed, generate specific new research queries to fill the NEW legal gaps.
5. Format your response as a JSON object with the following structure:

{{
  "key_points": ["point 1", "point 2", "..."],
  "knowledge_gaps": ["gap 1", "gap 2", "..."],
  "new_queries": ["query 1", "query 2", "..."],
  "search_complete": true/false,
  "reasoning": "Your explanation of why the research is complete or needs to continue"
}}

CRITICAL: Your entire response MUST be a valid, parseable JSON object and nothing else. Do not include any text before or after the JSON object. Do not include any explanation, markdown formatting, or code blocks around the JSON. The response must start with '{{' and end with '}}' and contain only valid JSON.

If there are no legal gaps or the research should stop, return an empty array for "knowledge_gaps" and "new_queries" and set "search_complete" to true.

IMPORTANT: If this is already iteration {max_iterations} or higher, set "search_complete" to true regardless of legal gaps.
"""
    try:
        response = ask_llm(prompt)
        response = response.strip()
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end+1]
            result = json.loads(json_str)
            accepted = bool(result.get("search_complete", False))
            new_subquestions = result.get("new_queries", []) if not accepted else questions
            return accepted, new_subquestions
        print("⚠️ LLM did not return a valid JSON object. Accepting by default.")
        return True, questions
    except Exception as e:
        print(f"⚠️ Error in check_answers_quality: {e}")
        return True, questions

# --- Outline Generation ---
def write_outline(original_query: str, subquestions: list, answers: list, chat_history: str = "") -> str:
    search_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(subquestions, answers)])
    prompt = f"""
CHAT HISTORY:
{chat_history}

You are an expert research analyst and outline creator. Your task is to create a well-structured outline for answering a query based on search results.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

INSTRUCTIONS:
Your task is to formulate an OUTLINE ONLY for a complete answer with three distinct sections:

1. KEY POINTS: List 5-7 bullet points that would be the most important findings and facts
2. DIRECT ANSWER: Provide a brief description of what should be covered in the direct answer section (2-3 paragraphs)
3. DETAILED NOTES: Create a comprehensive outline with:
   a. Main section headings (3-5 sections)
   b. For each section, provide 2-3 sub-points that should be covered
   c. Note any specific technical details, examples, or comparisons that should be included
   d. Suggest logical flow for presenting the information

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly indicate which search results support each point using markdown hyperlinks
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation

Format your outline using proper markdown sections. THIS IS ONLY AN OUTLINE - do not write the full content.
Make the outline detailed enough that a content writer can easily expand it into a complete, informative answer.

The outline should follow this structure:
```
# OUTLINE: [Query Title]

## 1. KEY POINTS
- Key point 1 
- Key point 2 
...

## 2. DIRECT ANSWER
[Brief description of what the direct answer should cover]

## 3. DETAILED NOTES
### [Section Heading 1]
- Subpoint 1
- Subpoint 2
...

### [Section Heading 2]
- Subpoint 1
- Subpoint 2
...
```
"""
    try:
        outline = ask_llm(prompt)
        return outline.strip()
    except Exception as e:
        print(f"⚠️ Error in write_outline: {e}")
        return "(Outline unavailable due to error)"

# --- Final Answer Generation ---
def generate_final_answer(original_query: str, subquestions: list, answers: list, outline: str, chat_history: str = "") -> str:
    search_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(subquestions, answers)])
    prompt = f"""
CHAT HISTORY:
{chat_history}

You are an expert content writer. Your task is to expand an outline into a comprehensive, detailed answer.

ORIGINAL QUERY: {original_query}

SEARCH CONTEXT:
{search_context}

OUTLINE:
{outline}

INSTRUCTIONS:
Transform the provided outline into a comprehensive, detailed answer that follows the exact structure of the outline.
For each section:
1. Expand bullet points into detailed paragraphs with rich information
2. Maintain the hierarchical structure from the outline
3. Include technical details, examples, and comparisons suggested in the outline
4. Ensure smooth transitions between sections
5. Use an authoritative, clear writing style

IMPORTANT RULES:
1. ONLY include information that is directly supported by the search context
2. DO NOT make up or infer information not present in the search results
3. If information is missing or unclear, note it as a limitation rather than making assumptions
4. Clearly cite sources for each piece of information using the syntax: \\cite{{$ID}} for each fact or claim, for all key_points, reasoning and knowledge_gaps. Where the $ID is mentioned in the SEARCH DETAILS and OUTLINE.
5. Use direct quotes from search results when appropriate
6. Maintain academic rigor and avoid speculation
7. If the search context is insufficient to answer a point, clearly state this limitation
8. Do not use phrases like "based on the search results" or "according to the information provided" - instead cite specific sources

Your expanded answer should be thorough, informative, and directly address the original query,
while carefully following the outline structure and maintaining strict adherence to the search context.
"""
    try:
        answer = ask_llm(prompt)
        return answer.strip()
    except Exception as e:
        print(f"⚠️ Error in generate_final_answer: {e}")
        return outline + "\n\n" + "\n\n".join(answers)

def clean_llm_response(text: str) -> str:
    """Remove <think>...</think> and <THINK>...</THINK> tags and extra whitespace from LLM output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<THINK>.*?</THINK>', '', text, flags=re.DOTALL)
    return text.strip()

# --- Main Deep Search Pipeline ---
def deep_search_pipeline(query: str, chat_history: str = "") -> str:
    if not is_immigration_related(query, chat_history=chat_history):
        prompt = f"""
CHAT HISTORY:
{chat_history}

You are an expert immigration lawyer specialized in US immigration and citizenship. Your task is to give legal advice based on the original query.

USER QUERY: {query}

Please respond accordingly, if the user query is not related to immigration, please let them know.
"""
        direct_answer = ask_llm(prompt)
        return clean_llm_response(direct_answer)
    
    subquestions = query_expansion(query, chat_history=chat_history)
    answers = [None] * len(subquestions)
    max_iterations = 3
    previous_knowledge_gaps = []
    for i in range(max_iterations):
        answers = [ask_llm_with_context(q, chat_history=chat_history) for q in subquestions]
        accepted, new_subquestions = check_answers_quality(
            subquestions, answers, original_query=query, iteration=i + 1, previous_knowledge_gaps=previous_knowledge_gaps, max_iterations=max_iterations, chat_history=chat_history
        )
        if not accepted:
            previous_knowledge_gaps.extend([q for q in new_subquestions if q not in previous_knowledge_gaps])
            subquestions = new_subquestions
        if accepted:
            break
    outline = write_outline(query, subquestions, answers, chat_history=chat_history)
    final_answer = generate_final_answer(query, subquestions, answers, outline, chat_history=chat_history)
    return clean_llm_response(final_answer)
