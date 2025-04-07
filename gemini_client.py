import streamlit as st
import json
import google.generativeai as genai
import time
import re
from typing import Dict, List, Any, Optional

# Initialize Gemini API
def initialize_genai(api_key: Optional[str] = None):
    """Initialize the Gemini API with the provided key."""
    if api_key:
        genai.configure(api_key=api_key)

def create_gemini_prompt(query: str) -> str:
    """Create a structured prompt for Gemini to extract keywords from user query."""
    
    # Schema information for available options
    job_levels = [
        "Director", "Entry-Level", "Executive", "Front Line Manager", "General Population", 
        "Graduate", "Manager", "Mid-Professional", "Professional Individual Contributor", "Supervisor"
    ]
    
    test_types = [
        "Ability & Aptitude", "Biodata & Situational Judgement", "Competencies", 
        "Development & 360", "Assessment Exercises", "Knowledge & Skills", 
        "Personality & Behavior", "Simulations"
    ]
    
    job_levels_str = json.dumps(job_levels, indent=2)
    test_types_str = json.dumps(test_types, indent=2)
    
    return f"""
    You are an agentic assistant specialized in translating natural language job requirements into structured search parameters for an SHL assessment recommendation system.
    
    Analyze the following query and extract the key information that would help search a database of SHL assessments.
    
    Query: "{query}"
    
    Here are the available job levels in our system:
    {job_levels_str}
    
    Here are the available test type categories:
    {test_types_str}
    
    Based on the information provided, return a JSON object with these fields:
    
    {{
        "job_levels": [
            // List job levels from the provided options that match the query
            // Map similar terms to our standard terminology
        ],
        "skills": [
            // List technical skills mentioned (e.g. "Java", "Python", "SQL")
            // Standardize terminology where possible
        ],
        "duration": null, // Maximum duration in minutes (convert from hours if needed)
        "remote_testing": null, // true if remote testing is required, null if not mentioned
        "adaptive_irt": null, // true if adaptive/IRT support is required, null if not mentioned
        "test_types": [
            // List test types from our categories that match the query
        ],
        "languages": [
            // List languages mentioned or required
        ],
        "extracted_url": null // Extract any job description URL if present in the query
    }}
    
    Rules:
    1. Duration must be numeric (in minutes). Convert phrases like "an hour" to 60, "30 minutes" to 30.
    2. Set boolean values to true only when explicitly mentioned, otherwise null.
    3. Only include information that is directly stated or strongly implied.
    4. Map concepts to our standard terminology where possible.
    5. If a URL is found in the query, extract it into the extracted_url field.
    6. The original_query field should contain the full original query.
    """

def create_dynamic_weights_prompt(query: str, extracted_info: Dict) -> str:
    """Create a prompt for Gemini to determine dynamic weights based on the query."""
    
    extracted_info_str = json.dumps(extracted_info, indent=2)
    
    return f"""
    You are an AI assistant specialized in understanding hiring requirements and assessment priorities.
    
    Analyze the following query about job assessment needs:
    
    "{query}"
    
    I have extracted the following information from the query:
    {extracted_info_str}
    
    Based on this query and extracted information, determine the relative importance (weights) of different criteria for finding relevant assessments.
    
    Return a JSON object with the following structure:
    {{
        "weights": {{
            "skills": 0.0-5.0,  // Importance of technical skills matching
            "job_levels": 0.0-5.0,  // Importance of job level matching
            "duration": 0.0-5.0,  // Importance of meeting duration constraints
            "remote_testing": 0.0-5.0,  // Importance of remote testing capability
            "adaptive_irt": 0.0-5.0,  // Importance of adaptive testing
            "test_types": 0.0-5.0,  // Importance of specific test types (personality, knowledge, etc.)
            "languages": 0.0-5.0,  // Importance of language support
            "semantic": 0.0-5.0  // Importance of overall semantic similarity
        }},
        "explanation": "Brief explanation of why you assigned these weights based on the query"
    }}
    
    Rules:
    1. Assign higher weights (4.0-5.0) to criteria explicitly mentioned or clearly important in the query
    2. Assign medium weights (2.0-3.0) to criteria implied but not explicitly stated
    3. Assign lower weights (0.0-1.0) to criteria not mentioned or irrelevant to the query
    4. The weights should reflect the priorities expressed in the query
    5. Return only the JSON object with no additional explanation
    """

# Process user query with Gemini
def process_query_with_gemini(query: str, api_key: str) -> Dict:
    """
    Use Gemini to extract relevant keywords and parameters from the user query.
    """
    try:
        initialize_genai(api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Add original query to the extracted info
        prompt = create_gemini_prompt(query)
        response = model.generate_content(prompt)
        
        # Parse the response as JSON
        try:
            extracted_info = json.loads(response.text)
            # Add original query
            extracted_info["original_query"] = query
            return extracted_info
        except json.JSONDecodeError:
            # If response isn't valid JSON, try to extract JSON part
            json_match = re.search(r'({[\s\S]*})', response.text)
            if json_match:
                extracted_info = json.loads(json_match.group(1))
                extracted_info["original_query"] = query
                return extracted_info
            else:
                raise ValueError("Could not parse Gemini response as JSON")
    
    except Exception as e:
        st.error(f"Error processing query with Gemini: {e}")
        return {
            "job_levels": [],
            "skills": [],
            "duration": None,
            "remote_testing": None,
            "adaptive_irt": None,
            "test_types": [],
            "languages": [],
            "extracted_url": None,
            "original_query": query
        }

def determine_dynamic_weights(query: str, extracted_info: Dict, api_key: str) -> Dict:
    """
    Use Gemini to determine dynamic weights based on the query and extracted information.
    """
    try:
        initialize_genai(api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = create_dynamic_weights_prompt(query, extracted_info)
        response = model.generate_content(prompt)
        
        # Parse the response as JSON
        try:
            weights_info = json.loads(response.text)
            weights = weights_info.get("weights", {})
            # Update session state with new weights
            st.session_state.weights = weights
            return weights
        except json.JSONDecodeError:
            # If response isn't valid JSON, try to extract JSON part
            json_match = re.search(r'({[\s\S]*})', response.text)
            if json_match:
                weights_info = json.loads(json_match.group(1))
                weights = weights_info.get("weights", {})
                # Update session state with new weights
                st.session_state.weights = weights
                return weights
            else:
                raise ValueError("Could not parse weights from Gemini response")
    
    except Exception as e:
        st.error(f"Error determining dynamic weights: {e}")
        # Default weights
        default_weights = {
            "skills": 3.0,
            "job_levels": 2.0,
            "duration": 2.0,
            "remote_testing": 1.0,
            "adaptive_irt": 1.0,
            "test_types": 3.0,
            "languages": 1.0,
            "semantic": 2.0
        }
        # Update session state with default weights
        st.session_state.weights = default_weights
        return default_weights

def create_embeddings(texts: List[str], api_key: str, task_type: str) -> List[List[float]]:
    try:
        initialize_genai(api_key)
        embeddings = []
        
        # Create embeddings in batches with smaller batch size
        batch_size = 3  # Reduced from 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    truncated_text = text[:5000]  # Limit text length
                    response = genai.embed_content(
                        model="models/gemini-embedding-exp-03-07",
                        content=truncated_text,
                        task_type=task_type,
                    )
                    batch_embeddings.append(response["embedding"])
                except Exception as e:
                    st.warning(f"Skipping embedding for one item: {str(e)}")
                    # Add empty embedding as placeholder
                    batch_embeddings.append([0.0] * 768)  # Common embedding dimension
            
            embeddings.extend(batch_embeddings)
            
            # Add delay between batches
            time.sleep(1)
        
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        # Return empty embeddings with correct dimensions
        return [[0.0] * 768 for _ in texts]  # Use standard dimension