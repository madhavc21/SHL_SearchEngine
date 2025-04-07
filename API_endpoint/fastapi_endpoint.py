# api.py

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions or natural language queries",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema information for available options
SCHEMA_INFO = {
    "job_levels": {
        "Director": "Senior leadership role responsible for a business unit or function",
        "Entry-Level": "Minimal experience required, typically first professional role",
        "Executive": "C-level or highest leadership positions in an organization",
        "Front Line Manager": "First level of management, directly supervising individual contributors",
        "General Population": "Roles across various levels and functions",
        "Graduate": "Recent college/university graduates",
        "Manager": "Mid-level management overseeing teams or departments",
        "Mid-Professional": "Experienced professional with several years in the field",
        "Professional Individual Contributor": "Skilled professional working independently",
        "Supervisor": "Team leader with operational oversight responsibilities"
    },
    "test_types": {
        "Ability & Aptitude": "Measures cognitive abilities and potential",
        "Biodata & Situational Judgement": "Assesses past experiences and judgment in workplace scenarios",
        "Competencies": "Evaluates specific skills and abilities for job performance",
        "Development & 360": "Focuses on growth areas and feedback from multiple sources",
        "Assessment Exercises": "Practical tasks mimicking job responsibilities",
        "Knowledge & Skills": "Tests specific technical knowledge and capabilities",
        "Personality & Behavior": "Examines work style, preferences, and behavioral tendencies",
        "Simulations": "Interactive scenarios that mirror job challenges"
    },
    "test_type_mappings": {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations"
    }
}

# Response model for recommendations
class RecommendationResponse(BaseModel):
    individual_recommendations: List[Dict[str, Any]]
    pre_packaged_recommendations: List[Dict[str, Any]]
    extracted_parameters: Dict[str, Any]

# Initialize Gemini API
def initialize_genai(api_key: Optional[str] = None):
    """Initialize the Gemini API with the provided key."""
    if api_key:
        genai.configure(api_key=api_key)

# Load JSON data
def load_data(file_path: str = "shl_solutions.json") -> Dict:
    """Load the SHL solutions data."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def detect_url(text: str) -> Optional[str]:
    """Detect and extract a URL from text if present."""
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, text)
    if match:
        return match.group(0)
    return None

def scrape_job_description(url: str) -> str:
    """Scrape job description content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try common selectors for job descriptions
        selectors = [
            'div.job-description',
            'div.description',
            'section.job-description',
            'div.jobsearch-jobDescriptionText',
            'div#jobDescriptionText',
            'div.job-details',
            'article'
        ]
        
        content = ""
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(strip=True)
                break
        
        # If no matching selectors, get main content
        if not content:
            content = soup.get_text()
            # Basic cleaning
            content = re.sub(r'\s+', ' ', content).strip()
            # Limit to reasonable size
            content = content[:5000]
        
        return content
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def create_gemini_prompt(query: str) -> str:
    """Create a structured prompt for Gemini to extract keywords from user query."""
    
    job_levels_str = json.dumps(list(SCHEMA_INFO["job_levels"].keys()), indent=2)
    test_types_str = json.dumps(list(SCHEMA_INFO["test_types"].keys()), indent=2)
    
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
            "languages": 0.0-5.0  // Importance of language support
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
        raise HTTPException(status_code=500, detail=f"Error processing query with Gemini: {str(e)}")

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
            return weights
        except json.JSONDecodeError:
            # If response isn't valid JSON, try to extract JSON part
            json_match = re.search(r'({[\s\S]*})', response.text)
            if json_match:
                weights_info = json.loads(json_match.group(1))
                weights = weights_info.get("weights", {})
                return weights
            else:
                raise ValueError("Could not parse weights from Gemini response")
    
    except Exception as e:
        # Default weights
        default_weights = {
            "skills": 3.0,
            "job_levels": 2.0,
            "duration": 2.0,
            "remote_testing": 1.0,
            "adaptive_irt": 1.0,
            "test_types": 3.0,
            "languages": 1.0
        }
        return default_weights

def calculate_criterion_score(test: Dict, extracted_info: Dict, criterion: str):
    """Calculate score for a specific criterion."""
    
    if criterion == "skills":
        # More sophisticated skill matching
        skills = set([skill.lower() for skill in extracted_info.get("skills", [])])
        description = test.get("description", "").lower()
        
        skill_score = 0
        matching_skills = []
        
        for skill in skills:
            if not skill:
                continue
                
            # Exact match
            if skill in description:
                skill_score += 1
                matching_skills.append(skill)
            
            # Partial match for compound skills (e.g. "javascript" in "javascript framework")
            elif any(skill in term for term in description.split()):
                skill_score += 0.5
                matching_skills.append(f"{skill} (partial)")
                
        return skill_score, matching_skills
        
    elif criterion == "job_levels":
        test_levels = set([level.lower() for level in test.get("job_levels", []) if level])
        query_levels = set([level.lower() for level in extracted_info.get("job_levels", []) if level])
        
        # Get intersection of levels
        matching_levels = test_levels.intersection(query_levels)
        
        # Return number of matching levels and the matches
        return len(matching_levels), list(matching_levels)
        
    elif criterion == "duration":
        max_duration = extracted_info.get("duration")
        
        try:
            test_duration = int(test.get("duration", 0)) if test.get("duration") else 0
        except (ValueError, TypeError):
            test_duration = 0
        
        if max_duration and test_duration:
            if test_duration <= max_duration:
                # Score inversely proportional to duration (shorter is better within max)
                score = 1 + (max_duration - test_duration) / max_duration
                return score, f"{test_duration} mins (within {max_duration} min limit)"
            else:
                # Penalize for exceeding time limit, but not severely
                overrun_penalty = min(1, (test_duration - max_duration) / max_duration)
                score = max(0, 1 - overrun_penalty)
                return score, f"{test_duration} mins (exceeds {max_duration} min limit)"
        
        return 0, None
        
    elif criterion in ["remote_testing", "adaptive_irt"]:
        # Boolean match
        query_value = extracted_info.get(criterion)
        test_value = test.get(criterion)
        
        if query_value is not None and test_value == query_value:
            return 1, f"Matched: {query_value}"
        elif query_value is not None:
            return 0, f"Not matched: wanted {query_value}, got {test_value}"
        
        return 0, None
        
    elif criterion == "test_types":
        query_types = set([t.lower() for t in extracted_info.get("test_types", [])])
        test_types = set([t.lower() for t in test.get("test_type_descriptions", [])])
        
        # Get intersection of test types
        matching_types = query_types.intersection(test_types)
        
        # Return proportion of matching types and the matches
        if query_types:
            score = len(matching_types) / len(query_types)
            return score, list(matching_types)
        
        return 0, None
        
    elif criterion == "languages":
        query_langs = set([lang.lower() for lang in extracted_info.get("languages", [])])
        test_langs = set([lang.lower() for lang in test.get("languages", [])])
        
        # Get intersection of languages
        matching_langs = query_langs.intersection(test_langs)
        
        # Return proportion of matching languages and the matches
        if query_langs:
            score = len(matching_langs) / len(query_langs)
            return score, list(matching_langs)
        
        return 0, None
        
    return 0, None

def search_assessments(
    data: Dict, 
    extracted_info: Dict,
    dynamic_weights: Dict[str, float],
    max_results: int = 10
):
    """
    Search for relevant assessments using criteria matching.
    Returns results for both individual tests and pre-packaged solutions.
    """
    individual_tests = data.get("individual_test_solutions", {})
    pre_packaged_solutions = data.get("pre_packaged_job_solutions", {})
    individual_results = []
    pre_packaged_results = []
    
    # Process individual tests
    for test_id, test in individual_tests.items():
        score = 0
        score_details = {}
        
        # Apply scoring for each criterion
        criteria = [
            "skills", "job_levels", "duration", "remote_testing", 
            "adaptive_irt", "test_types", "languages"
        ]
        
        for criterion in criteria:
            criterion_raw_score, matches = calculate_criterion_score(test, extracted_info, criterion)
            criterion_weight = dynamic_weights.get(criterion, 1.0)
            weighted_score = criterion_raw_score * criterion_weight
            
            score += weighted_score
            
            score_details[criterion] = {
                "raw_score": criterion_raw_score,
                "weighted_score": weighted_score,
                "matches": matches,
                "weight": criterion_weight
            }
        
        # Add the assessment with its score if it has any relevance
        if score > 0:
            individual_results.append({
                **test,
                "relevance_score": score,
                "score_details": score_details
            })
    
    # Process pre-packaged solutions
    for solution_id, solution in pre_packaged_solutions.items():
        score = 0
        score_details = {}
        
        # Apply scoring for each criterion
        criteria = [
            "skills", "job_levels", "duration", "remote_testing", 
            "adaptive_irt", "test_types", "languages"
        ]
        
        for criterion in criteria:
            criterion_raw_score, matches = calculate_criterion_score(solution, extracted_info, criterion)
            criterion_weight = dynamic_weights.get(criterion, 1.0)
            weighted_score = criterion_raw_score * criterion_weight
            
            score += weighted_score
            
            score_details[criterion] = {
                "raw_score": criterion_raw_score,
                "weighted_score": weighted_score,
                "matches": matches,
                "weight": criterion_weight
            }
        
        # Add the solution with its score if it has any relevance
        if score > 0:
            pre_packaged_results.append({
                **solution,
                "relevance_score": score,
                "score_details": score_details
            })
    
    # Sort by relevance score and limit results
    individual_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    pre_packaged_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return individual_results[:max_results], pre_packaged_results[:max_results], extracted_info

def process_complete_query(query: str, api_key: str):
    """
    Process the complete query including URL extraction and dynamic weight determination.
    """
    # Extract URL if present
    url = detect_url(query)
    enhanced_query = query
    
    if url:
        try:
            job_description = scrape_job_description(url)
            # Combine original query with job description
            enhanced_query = f"{query}\n\nJob Description from URL: {job_description}"
        except Exception:
            pass
    
    # Extract information from query
    extracted_info = process_query_with_gemini(enhanced_query, api_key)
    
    # Determine dynamic weights
    dynamic_weights = determine_dynamic_weights(enhanced_query, extracted_info, api_key)
    
    return extracted_info, dynamic_weights

@app.get("/recommend", response_model=RecommendationResponse, tags=["recommendations"])
async def get_recommendations(
    query: str = Query(..., description="Natural language query or job description text/URL"),
    max_results: int = Query(10, ge=1, le=10, description="Maximum number of recommendations to return"),
    api_key: str = Query(..., description="Gemini API Key")
):
    """
    Get assessment recommendations based on a natural language query or job description.
    
    - **query**: Natural language query describing job requirements or URL to job description
    - **max_results**: Maximum number of recommendations to return (1-10)
    - **api_key**: Gemini API Key for processing
    
    Returns lists of recommended individual assessments and pre-packaged solutions with relevance scores and extracted parameters.
    """
    try:
        # Load data
        data = load_data()
        
        # Process the query
        extracted_info, dynamic_weights = process_complete_query(query, api_key)
        
        # Search for assessments
        individual_recommendations, pre_packaged_recommendations, _ = search_assessments(
            data,
            extracted_info,
            dynamic_weights,
            max_results=max_results
        )
        
        # Prepare output for individual tests
        clean_individual_recommendations = []
        for rec in individual_recommendations:
            # Remove score details and other unnecessary fields for API response
            clean_rec = {
                "name": rec.get("name", ""),
                "url": rec.get("url", ""),
                "description": rec.get("description", ""),
                "remote_testing": rec.get("remote_testing", False),
                "adaptive_irt": rec.get("adaptive_irt", False),
                "duration": rec.get("duration", ""),
                "test_type": rec.get("test_type", ""),
                "relevance_score": round(rec.get("relevance_score", 0), 2)
            }
            clean_individual_recommendations.append(clean_rec)
        
        # Prepare output for pre-packaged solutions
        clean_pre_packaged_recommendations = []
        for rec in pre_packaged_recommendations:
            # Remove score details and other unnecessary fields for API response
            clean_rec = {
                "name": rec.get("name", ""),
                "url": rec.get("url", ""),
                "description": rec.get("description", ""),
                "remote_testing": rec.get("remote_testing", False),
                "adaptive_irt": rec.get("adaptive_irt", False),
                "duration": rec.get("duration", ""),
                "test_type": rec.get("test_type", ""),
                "relevance_score": round(rec.get("relevance_score", 0), 2)
            }
            clean_pre_packaged_recommendations.append(clean_rec)
        
        # Clean extracted parameters
        clean_params = {
            "job_levels": extracted_info.get("job_levels", []),
            "skills": extracted_info.get("skills", []),
            "duration": extracted_info.get("duration"),
            "remote_testing": extracted_info.get("remote_testing"),
            "adaptive_irt": extracted_info.get("adaptive_irt"),
            "test_types": extracted_info.get("test_types", []),
            "languages": extracted_info.get("languages", []),
            "extracted_url": extracted_info.get("extracted_url")
        }
        
        return {
            "individual_recommendations": clean_individual_recommendations,
            "pre_packaged_recommendations": clean_pre_packaged_recommendations,
            "extracted_parameters": clean_params
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")

@app.get("/health", tags=["health"])
async def health_check():
    """
    Simple health check endpoint.
    
    Returns a status message indicating the API is running.
    """
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)