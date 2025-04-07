import re
import requests
from typing import Optional, Dict
from bs4 import BeautifulSoup

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

# Schema information for available options - moved from app.py
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

def get_default_weights() -> Dict[str, float]:
    """Return default weights for search criteria."""
    return {
        "skills": 3.0,
        "job_levels": 2.0,
        "duration": 2.0,
        "remote_testing": 1.0,
        "adaptive_irt": 1.0,
        "test_types": 3.0,
        "languages": 1.0,
        "semantic": 2.0
    }

def format_results_for_display(individual_results, pre_packaged_results):
    """Format search results for consistent display across UI functions."""
    return {
        "individual": individual_results,
        "pre_packaged": pre_packaged_results
    }