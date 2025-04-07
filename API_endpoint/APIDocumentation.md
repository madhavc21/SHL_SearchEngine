# SHL Assessment Recommender API Documentation

This API recommends SHL assessments based on job descriptions or natural language queries, leveraging Google's Gemini AI to extract relevant parameters and match them with appropriate assessment solutions.

## Overview

The SHL Assessment Recommender API analyzes job descriptions or queries to recommend both individual assessments and pre-packaged solutions from SHL's catalog. It extracts key parameters like job levels, skills, and test types to provide tailored recommendations.

## Installation and Setup

### Prerequisites

- Python 3.7+, 3.11 recommended
- FastAPI
- Uvicorn
- Google Generative AI Python SDK
- Beautiful Soup 4
- Requests

### Installation Steps

1. Clone the repository or create a new directory for your project:

```bash
https://github.com/madhavc21/SHL_SearchEngine.git
cd SHL_SearchEngine/API_endpoint
```

2. Install the required dependencies:

```bash
pip install fastapi uvicorn[standard] pydantic google-generativeai beautifulsoup4 requests
```

3. Save the `fastapi_endpoint.py` file in your project directory.

4. Ensure the JSON file named `shl_solutions.json` in the same directory as the endpoint. This is file is created using scraper.py, and contains the scraper, structured database of SHL product catalogue. The file should have the following structure:

```json
{
  "individual_test_solutions": {
    "test_id_1": {
      "name": "Test Name",
      "url": "https://example.com/test",
      "description": "Test description",
      "job_levels": ["Professional Individual Contributor", "Manager"],
      "remote_testing": true,
      "adaptive_irt": false,
      "duration": 30,
      "test_type": "P",
      "test_type_descriptions": ["Personality & Behavior"],
      "languages": ["English", "Spanish"]
    },
    // Additional tests...
  },
  "pre_packaged_job_solutions": {
    "solution_id_1": {
      "name": "Solution Package Name",
      "url": "https://example.com/solution",
      "description": "Solution package description",
      "job_levels": ["Manager", "Director"],
      "remote_testing": true,
      "adaptive_irt": true,
      "duration": 60,
      "test_type": "A,P",
      "test_type_descriptions": ["Ability & Aptitude", "Personality & Behavior"],
      "languages": ["English", "French", "German"]
    },
    // Additional solutions...
  }
}
```

## Running the API

Start the API server using Uvicorn:

```bash
uvicorn fastapi_endpoint:app --reload
```

By default, the API will run on http://127.0.0.1:8000.

## API Endpoints

### GET /recommend

Get assessment recommendations based on a natural language query or job description.

#### Parameters

- `query` (string, required): Natural language query describing job requirements or URL to job description
- `max_results` (integer, optional, default=10): Maximum number of recommendations to return (1-10)
- `api_key` (string, required): Gemini API Key for processing

#### Response

```json
{
  "individual_recommendations": [
    {
      "name": "Assessment Name",
      "url": "https://example.com/assessment",
      "description": "Assessment description",
      "remote_testing": true,
      "adaptive_irt": false,
      "duration": "30",
      "test_type": "P",
      "relevance_score": 4.75
    }
  ],
  "pre_packaged_recommendations": [
    {
      "name": "Package Name",
      "url": "https://example.com/package",
      "description": "Package description",
      "remote_testing": true,
      "adaptive_irt": true,
      "duration": "60",
      "test_type": "A,P",
      "relevance_score": 3.85
    }
  ],
  "extracted_parameters": {
    "job_levels": ["Manager", "Professional Individual Contributor"],
    "skills": ["Python", "JavaScript", "Communication"],
    "duration": 45,
    "remote_testing": true,
    "adaptive_irt": null,
    "test_types": ["Ability & Aptitude", "Personality & Behavior"],
    "languages": ["English"],
    "extracted_url": "https://example.com/job-posting"
  }
}
```

### GET /health

Simple health check endpoint to verify the API is running.

#### Response

```json
{
  "status": "ok",
  "message": "API is running"
}
```

## Testing the API with Swagger UI

FastAPI automatically generates interactive API documentation using Swagger UI.

1. Start the API server as described above.
2. Open a web browser and navigate to http://127.0.0.1:8000/docs
3. You'll see the Swagger UI interface listing all available endpoints.
4. To test the `/recommend` endpoint:
   - Click on the endpoint to expand it
   - Click "Try it out"
   - Enter a natural language query (e.g., "Looking for aptitude tests for a senior software developer position that can be taken remotely")
   - Enter your Gemini API key
   - Optionally adjust the max_results parameter
   - Click "Execute"
   - View the response

## Example Queries

Here are some examples to test the API:

1. Simple job role query:
   ```
   "Recommend assessments for a mid-level project manager position"
   ```

2. Detailed requirements:
   ```
   "Looking for personality and aptitude tests for a senior software developer position with Python and JavaScript skills, that can be taken remotely and shouldn't take more than 45 minutes"
   ```

3. With a job posting URL:
   ```
   "Please analyze this job posting and recommend suitable assessments: https://example.com/job-posting"
   ```

## Notes

- You need a valid Google Gemini API key to use this service
- The API will attempt to scrape job descriptions from URLs included in the query
- The recommendations are ranked by relevance score based on how well they match the extracted parameters
- Dynamic weights are applied to different criteria based on the query's context
