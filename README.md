# SHL Assessment Recommendation System

This project implements an intelligent recommendation system to help hiring managers find the right SHL assessments for their job roles. The system takes a natural language query or job description text/URL as input and returns relevant SHL assessment recommendations.

## Features

- **Natural language processing** of hiring requirements and job descriptions
- **Dynamic scoring algorithm** that adapts to the priorities expressed in each query
- **Semantic search** using Gemini embeddings to find conceptually matching assessments
- **Web scraping** functionality to gather SHL assessment data
- **URL processing** to extract job descriptions from provided links
- **Detailed scoring breakdown** to explain why each assessment was recommended

## Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - streamlit
  - google-generativeai
  - scikit-learn
  - beautifulsoup4
  - requests
  - numpy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/shl-assessment-recommender.git
   cd shl-assessment-recommender
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get a Google Gemini API key from [Google AI Studio](https://ai.google.dev/)

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Enter your Google Gemini API key in the sidebar
3. If you don't have the SHL data file, click "Run Scraper" to gather the latest data
4. Enter your job requirements in the text area or select a sample query
5. Click "Get Recommendations" to see the results

## Data Collection

The system uses a web scraper to collect assessment information from the SHL product catalog. The scraper:

1. Visits the SHL product catalog page
2. Extracts information about both individual assessments and pre-packaged solutions
3. Organizes the data into a structured JSON format
4. Saves the data locally for efficient retrieval

To run the scraper manually:
```python
from scraper import SHLScraper
scraper = SHLScraper(max_workers=5, json_file="shl_solutions.json")
scraper.run()
```

## Recommendation Algorithm

The recommendation system uses a sophisticated scoring algorithm that combines several matching strategies:

### 1. Query Analysis

The system first analyzes the user's query using the Gemini AI model to extract:
- Technical skills required (e.g., Java, Python)
- Job levels (e.g., Entry-Level, Professional)
- Time constraints for assessments
- Required test types (e.g., Ability & Aptitude, Personality & Behavior)
- Remote testing requirements
- Adaptive/IRT support needs
- Language requirements

### 2. Dynamic Weight Assignment

The system determines the importance of each criterion based on the user's query:
- Explicitly mentioned criteria receive higher weights (4.0-5.0)
- Implied criteria receive medium weights (2.0-3.0)
- Unmentioned criteria receive lower weights (0.0-1.0)

For example, if the query emphasizes technical skills and time constraints, these factors will have higher weights in the scoring calculation.

### 3. Multi-criteria Scoring

Each assessment is scored across multiple dimensions:

- **Skills Matching**: Identifies technical skills in the assessment description
  - Exact matches score 1.0
  - Partial matches score 0.5

- **Job Level Matching**: Compares the assessment's job levels with query requirements
  - Score based on the number of matching job levels

- **Duration Matching**: Ensures the assessment fits within time constraints
  - Assessments under the time limit score higher (inverse proportion)
  - Assessments exceeding the limit are penalized but not excluded

- **Test Type Matching**: Matches the test types with requested categories
  - Score based on the proportion of matching test types

- **Remote Testing & Adaptive/IRT Support**: Boolean matching of these features

- **Language Support**: Ensures the assessment is available in required languages

- **Semantic Similarity**: Uses Gemini embeddings to calculate conceptual similarity
  - Captures nuanced matches beyond keyword matching

### 4. Weighted Scoring Formula

The final relevance score for each assessment is calculated as:

```
relevance_score = Σ (criterion_raw_score × criterion_weight)
```

Where:
- `criterion_raw_score` is the normalized score (0-1) for each matching criterion
- `criterion_weight` is the dynamically assigned importance weight (0-5)

### 5. Result Ranking

Assessments are ranked by their relevance scores, and the top N are returned to the user.

## API Integration

The system uses the Google Generative AI (Gemini) API for:

1. **Query understanding**: Extracting structured information from natural language
2. **Weight determination**: Analyzing query priorities to assign weights
3. **Embedding generation**: Creating vector representations for semantic search

To use the API:
1. Obtain a key from [Google AI Studio](https://ai.google.dev/)
2. Enter the key in the application sidebar
3. The application will cache embeddings to minimize API usage

## Evaluation Metrics

The system can be evaluated using these metrics:

### Mean Recall@K
This measures how many relevant assessments were retrieved in the top K recommendations, averaged across all test queries.

```
Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments for the query)
MeanRecall@K = (1/N) * Σ Recall@K_i
```

### Mean Average Precision@K (MAP@K)
This evaluates both relevance and ranking order by calculating Precision@k at each relevant result and averaging it.

```
AP@K = (1/min(K,R)) * Σ P(k) * rel(k)
MAP@K = (1/N) * Σ AP@K_i
```

Where:
- R = total relevant assessments for the query
- P(k) = precision at position k
- rel(k) = 1 if result at position k is relevant, otherwise 0
- N = total number of test queries

## Sample Queries

The system performs well on queries like:

1. "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes."

2. "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."

3. "I need to assess entry-level sales representatives with strong persuasive abilities. The assessment should take no more than 30 minutes."

## Acknowledgments

- SHL for providing the assessment product catalog
- Google for the Gemini AI API