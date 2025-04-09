import streamlit as st
import os
import json
from data_manager import load_data
from gemini_client import initialize_genai, process_query_with_gemini, determine_dynamic_weights
from search_engine import search_assessments
from utils import detect_url, scrape_job_description, SCHEMA_INFO

#Set API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure page
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="wide"
)

def process_complete_query(query: str, api_key: str):
    """
    Process the complete query including URL extraction and dynamic weight determination.
    """
    # Extract URL if present
    url = detect_url(query)
    enhanced_query = query
    
    if url:
        try:
            st.info(f"Extracting content from URL: {url}")
            job_description = scrape_job_description(url)
            # Combine original query with job description
            enhanced_query = f"{query}\n\nJob Description from URL: {job_description}"
        except Exception as e:
            st.warning(f"Could not process URL: {e}")
    
    # Extract information from query
    extracted_info = process_query_with_gemini(enhanced_query, api_key)
    
    # Determine dynamic weights
    dynamic_weights = determine_dynamic_weights(enhanced_query, extracted_info, api_key)
    
    return extracted_info, dynamic_weights

# Display assessment recommendations
def display_recommendations(recommendations, show_scoring_details=False):
    """Display the recommended assessments in a table format."""
    if not recommendations:
        st.info("No matching assessments found. Try broadening your search criteria.")
        return
    
    st.write(f"Found {len(recommendations)} matching assessments:")
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec['name']} (Score: {rec['relevance_score']:.2f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### [{rec['name']}]({rec['url']})")
                st.write(rec['description'])
                
                if 'job_levels' in rec and any(rec.get('job_levels', [])):
                    st.markdown("**Job Levels:** " + ", ".join(filter(None, rec.get('job_levels', []))))
                
                if 'languages' in rec and any(rec.get('languages', [])):
                    st.markdown("**Languages:** " + ", ".join(filter(None, rec.get('languages', []))))
            
            with col2:
                st.markdown(f"**Duration:** {rec.get('duration', 'N/A')} minutes")
                st.markdown(f"**Remote Testing:** {'Yes' if rec.get('remote_testing') else 'No'}")
                st.markdown(f"**Adaptive/IRT:** {'Yes' if rec.get('adaptive_irt') else 'No'}")
                st.markdown(f"**Test Type:** {rec.get('test_type', 'N/A')}")
                
                if 'test_type_descriptions' in rec and any(rec.get('test_type_descriptions', [])):
                    st.markdown("**Test Categories:**")
                    for desc in filter(None, rec.get('test_type_descriptions', [])):
                        st.markdown(f"- {desc}")
            
            if show_scoring_details and 'score_details' in rec:
                st.divider()
                st.markdown("##### Relevance Score Breakdown")
                
                score_cols = st.columns(4)
                
                criteria_order = [
                    ("semantic", "Semantic Match"),
                    ("skills", "Skills Match"),
                    ("job_levels", "Job Level Match"),
                    ("duration", "Duration Match"),
                    ("test_types", "Test Type Match"),
                    ("remote_testing", "Remote Testing"),
                    ("adaptive_irt", "Adaptive Testing"),
                    ("languages", "Language Match")
                ]
                
                col_idx = 0
                for criterion_key, criterion_label in criteria_order:
                    if criterion_key in rec['score_details']:
                        detail = rec['score_details'][criterion_key]
                        with score_cols[col_idx % 4]:
                            st.metric(
                                criterion_label, 
                                f"{detail['weighted_score']:.2f}",
                                f"Weight: {detail.get('weight', 0):.1f}"
                            )
                            
                            if detail.get('matches'):
                                if isinstance(detail['matches'], list):
                                    if detail['matches']:
                                        st.caption("Matched: " + ", ".join(str(m) for m in detail['matches']))
                                else:
                                    st.caption(str(detail['matches']))
                            
                            col_idx += 1

# Main application
def main():
    # Initialize session state for weights if not exists
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "skills": 3.0,
            "job_levels": 2.0,
            "duration": 2.0,
            "remote_testing": 1.0,
            "adaptive_irt": 1.0,
            "test_types": 3.0,
            "languages": 1.0,
            "semantic": 2.0
        }
    
    st.title("🧠 SHL Assessment Recommendation System")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("Configuration")
        data_file = st.text_input(
            "Path to SHL solutions data:", 
            value="shl_solutions.json",
            help="Path to the JSON file containing SHL assessment data"
        )
        
        # Check if the data file exists
        if not os.path.exists(data_file):
            st.warning(f"Data file '{data_file}' not found. You need to scrape it first.")
            if st.button("Run Scraper", type="primary"):
                with st.spinner("Scraping SHL website. This may take a while..."):
                    try:
                        # Import scraper module dynamically to avoid dependency issues
                        from scraper import SHLScraper
                        
                        # Run the scraper
                        scraper = SHLScraper(max_workers=5, json_file=data_file)
                        scraper.run()
                        st.success(f"Successfully scraped data and saved to {data_file}")
                    except Exception as e:
                        st.error(f"Error running scraper: {e}")
        
        max_results = st.slider(
            "Maximum recommendations:", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Maximum number of assessment recommendations to display"
        )
                
        # Advanced settings in an expander
        with st.expander("Advanced Settings"):
            use_semantic_search = st.checkbox(
                "Use semantic search", 
                value=True,
                help="Use embeddings for more conceptual matching beyond keywords"
            )
            
            show_scoring_details = st.checkbox(
                "Show scoring details", 
                value=True,
                help="Display detailed scoring breakdown for each recommendation"
            )
        
        st.divider()
        
        st.markdown("""
        ### About
        This application recommends SHL assessments based on your job requirements.
        
        Simply describe the role you're hiring for, and we'll suggest the most relevant assessments.
        The system analyzes your query to understand what factors are most important to you and ranks
        assessments accordingly.
        """)
        
        with st.expander("How scoring works"):
            st.markdown("""
            ### Dynamic Scoring System
            
            This recommendation system analyzes your query to determine what factors are most important to you:
            
            1. **Query Understanding**: We use AI to identify your priorities (skills, duration, test types, etc.)
            2. **Dynamic Weights**: The system automatically assigns importance to different criteria based on your query
            3. **Semantic Matching**: We find assessments that conceptually match your needs, not just keyword matches
            4. **Relevance Calculation**: Each assessment is scored based on how well it meets your specific requirements
            
            This approach ensures you get recommendations tailored to what matters most in your specific hiring scenario.
            """)
    
    # Load data
    data = load_data(data_file)
    if not data.get("individual_test_solutions") or not data.get("pre_packaged_job_solutions"):
        st.error("No assessment data loaded. Please check the data file path.")
        return
    
    # Main content
    query_container = st.container()
    
    with query_container:
        st.write("Enter your hiring requirements below. You can include a job description URL if you have one.")
        
        sample_queries = [
            "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes.",
            "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "I need to assess entry-level sales representatives with strong persuasive abilities. The assessment should take no more than 30 minutes.",
            "Looking for cognitive and personality tests for analyst positions, must be under 45 minutes total.",
            "Need remote testing options for assessing leadership skills in senior management candidates."
        ]
        
        selected_sample = st.selectbox(
            "Try a sample query:",
            [""] + sample_queries,
            index=0,
            help="Select a sample query or enter your own below"
        )
        
        # Set the selected sample as the query if one was chosen
        query = st.text_area(
            "Describe the role and assessment requirements:",
            value=selected_sample,
            placeholder="E.g., 'I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes.'",
            height=150
        )
        
        if st.button("Get Recommendations", type="primary", disabled=not api_key):
            if not query:
                st.warning("Please enter a query or job description URL.")
            else:
                with st.spinner("Processing your query..."):
                    # Process complete query with URL handling
                    extracted_info, dynamic_weights = process_complete_query(query, api_key)
                    
                    # Find and display recommendations using custom weights
                    individual_recommendations, pre_packaged_recommendations = search_assessments(
                        data, 
                        extracted_info, 
                        dynamic_weights,
                        api_key,
                        max_results=max_results,
                        use_semantic=use_semantic_search
                    )
                    
                    # Display recommendations for both types
                    st.divider()
                    
                    # Display Individual Test Results
                    st.subheader("Recommended Individual Assessments")
                    display_recommendations(individual_recommendations, show_scoring_details)
                    
                    # Display Pre-packaged Solutions Results
                    st.divider()
                    st.subheader("Recommended Pre-packaged Solutions")
                    if pre_packaged_recommendations:
                        display_recommendations(pre_packaged_recommendations, show_scoring_details)
                    else:
                        st.info("No matching pre-packaged solutions found.")

                    with st.expander("Search details"):
                        # Display the extracted information
                        st.subheader("Query Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Search Parameters**")
                            
                            # Display priority weights with visualization
                            st.write("**Priority Weights** (dynamically determined from your query)")
                            
                            # Create tuples of (criterion, weight) and sort by weight
                            weight_items = [(k, v) for k, v in dynamic_weights.items()]
                            weight_items.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display as a progress bar for each weight
                            for criterion, weight in weight_items:
                                # Format criterion name for display
                                criterion_display = criterion.replace("_", " ").title()
                                # Calculate percentage for progress bar (0-100%)
                                percentage = min(100, round(weight * 20))  # Scale 0-5 to 0-100%
                                
                                # Display as metric with progress bar
                                cols = st.columns([3, 7])
                                with cols[0]:
                                    st.write(f"**{criterion_display}**")
                                with cols[1]:
                                    st.progress(percentage / 100)
                                    st.write(f"{weight:.1f}/5.0")
                            
                            # Show extracted parameters
                            st.divider()
                            st.write("**Extracted Search Parameters**")
                            
                            if extracted_info.get("job_levels"):
                                st.write(f"🔹 **Job Levels**: {', '.join(extracted_info['job_levels'])}")
                            
                            if extracted_info.get("skills"):
                                st.write(f"🔹 **Technical Skills**: {', '.join(extracted_info['skills'])}")
                            
                            if extracted_info.get("duration"):
                                st.write(f"🔹 **Maximum Duration**: {extracted_info['duration']} minutes")
                            
                            if extracted_info.get("remote_testing") is not None:
                                st.write(f"🔹 **Remote Testing**: {'Yes' if extracted_info['remote_testing'] else 'Not specified'}")
                            
                            if extracted_info.get("adaptive_irt") is not None:
                                st.write(f"🔹 **Adaptive/IRT**: {'Yes' if extracted_info['adaptive_irt'] else 'Not specified'}")
                            
                            if extracted_info.get("test_types"):
                                st.write(f"🔹 **Test Types**: {', '.join(extracted_info['test_types'])}")
                        
                        # Current weights
                        st.write("**Current Importance Weights**")
                        num_cols = 5 
                        weights_items = list(st.session_state.weights.items())
                        weights_rows = [weights_items[i:i+num_cols] for i in range(0, len(weights_items), num_cols)]

                        for row in weights_rows:
                            cols = st.columns(len(row))
                            for i, (criterion, weight) in enumerate(row):
                                with cols[i]:
                                        st.metric(criterion.replace("_", " ").title(), f"{weight:.1f}")
                        

if __name__ == "__main__":
    main()
