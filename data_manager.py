import streamlit as st
import os
import json
import pickle
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from gemini_client import initialize_genai, create_embeddings

# Load JSON data
@st.cache_data
def load_data(file_path: str = "shl_solutions.json") -> Dict:
    """Load and cache the SHL solutions data."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"individual_test_solutions": {}, "pre_packaged_job_solutions": {}}

def get_cached_embeddings(individual_tests, pre_packaged_solutions, api_key):
    """Get embeddings from disk cache or create new ones."""
    cache_dir = "embedding_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache keys based on data content hash
    ind_test_ids = list(individual_tests.keys())
    pre_pkg_ids = list(pre_packaged_solutions.keys())
    ind_cache_key = f"ind_embeddings_{len(ind_test_ids)}"
    pre_pkg_cache_key = f"pre_pkg_embeddings_{len(pre_pkg_ids)}"
    
    ind_cache_file = os.path.join(cache_dir, f"{ind_cache_key}.pkl")
    pre_pkg_cache_file = os.path.join(cache_dir, f"{pre_pkg_cache_key}.pkl")
    
    individual_embeddings = None
    pre_packaged_embeddings = None
    
    # Check if individual test cache exists
    if os.path.exists(ind_cache_file):
        try:
            with open(ind_cache_file, 'rb') as f:
                individual_embeddings = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load cached individual embeddings: {e}")
    
    # Check if pre-packaged cache exists
    if os.path.exists(pre_pkg_cache_file):
        try:
            with open(pre_pkg_cache_file, 'rb') as f:
                pre_packaged_embeddings = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load cached pre-packaged embeddings: {e}")
    
    # Create new embeddings if needed
    if individual_embeddings is None:
        st.write("Creating embeddings for individual assessments...")
        individual_texts = [f"{test.get('name', '')} {test.get('description', '')}" 
                           for test in individual_tests.values()]
        individual_embeddings = create_embeddings(individual_texts, api_key, task_type="retrieval_document")
        
        # Save to cache
        try:
            with open(ind_cache_file, 'wb') as f:
                pickle.dump(individual_embeddings, f)
        except Exception as e:
            st.warning(f"Could not save individual embeddings to cache: {e}")
    
    if pre_packaged_embeddings is None:
        st.write("Creating embeddings for pre-packaged solutions...")
        pre_packaged_texts = [f"{solution.get('name', '')} {solution.get('description', '')}" 
                             for solution in pre_packaged_solutions.values()]
        pre_packaged_embeddings = create_embeddings(pre_packaged_texts, api_key, task_type="retrieval_document")
        
        # Save to cache
        try:
            with open(pre_pkg_cache_file, 'wb') as f:
                pickle.dump(pre_packaged_embeddings, f)
        except Exception as e:
            st.warning(f"Could not save pre-packaged embeddings to cache: {e}")
    
    return individual_embeddings, pre_packaged_embeddings