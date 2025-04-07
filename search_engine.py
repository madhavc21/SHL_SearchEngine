import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity

def calculate_criterion_score(test: Dict, extracted_info: Dict, criterion: str) -> Tuple[float, Optional[Any]]:
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
    individual_embeddings: Optional[List[List[float]]] = None,
    pre_packaged_embeddings: Optional[List[List[float]]] = None,
    query_embedding: Optional[List[float]] = None,
    max_results: int = 10,
    use_semantic: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Search for relevant assessments using both keyword matching and semantic similarity.
    Returns separate results for individual tests and pre-packaged solutions.
    """
    individual_tests = data.get("individual_test_solutions", {})
    pre_packaged_solutions = data.get("pre_packaged_job_solutions", {})
    individual_results = []
    pre_packaged_results = []
    
    individual_test_ids = list(individual_tests.keys())
    pre_packaged_ids = list(pre_packaged_solutions.keys())
    individual_semantic_scores = {}
    pre_packaged_semantic_scores = {}
    
    # Calculate semantic scores if semantic search is enabled and embeddings are provided
    if use_semantic and query_embedding:
        try:
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            # Calculate similarities for individual tests
            if individual_embeddings:
                individual_embeddings_np = np.array(individual_embeddings)
                if query_embedding_np.size > 0 and individual_embeddings_np.size > 0:
                    similarities = cosine_similarity(query_embedding_np, individual_embeddings_np)[0]
                    for i, test_id in enumerate(individual_test_ids):
                        individual_semantic_scores[test_id] = similarities[i]
            
            # Calculate similarities for pre-packaged solutions
            if pre_packaged_embeddings:
                pre_packaged_embeddings_np = np.array(pre_packaged_embeddings)
                if query_embedding_np.size > 0 and pre_packaged_embeddings_np.size > 0:
                    similarities = cosine_similarity(query_embedding_np, pre_packaged_embeddings_np)[0]
                    for i, solution_id in enumerate(pre_packaged_ids):
                        pre_packaged_semantic_scores[solution_id] = similarities[i]
                
        except Exception:
            # If semantic search fails, we'll fall back to keyword search only
            pass
    
    # Process individual tests
    for test_id in individual_test_ids:
        test = individual_tests[test_id]
        
        # Start with semantic score if available
        semantic_score = individual_semantic_scores.get(test_id, 0)
        score = semantic_score * dynamic_weights.get("semantic", 1.0)
        
        score_details = {
            "semantic": {
                "raw_score": semantic_score,
                "weighted_score": semantic_score * dynamic_weights.get("semantic", 1.0)
            }
        }
        
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
    for solution_id in pre_packaged_ids:
        solution = pre_packaged_solutions[solution_id]
        
        # Start with semantic score if available
        semantic_score = pre_packaged_semantic_scores.get(solution_id, 0)
        score = semantic_score * dynamic_weights.get("semantic", 1.0)
        
        score_details = {
            "semantic": {
                "raw_score": semantic_score,
                "weighted_score": semantic_score * dynamic_weights.get("semantic", 1.0)
            }
        }
        
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
    
    return individual_results[:max_results], pre_packaged_results[:max_results]