import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
import logging
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("scraper_log.txt"), 
                              logging.StreamHandler()])

class SHLScraper:
    def __init__(self, max_workers=10, json_file="shl_solutions.json"):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.json_file = json_file
        self.solutions = {
            "pre_packaged_job_solutions": {},
            "individual_test_solutions": {}
        }
        
        # Test type mappings
        self.test_type_mappings = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgement',
            'C': 'Competencies',
            'D': 'Development & 360',
            'E': 'Assessment Exercises',
            'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior',
            'S': 'Simulations'
        }
        
        # Set up a session with connection pooling and retry strategy
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=retries))
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Separate tracking sets for different purposes
        self.visited_pages = set()  # Pages we've visited (not solutions)
        self.visited_solution_urls = set()  # Solution detail pages we've visited
        self.processed_solution_names = set()  # Solutions we've fully processed
        self.queued_urls = set()  # URLs already in queue to avoid duplicates
        
        self.max_workers = max_workers
        
        # Load existing data if available
        self.load_existing_data()
        
    def generate_hash(self, solution_name, url):
        """Generate a permanent hash for a solution based on name and URL"""
        hash_str = f"{solution_name}::{url}"
        return hashlib.md5(hash_str.encode()).hexdigest()
        
    def load_existing_data(self):
        """Load existing data from JSON file if it exists"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                if not isinstance(existing_data, dict):
                    logging.warning(f"Existing data in {self.json_file} has incorrect format. Starting fresh.")
                    return
                
                # Initialize solutions from existing data
                for category in self.solutions:
                    if category in existing_data:
                        self.solutions[category] = existing_data[category]
                
                # Build tracking sets from existing data
                for category, solutions in self.solutions.items():
                    for name, data in solutions.items():
                        if 'url' in data:
                            self.visited_solution_urls.add(data['url'])
                            self.processed_solution_names.add(name)
                
                logging.info(f"Loaded {len(self.visited_solution_urls)} existing solutions from {self.json_file}")
            except Exception as e:
                logging.error(f"Error loading existing data: {str(e)}")
                logging.info("Starting with fresh data")
        else:
            logging.info(f"No existing data file found at {self.json_file}. Will create new file.")
        
    def get_soup(self, url):
        """Get BeautifulSoup object from URL with error handling"""
        try:
            # Check if URL is properly formed and contains the base URL domain
            parsed_url = urlparse(url)
            if not parsed_url.netloc:
                url = urljoin(self.base_url, url)
                
            # Ensure we're staying within the product catalog
            if not url.startswith(self.base_url):
                logging.warning(f"URL {url} is outside the product catalog scope. Skipping.")
                return None
                
            # Return None for already visited pages (but not solution pages)
            if "product-catalog/" in url and "?" in url and url in self.visited_pages:
                return None
                
            logging.info(f"Fetching {url}")
            
            # Mark as visited before making the request to prevent concurrent duplicates
            self.visited_pages.add(url)
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Use full HTML parser for better accuracy
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None
    
    def parse_test_type_letters(self, element):
        """
        Parse test type letters from an element and map them to their full names
        Returns both the letter codes and their full descriptions
        """
        if not element:
            return "", []
             
        text = element.get_text(strip=True)
        # Extract letters that match our known test type keys
        letters = ''.join([c for c in text if c in self.test_type_mappings])
        
        # Map letters to descriptions
        descriptions = [self.test_type_mappings.get(letter, '') for letter in letters if letter in self.test_type_mappings]
        return letters, descriptions
    
    def parse_solution_page(self, url, solution_name):
        """Extract detailed information from a solution page"""
        soup = self.get_soup(url)
        if not soup:
            return {}
        
        # Initialize solution data
        solution_data = {
            "name": solution_name,
            "url": url,
            "description": "",
            "job_levels": [],
            "languages": [],
            "duration": "",
            "remote_testing": False,
            "adaptive_irt": False,
            "test_type": "",
            "test_type_descriptions": [],
            "hash": self.generate_hash(solution_name, url)
        }
        
        # Extract description - prioritize exact match first
        description_elem = soup.find(string=re.compile("^Description$", re.IGNORECASE))
        if not description_elem:
            description_elem = soup.find(string=re.compile("Description", re.IGNORECASE))
            
        if description_elem and description_elem.find_next():
            solution_data["description"] = description_elem.find_next().get_text(strip=True)
        else:
            # Alternative approach - look for specific divs or sections
            description_div = soup.find('div', class_=re.compile('description|content|details', re.IGNORECASE)) 
            if description_div:
                solution_data["description"] = description_div.get_text(strip=True)
        
        # Extract job levels
        job_levels_elem = soup.find(string=re.compile("Job levels", re.IGNORECASE))
        if job_levels_elem and job_levels_elem.find_next():
            solution_data["job_levels"] = [level.strip() for level in job_levels_elem.find_next().get_text(strip=True).split(',')]
        
        # Extract languages
        languages_elem = soup.find(string=re.compile("Languages", re.IGNORECASE))
        if languages_elem and languages_elem.find_next():
            solution_data["languages"] = [lang.strip() for lang in languages_elem.find_next().get_text(strip=True).split(',')]
        
        # Extract duration
        duration_pattern = re.compile(r'(\d+)\s*minutes|minutes\s*=\s*(\d+)|(\d+)\s*min|time:\s*(\d+)', re.IGNORECASE)
        duration_matches = soup.find_all(string=duration_pattern)
        
        if duration_matches:
            for text in duration_matches:
                match = duration_pattern.search(text)
                if match:
                    # Find the first non-None group
                    for group in match.groups():
                        if group:
                            solution_data["duration"] = group
                            break
                    break
        
        # Extract booleans
        for field, pattern in [
            ("remote_testing", re.compile("Remote Testing", re.IGNORECASE)),
            ("adaptive_irt", re.compile("Adaptive/IRT", re.IGNORECASE))
        ]:
            field_elem = soup.find(string=pattern)
            if field_elem:
                parent_elem = None
                for parent in field_elem.parents:
                    if parent.name in ['tr', 'div', 'section']:
                        parent_elem = parent
                        break
                
                if parent_elem:
                    # Look for yes indicators
                    if parent_elem.find_all(['span', 'i', 'div'], class_=re.compile('green|check|yes|tick|circle', re.IGNORECASE)) or \
                       parent_elem.find(string=re.compile(r'\byes\b', re.IGNORECASE)):
                        solution_data[field] = True
        
        # Extract test type
        test_type_elem = soup.find(string=re.compile("Test Type", re.IGNORECASE))
        if test_type_elem:
            parent_elem = None
            for parent in test_type_elem.parents:
                if parent.name in ['tr', 'div', 'section', 'td']:
                    parent_elem = parent
                    break
            
            if parent_elem:
                # Try to find siblings or following elements that might contain the test type
                next_siblings = list(parent_elem.next_siblings)
                if next_siblings:
                    for sibling in next_siblings:
                        if sibling.name in ['td', 'div', 'span']:
                            test_type_letters, descriptions = self.parse_test_type_letters(sibling)
                            if test_type_letters:
                                solution_data["test_type"] = test_type_letters
                                solution_data["test_type_descriptions"] = descriptions
                                break
                
                # If still no test type found, try within the parent element
                if not solution_data["test_type"]:
                    test_type_letters, descriptions = self.parse_test_type_letters(parent_elem)
                    if test_type_letters:
                        solution_data["test_type"] = test_type_letters
                        solution_data["test_type_descriptions"] = descriptions
        
        logging.info(f"Parsed solution: {solution_name}")
        return solution_data
    
    def extract_test_type_from_row(self, row):
        """Extract test type letters from a table row"""
        try:
            # Find the test type cell specifically
            test_type_cell = row.find('td', class_=re.compile('product-catalogue__keys', re.IGNORECASE))
            
            # If not found by class, try finding by position or column header
            if not test_type_cell:
                # Find cells in the row
                cells = row.find_all(['td', 'th'])
                
                # Try to find by column header "Test Type"
                headers = row.parent.find('tr').find_all('th') if row.parent else []
                for idx, header in enumerate(headers):
                    if header.get_text(strip=True).lower() == "test type" and idx < len(cells):
                        test_type_cell = cells[idx]
                        break
                
                # If still not found and we have enough cells, use the last cell (common pattern)
                if not test_type_cell and len(cells) >= 3:
                    test_type_cell = cells[-1]
                    
            if test_type_cell:
                return self.parse_test_type_letters(test_type_cell)
            
            return "", []
        except Exception as e:
            logging.error(f"Error extracting test type from row: {str(e)}")
            return "", []
        
    def extract_pagination_links(self, soup, base_url):
        """Extract pagination links from the page"""
        pagination_links = set()
        
        # Find pagination elements
        pagination_containers = soup.select('ul.pagination, div.pagination, nav.pagination')
        
        if pagination_containers:
            for container in pagination_containers:
                links = container.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        # Only add if it's a catalog URL and not already queued
                        if "product-catalog" in full_url and full_url not in self.queued_urls:
                            pagination_links.add(full_url)
                            self.queued_urls.add(full_url)
        else:
            # Fallback to scanning all links
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue
                    
                # Check if the link text suggests pagination
                link_text = link.get_text(strip=True)
                if re.match(r'^\d+$|^Next$|^Prev$|^»$|^›$|^<$|^>$', link_text, re.IGNORECASE):
                    full_url = urljoin(base_url, href)
                    if "product-catalog" in full_url and full_url not in self.queued_urls:
                        pagination_links.add(full_url)
                        self.queued_urls.add(full_url)
        
        return pagination_links
    
    def extract_solution_links_from_table(self, table, table_type):
        """Extract solution links from a table"""
        solution_links = []
        
        # Find all rows
        rows = table.find_all('tr')
        
        for row in rows:
            # Skip header row
            if row.find('th'):
                continue
                
            # Get all links in this row
            links = row.find_all('a', href=True)
            if not links:
                continue
                
            for link in links:
                solution_name = link.get_text(strip=True)
                
                # Skip empty links or pagination
                if not solution_name or re.match(r'^\d+$|^Next$|^Prev$|^»$|^›$|^<$|^>$', solution_name, re.IGNORECASE):
                    continue
                    
                solution_url = urljoin(self.base_url, link['href'])
                
                # Get test type from the row
                test_type, test_type_descriptions = self.extract_test_type_from_row(row)
                
                # Add to list
                solution_links.append({
                    'name': solution_name,
                    'url': solution_url,
                    'test_type': test_type,
                    'test_type_descriptions': test_type_descriptions,
                    'table_type': table_type
                })
        
        return solution_links
    
    def process_solution_links(self, solution_links):
        """Process a batch of solution links concurrently"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_link = {}
            
            for link_data in solution_links:
                # Skip if we've already processed this solution
                if link_data['name'] in self.processed_solution_names:
                    continue
                    
                # Skip if we've already visited this URL
                if link_data['url'] in self.visited_solution_urls:
                    continue
                    
                future = executor.submit(
                    self.parse_solution_page, 
                    link_data['url'], 
                    link_data['name']
                )
                future_to_link[future] = link_data
            
            for future in concurrent.futures.as_completed(future_to_link):
                link_data = future_to_link[future]
                try:
                    solution_data = future.result()
                    
                    if solution_data:
                        # Use test type from table if not found in detail page
                        if not solution_data["test_type"] and link_data['test_type']:
                            solution_data["test_type"] = link_data['test_type']
                            solution_data["test_type_descriptions"] = link_data['test_type_descriptions']
                        
                        # Ensure no duplicates in test type as final safety check
                        if solution_data["test_type"]:
                            seen = set()
                            solution_data["test_type"] = ''.join([c for c in solution_data["test_type"] 
                                                            if not (c in seen or seen.add(c))])
                            
                            # Rebuild descriptions to match the de-duplicated test types
                            solution_data["test_type_descriptions"] = [
                                self.test_type_mappings.get(letter, '') 
                                for letter in solution_data["test_type"] 
                                if letter in self.test_type_mappings
                            ]
                        
                        results.append((link_data['name'], solution_data, link_data['table_type']))
                except Exception as e:
                    logging.error(f"Error processing solution {link_data['name']}: {str(e)}")
        
        return results
    
    def scrape_page(self, url):
        """Scrape a single page of solutions"""
        soup = self.get_soup(url)
        if not soup:
            return [], []
        
        # Determine the solution category for this page
        if "?view=individual-test-solutions" in url or "individual" in url.lower():
            table_type = "individual_test_solutions"
            table_heading = "Individual Test Solutions"
        else:
            table_type = "pre_packaged_job_solutions"
            table_heading = "Pre-packaged Job Solutions"
        
        # Find tables - look for multiple approaches
        tables = []
        
        # 1. Find by header text
        header = soup.find(lambda tag: tag.name and tag.get_text(strip=True) == table_heading)
        if header:
            # Find the containing table
            parent = header
            while parent and parent.name != 'table':
                parent = parent.parent
                if parent is None:
                    break
            if parent and parent.name == 'table':
                tables.append(parent)
        
        # 2. If no table found by header, find all tables
        if not tables:
            tables = soup.find_all('table')
        
        solution_links = []
        for table in tables:
            links = self.extract_solution_links_from_table(table, table_type)
            solution_links.extend(links)
        
        # Get pagination links
        next_pages = self.extract_pagination_links(soup, url)
        
        return solution_links, next_pages
    
    def scrape_catalog(self):
        """Scrape the entire SHL catalog using breadth-first search"""
        logging.info("Starting to scrape SHL catalog...")
        
        # Queue of pages to process
        pages_queue = [
            self.base_url,  # For pre-packaged solutions
            self.base_url + "?view=individual-test-solutions"  # For individual solutions
        ]
        
        # Add both URLs to queued set
        for url in pages_queue:
            self.queued_urls.add(url)
        
        # Track statistics
        total_solution_links = 0
        total_pages_processed = 0
        
        while pages_queue:
            current_url = pages_queue.pop(0)
            total_pages_processed += 1
            
            logging.info(f"Processing page {total_pages_processed}: {current_url}")
            solution_links, next_pages = self.scrape_page(current_url)
            
            # Process this batch of solution links
            if solution_links:
                total_solution_links += len(solution_links)
                logging.info(f"Found {len(solution_links)} solution links on page (total so far: {total_solution_links})")
                
                results = self.process_solution_links(solution_links)
                
                # Add solutions to the appropriate category
                for solution_name, solution_data, category in results:
                    if solution_name not in self.solutions[category]:
                        self.solutions[category][solution_name] = solution_data
                        
            # Add new pages to the queue
            for next_url in next_pages:
                if next_url not in pages_queue:
                    pages_queue.append(next_url)
            
            # Add a small delay to prevent overloading the server
            time.sleep(0.2)
        
        # Print summary
        pre_packaged_count = len(self.solutions["pre_packaged_job_solutions"])
        individual_count = len(self.solutions["individual_test_solutions"])
        total_count = pre_packaged_count + individual_count
        
        logging.info("\nScraping complete!")
        logging.info(f"Pre-packaged Job Solutions: {pre_packaged_count}")
        logging.info(f"Individual Test Solutions: {individual_count}")
        logging.info(f"Total Solutions: {total_count}")
        logging.info(f"Total pages processed: {total_pages_processed}")
        
    def save_to_json(self, filename=None):
        """Save the scraped data to a JSON file"""
        if filename is None:
            filename = self.json_file
            
        try:
            # Create backup of existing file first
            if os.path.exists(filename) and filename == self.json_file:
                backup_name = f"{filename}.bak.{int(time.time())}"
                try:
                    with open(filename, 'r', encoding='utf-8') as src, open(backup_name, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                    logging.info(f"Created backup of existing file: {backup_name}")
                except Exception as e:
                    logging.error(f"Error creating backup: {str(e)}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.solutions, f, indent=4)
            logging.info(f"Data saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving JSON file: {str(e)}")
            # Try to save with a backup name
            try:
                backup_filename = f"shl_solutions_backup_{int(time.time())}.json"
                with open(backup_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.solutions, f, indent=4)
                logging.info(f"Data saved to backup file {backup_filename}")
            except:
                logging.error("Failed to save even to backup file")
        
    def run(self):
        """Run the scraper and save results"""
        try:
            start_time = time.time()
            self.scrape_catalog()
            self.save_to_json()
            elapsed_time = time.time() - start_time
            logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
            logging.info(f"Processed {len(self.visited_solution_urls)} unique solution links")
            logging.info(f"Total solutions in database: {len(self.solutions['pre_packaged_job_solutions']) + len(self.solutions['individual_test_solutions'])}")
        except Exception as e:
            logging.critical(f"Critical error in main execution: {str(e)}")

if __name__ == "__main__":
    scraper = SHLScraper(max_workers=5, json_file="shl_solutions.json")
    scraper.run()