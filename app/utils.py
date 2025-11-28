import re
import io
from typing import Optional, List, Tuple, Any
from urllib.parse import urljoin, urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup


def normalize_url(base_url: str, link: str) -> str:
    """Convert relative URL to absolute URL."""
    return urljoin(base_url, link)


def find_submit_url_from_text(text: str) -> Optional[str]:
    """
    Find submit URL in text (legacy function for compatibility).
    Looks for full URLs containing /submit or submit endpoints.
    """
    # Pattern for full URLs with submit
    pattern = r'https?://[^\s<>"\']+/submit[^\s<>"\']*'
    match = re.search(pattern, text)
    
    if match:
        return match.group(0).rstrip('.,;!?)')
    
    return None


def find_download_links_from_html(html: str) -> List[str]:
    """
    Extract downloadable file links from HTML.
    Looks for links to data files (CSV, JSON, PDF, Excel, etc.).
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    # File extensions that indicate data files
    data_extensions = [
        '.csv', '.json', '.xlsx', '.xls', '.pdf', 
        '.txt', '.xml', '.tsv', '.parquet', '.zip'
    ]
    
    # Find all <a> tags with href
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        
        # Check if link points to a data file
        if any(ext in href.lower() for ext in data_extensions):
            links.append(href)
    
    # Also check for links in onclick, data attributes, etc.
    for tag in soup.find_all(attrs={'data-url': True}):
        url = tag['data-url']
        if any(ext in url.lower() for ext in data_extensions):
            links.append(url)
    
    return links


def extract_column_sum_from_question(question_text: str) -> Optional[str]:
    """
    Heuristic to detect if question asks for sum of a specific column.
    Returns the column name if found.
    
    Examples:
        "What is the sum of the 'value' column?" -> "value"
        "Sum the price column" -> "price"
    """
    # Pattern 1: sum of the "column_name" column
    pattern1 = r'sum\s+(?:of\s+)?(?:the\s+)?["\']([^"\']+)["\'](?:\s+column)?'
    match = re.search(pattern1, question_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: sum the column_name column
    pattern2 = r'sum\s+(?:the\s+)?(\w+)\s+column'
    match = re.search(pattern2, question_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def download_and_load_data(url: str, timeout: int = 30) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Download a file from URL and attempt to load it as structured data.
    
    Args:
        url: URL to download from
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (metadata_string, dataframe_or_none)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content = response.content
        
        # Detect file type from URL or content-type
        content_type = response.headers.get('Content-Type', '').lower()
        url_lower = url.lower()
        
        # CSV
        if '.csv' in url_lower or 'text/csv' in content_type:
            df = pd.read_csv(io.BytesIO(content))
            meta = f"CSV file with {len(df)} rows and {len(df.columns)} columns\n"
            meta += f"Columns: {', '.join(df.columns)}\n"
            meta += f"First few rows:\n{df.head(10).to_string()}\n"
            meta += f"Summary statistics:\n{df.describe().to_string()}"
            return meta, df
        
        # JSON
        elif '.json' in url_lower or 'application/json' in content_type:
            try:
                df = pd.read_json(io.BytesIO(content))
                meta = f"JSON file loaded as dataframe with {len(df)} rows\n"
                meta += f"Columns: {', '.join(df.columns)}\n"
                meta += f"First few rows:\n{df.head(10).to_string()}"
                return meta, df
            except:
                # If not tabular JSON, just show the content
                import json
                data = json.loads(content)
                meta = f"JSON data:\n{json.dumps(data, indent=2)[:2000]}"
                return meta, None
        
        # Excel
        elif any(ext in url_lower for ext in ['.xlsx', '.xls']) or 'spreadsheet' in content_type:
            df = pd.read_excel(io.BytesIO(content))
            meta = f"Excel file with {len(df)} rows and {len(df.columns)} columns\n"
            meta += f"Columns: {', '.join(df.columns)}\n"
            meta += f"First few rows:\n{df.head(10).to_string()}\n"
            meta += f"Summary statistics:\n{df.describe().to_string()}"
            return meta, df
        
        # PDF
        elif '.pdf' in url_lower or 'application/pdf' in content_type:
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page_num, page in enumerate(pdf_reader.pages[:10], 1):  # First 10 pages
                    text += f"\n--- Page {page_num} ---\n"
                    text += page.extract_text()
                
                meta = f"PDF with {len(pdf_reader.pages)} pages\n"
                meta += f"Extracted text (first 10 pages):\n{text[:3000]}"
                return meta, None
            except Exception as e:
                return f"PDF file (could not extract text: {e})", None
        
        # Plain text
        elif '.txt' in url_lower or 'text/plain' in content_type:
            text = content.decode('utf-8', errors='ignore')
            meta = f"Text file content:\n{text[:3000]}"
            return meta, None
        
        # Unknown format
        else:
            meta = f"Downloaded {len(content)} bytes (unknown format)\n"
            meta += f"Content-Type: {content_type}\n"
            # Try to show as text if possible
            try:
                text = content.decode('utf-8', errors='ignore')[:1000]
                meta += f"Preview:\n{text}"
            except:
                meta += "(Binary content)"
            return meta, None
    
    except requests.exceptions.Timeout:
        return f"Error: Download timeout after {timeout}s", None
    
    except requests.exceptions.RequestException as e:
        return f"Error downloading file: {str(e)}", None
    
    except Exception as e:
        return f"Error processing file: {str(e)}", None


def extract_secret_code_from_text(text: str) -> Optional[str]:
    """
    Extract a secret code from text.
    Looks for patterns like "secret code: ABC123" or similar.
    """
    # Pattern 1: "secret code: VALUE" or "secret: VALUE"
    pattern1 = r'secret\s*(?:code)?[:\s]+([A-Za-z0-9_\-]+)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: "code: VALUE" or "code is VALUE"
    pattern2 = r'code[:\s]+(?:is\s+)?([A-Za-z0-9_\-]+)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 3: Look for alphanumeric strings that might be codes
    # (between 6-50 characters, mix of letters and numbers)
    pattern3 = r'\b([A-Za-z0-9]{6,50})\b'
    matches = re.findall(pattern3, text)
    
    # Filter to codes that look like actual codes (have both letters and numbers)
    for candidate in matches:
        has_letter = re.search(r'[A-Za-z]', candidate)
        has_number = re.search(r'[0-9]', candidate)
        if has_letter and has_number:
            return candidate
    
    return None


def clean_html(html: str) -> str:
    """Remove script and style tags from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style']):
        script.decompose()
    
    return str(soup)




