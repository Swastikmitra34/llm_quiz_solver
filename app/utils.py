"""
utils.py
Enhanced utilities for handling various question types:
- Web scraping (with JS support)
- API calls with headers
- PDF extraction
- Image processing
- Data transformation
- Visualization generation
"""

import re
import base64
import io
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup

# For PDF handling
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# For image handling
try:
    from PIL import Image
    import pytesseract
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False

# For visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def normalize_url(base_url: str, link: str) -> str:
    """Convert relative URLs to absolute URLs"""
    if link.startswith("http://") or link.startswith("https://"):
        return link
    return urljoin(base_url, link)


def find_submit_url_from_text(text: str) -> Optional[str]:
    """Extract submit URL from page text"""
    patterns = [
        r"post.*?to\s+(https?://[^\s\"'<>]+)",
        r"submit.*?to\s+(https?://[^\s\"'<>]+)",
        r"POST\s+(https?://[^\s\"'<>]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".,;:")
    
    return None


def find_download_links_from_html(html: str) -> List[str]:
    """Find downloadable file links (CSV, Excel, PDF, JSON, etc.)"""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    
    file_extensions = ['.csv', '.xlsx', '.xls', '.json', '.pdf', '.txt', '.parquet']
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(href.lower().endswith(ext) for ext in file_extensions):
            links.append(href)
    
    return links


def extract_api_headers_from_text(text: str) -> Dict[str, str]:
    """Extract API headers mentioned in the page text"""
    headers = {}
    
    # Look for header patterns like: "X-API-Key: abc123" or "Authorization: Bearer token"
    header_pattern = r"([A-Z][a-zA-Z-]+):\s*([^\n\r]+)"
    matches = re.findall(header_pattern, text)
    
    for key, value in matches:
        if key.lower() in ['authorization', 'x-api-key', 'api-key', 'token']:
            headers[key] = value.strip()
    
    return headers


def download_and_load_data(url: str, headers: Optional[Dict] = None) -> Tuple[str, pd.DataFrame]:
    """
    Download and load data file (CSV, Excel, JSON, etc.)
    Returns (metadata_string, dataframe)
    """
    try:
        response = requests.get(url, headers=headers or {}, timeout=30)
        response.raise_for_status()
        
        # Determine file type
        content_type = response.headers.get('content-type', '').lower()
        url_lower = url.lower()
        
        # CSV
        if 'csv' in content_type or url_lower.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(response.content))
            meta = f"CSV file: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            return meta, df
        
        # Excel
        elif 'excel' in content_type or url_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(response.content))
            meta = f"Excel file: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            return meta, df
        
        # JSON
        elif 'json' in content_type or url_lower.endswith('.json'):
            df = pd.read_json(io.BytesIO(response.content))
            meta = f"JSON file: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            return meta, df
        
        # Parquet
        elif url_lower.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(response.content))
            meta = f"Parquet file: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            return meta, df
        
        # Try CSV as default
        else:
            df = pd.read_csv(io.BytesIO(response.content))
            meta = f"Data file: {url}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            return meta, df
            
    except Exception as e:
        raise Exception(f"Failed to load data from {url}: {str(e)}")


def extract_text_from_pdf(url: str, headers: Optional[Dict] = None) -> str:
    """Extract text from PDF file"""
    if not HAS_PDF:
        return "PDF libraries not available"
    
    try:
        response = requests.get(url, headers=headers or {}, timeout=30)
        response.raise_for_status()
        
        # Try pdfplumber first (better for tables)
        try:
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text_parts.append(f"--- Page {page.page_number} ---")
                    text_parts.append(page.extract_text() or "")
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            text_parts.append("\nTable:")
                            text_parts.append(df.to_string())
                
                return "\n".join(text_parts)
        except:
            pass
        
        # Fallback to PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            text_parts.append(f"--- Page {i + 1} ---")
            text_parts.append(page.extract_text())
        
        return "\n".join(text_parts)
        
    except Exception as e:
        return f"Failed to extract PDF text: {str(e)}"


def process_image(url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process image: OCR text extraction and basic analysis
    Returns dict with text, dimensions, format
    """
    if not HAS_IMAGE:
        return {"error": "Image libraries not available"}
    
    try:
        response = requests.get(url, headers=headers or {}, timeout=30)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        
        result = {
            "url": url,
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
        }
        
        # Try OCR if available
        try:
            text = pytesseract.image_to_string(image)
            result["ocr_text"] = text
        except:
            result["ocr_text"] = "OCR not available"
        
        # Convert to base64 for potential submission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or 'PNG')
        result["base64"] = base64.b64encode(buffer.getvalue()).decode()
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


def create_visualization(data: pd.DataFrame, chart_type: str = "auto") -> str:
    """
    Create visualization from dataframe
    Returns base64 encoded image
    """
    if not HAS_VIZ:
        return None
    
    try:
        plt.figure(figsize=(10, 6))
        
        if chart_type == "auto":
            # Auto-detect chart type
            if len(data.columns) == 2:
                # Two columns: likely x-y plot
                data.plot(x=data.columns[0], y=data.columns[1], kind='line')
            elif len(data.columns) == 1:
                # Single column: histogram
                data[data.columns[0]].hist(bins=30)
            else:
                # Multiple columns: correlation heatmap
                sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        
        elif chart_type == "line":
            data.plot(kind='line')
        elif chart_type == "bar":
            data.plot(kind='bar')
        elif chart_type == "scatter":
            if len(data.columns) >= 2:
                data.plot(x=data.columns[0], y=data.columns[1], kind='scatter')
        elif chart_type == "hist":
            data.hist()
        elif chart_type == "heatmap":
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return None


def call_api(url: str, method: str = "GET", headers: Optional[Dict] = None, 
             params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make API call with custom headers
    Returns parsed response
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            params=params,
            json=json_data,
            timeout=30
        )
        response.raise_for_status()
        
        # Try to parse as JSON
        try:
            data = response.json()
            return {"success": True, "data": data, "text": response.text[:1000]}
        except:
            return {"success": True, "text": response.text[:5000]}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_api_urls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract API endpoints mentioned in text"""
    api_info = []
    
    # Look for API patterns
    api_pattern = r"(GET|POST|PUT|DELETE)\s+(https?://[^\s\"'<>]+)"
    matches = re.findall(api_pattern, text, re.IGNORECASE)
    
    for method, url in matches:
        api_info.append({"method": method.upper(), "url": url})
    
    # Also look for plain API URLs
    url_pattern = r"API.*?(https?://[^\s\"'<>]+)"
    matches = re.findall(url_pattern, text, re.IGNORECASE)
    
    for url in matches:
        if not any(api["url"] == url for api in api_info):
            api_info.append({"method": "GET", "url": url})
    
    return api_info



