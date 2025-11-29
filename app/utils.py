"""
utils.py 
"""

import re
import base64
import io
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urljoin, urlparse
from collections import Counter

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
        r"post.*?to\s+(https?://[^\s\"'<>]+/submit[^\s\"'<>]*)",
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
        if key.lower() in ['authorization', 'x-api-key', 'api-key', 'token', 'accept']:
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
    Process image: OCR text extraction, dominant color, and basic analysis
    Returns dict with text, dimensions, format, and color
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
        
        # Extract dominant color (GENERAL PURPOSE)
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image
            
            # Resize for faster processing
            rgb_image.thumbnail((200, 200))
            pixels = list(rgb_image.getdata())
            
            # Count color frequencies
            color_counter = Counter(pixels)
            dominant_color = color_counter.most_common(1)[0][0]
            
            # Convert to hex
            result["dominant_color"] = '#{:02x}{:02x}{:02x}'.format(
                dominant_color[0], dominant_color[1], dominant_color[2]
            ).lower()
            
            # Also get top 5 colors for more options
            top_colors = []
            for color, count in color_counter.most_common(5):
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]).lower()
                top_colors.append({"color": hex_color, "count": count})
            result["top_colors"] = top_colors
            
        except Exception as e:
            result["dominant_color"] = "#000000"
            result["color_error"] = str(e)
        
        # Convert to base64 for potential submission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or 'PNG')
        result["base64"] = base64.b64encode(buffer.getvalue()).decode()
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


def transcribe_audio(url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Download and transcribe audio file
    Returns dict with transcription and metadata
    If audio libraries not available, returns URL for manual transcription
    """
    if not HAS_AUDIO:
        print(f"  ⚠ Audio libraries not installed - skipping transcription")
        return {
            "error": "Audio libraries not available (SpeechRecognition/pydub not installed)", 
            "transcription": "Audio file found but transcription libraries not available",
            "url": url,
            "note": "Install with: pip install SpeechRecognition pydub (requires Python < 3.13)"
        }
    
    try:
        print(f"  Downloading audio from: {url}")
        resp = requests.get(url, headers=headers or {}, timeout=30)
        resp.raise_for_status()
        
        # Determine audio format from URL
        audio_format = 'mp3'
        if url.lower().endswith('.wav'):
            audio_format = 'wav'
        elif url.lower().endswith('.ogg'):
            audio_format = 'ogg'
        elif url.lower().endswith('.m4a'):
            audio_format = 'm4a'
        
        print(f"  Audio format: {audio_format}")
        
        # Load audio with pydub
        audio = AudioSegment.from_file(io.BytesIO(resp.content), format=audio_format)
        
        # Convert to WAV for speech recognition
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(wav_io) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        print(f"  Transcribing audio...")
        
        # Try multiple recognition methods
        transcription = None
        method_used = None
        
        # Try Google Speech Recognition (free, no API key)
        try:
            transcription = recognizer.recognize_google(audio_data)
            method_used = "google"
            print(f"  ✓ Transcribed (Google): {transcription}")
        except sr.UnknownValueError:
            print(f"  ✗ Google could not understand audio")
        except sr.RequestError as e:
            print(f"  ✗ Google service error: {e}")
        
        # Fallback to Sphinx (offline)
        if not transcription:
            try:
                transcription = recognizer.recognize_sphinx(audio_data)
                method_used = "sphinx"
                print(f"  ✓ Transcribed (Sphinx): {transcription}")
            except:
                print(f"  ✗ Sphinx failed")
        
        if transcription:
            return {
                "transcription": transcription.strip(),
                "method": method_used,
                "duration": len(audio) / 1000.0,  # in seconds
                "format": audio_format
            }
        else:
            return {
                "transcription": "unable to transcribe audio",
                "error": "All recognition methods failed"
            }
                
    except Exception as e:
        print(f"  ✗ Audio transcription error: {e}")
        return {"transcription": "unable to transcribe audio", "error": str(e)}


def normalize_dataframe_to_json(df: pd.DataFrame) -> list:
    """
    Normalize DataFrame to clean JSON format (GENERAL PURPOSE)
    Handles common data cleaning tasks automatically
    """
    try:
        df = df.copy()
        
        # 1. Strip whitespace from all string columns
        for col in df.select_dtypes(include=['object']).columns:
            if hasattr(df[col], 'str'):
                df[col] = df[col].str.strip()
        
        # 2. Standardize date columns (YYYY-MM-DD)
        date_keywords = ['date', 'joined', 'created', 'updated', 'time', 'timestamp']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass
        
        # 3. Auto-sort by logical columns (id first, then name, then dates)
        sort_cols = []
        if 'id' in df.columns:
            sort_cols.append('id')
        
        name_cols = [col for col in df.columns if 'name' in col.lower()]
        if name_cols:
            sort_cols.append(name_cols[0])
        
        date_cols = [col for col in df.columns if any(kw in col.lower() for kw in date_keywords)]
        if date_cols:
            sort_cols.append(date_cols[0])
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        # 4. Reset index
        df = df.reset_index(drop=True)
        
        # 5. Convert to list of dicts
        result = df.to_dict('records')
        
        # 6. Clean up types (no numpy types, handle NaN/None)
        for record in result:
            for key, value in list(record.items()):
                # Convert numpy types
                if hasattr(value, 'item'):
                    record[key] = value.item()
                # Handle NaN
                elif pd.isna(value):
                    record[key] = None
                # Convert float to int if it's a whole number
                elif isinstance(value, float) and value.is_integer():
                    record[key] = int(value)
        
        return result
        
    except Exception as e:
        print(f"  ✗ DataFrame normalization error: {e}")
        return []


def parse_github_api_response(data: Dict[str, Any], filter_prefix: str = "", 
                               filter_extension: str = "") -> Dict[str, Any]:
    """
    Parse GitHub API tree response (GENERAL PURPOSE)
    Filters files by prefix and extension if provided
    """
    try:
        tree_items = data.get('tree', [])
        
        files = []
        dirs = []
        
        for item in tree_items:
            path = item.get('path', '')
            item_type = item.get('type', '')
            
            # Apply filters if provided
            if filter_prefix and not path.startswith(filter_prefix):
                continue
            
            if filter_extension and not path.endswith(filter_extension):
                continue
            
            if item_type == 'blob':  # File
                files.append({
                    'path': path,
                    'size': item.get('size', 0),
                    'sha': item.get('sha', '')
                })
            elif item_type == 'tree':  # Directory
                dirs.append(path)
        
        return {
            'total_files': len(files),
            'total_dirs': len(dirs),
            'files': files,
            'directories': dirs,
            'filter_prefix': filter_prefix,
            'filter_extension': filter_extension
        }
        
    except Exception as e:
        return {'error': str(e), 'total_files': 0}


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
                data.plot(x=data.columns[0], y=data.columns[1], kind='line')
            elif len(data.columns) == 1:
                data[data.columns[0]].hist(bins=30)
            else:
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


def find_audio_sources(html: str, base_url: str) -> List[str]:
    """Find all audio sources in HTML"""
    soup = BeautifulSoup(html, "html.parser")
    audio_urls = []
    
    # Find <audio> tags
    for audio in soup.find_all('audio'):
        src = audio.get('src')
        if src:
            audio_urls.append(normalize_url(base_url, src))
    
    # Find <source> tags inside <audio>
    for source in soup.find_all('source'):
        src = source.get('src')
        if src and source.parent and source.parent.name == 'audio':
            audio_urls.append(normalize_url(base_url, src))
    
    # Find links to audio files
    audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']
    for a in soup.find_all('a', href=True):
        href = a['href']
        if any(href.lower().endswith(ext) for ext in audio_extensions):
            audio_urls.append(normalize_url(base_url, href))
    
    return list(set(audio_urls))  # Remove duplicates
