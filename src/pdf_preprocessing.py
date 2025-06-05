import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# PDF processing libraries
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import fitz  # PyMuPDF - alternative for better text extraction

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "paragraph"  # paragraph, section, table, list
    word_count: int = 0
    
    def __post_init__(self):
        if self.word_count == 0:
            try:
                self.word_count = len(word_tokenize(self.text))
            except:
                # Fallback if tokenization fails
                self.word_count = len(self.text.split())

class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)
        self.extracted_texts = {}
        self.toc_sections = {}  # Store TOC information
        
    def extract_text_pypdf2(self, pdf_path: str) -> Dict[int, str]:
        """Extract text using PyPDF2"""
        texts = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    texts[page_num + 1] = text
        except Exception as e:
            print(f"Error extracting text from {pdf_path} with PyPDF2: {e}")
        return texts
    
    def extract_text_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer with better layout preservation"""
        try:
            # LAParams for better layout analysis
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                all_texts=False
            )
            text = extract_text(pdf_path, laparams=laparams)
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path} with pdfminer: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract text using PyMuPDF with better structure preservation"""
        pages_data = []
        try:
            doc = fitz.open(pdf_path)
            
            # Extract PDF outline/bookmarks
            outline = self._extract_pdf_outline(doc)
            
            # Extract Table of Contents from content
            toc_content = self._extract_toc_from_content(doc)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with formatting info
                text_dict = page.get_text("dict")
                
                # Extract plain text
                text = page.get_text()
                
                # Enhanced structure analysis
                structure = self._analyze_page_structure_enhanced(text_dict, page_num + 1, toc_content)
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text,
                    'structure': structure,
                    'outline': outline,
                    'toc_sections': toc_content
                }
                pages_data.append(page_data)
            
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path} with PyMuPDF: {e}")
        
        return pages_data
    
    def _extract_pdf_outline(self, doc) -> List[Dict]:
        """Extract PDF bookmarks/outline structure"""
        outline = []
        try:
            toc = doc.get_toc()  # Get table of contents
            for level, title, page_num in toc:
                outline.append({
                    'level': level,
                    'title': title.strip(),
                    'page': page_num,
                    'type': 'bookmark'
                })
            if outline:
                print(f"Extracted {len(outline)} bookmark entries")
        except Exception as e:
            print(f"No PDF outline found: {e}")
        
        return outline
    
    def _extract_toc_from_content(self, doc) -> Dict[str, Dict]:
        """Extract Table of Contents from document content"""
        toc_sections = {}
        
        try:
            print("Searching for TOC in first 5 pages...")
            # Look for "Sommario" or "SOMMARIO" in the document
            for page_num in range(min(5, len(doc))):
                print(f"  Checking page {page_num + 1}...")
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if re.search(r'SOMMARIO|Sommario|INDICE|Indice', text, re.IGNORECASE):
                    print(f"Found TOC on page {page_num + 1}")
                    toc_sections = self._parse_toc_content(text, page_num + 1)
                    
                    if toc_sections:
                        print(f"Successfully extracted {len(toc_sections)} sections from TOC")
                    else:
                        print("No sections were extracted from the TOC page")
                    break
            else:
                print("No TOC page found in first 5 pages")
                
        except Exception as e:
            print(f"Error extracting TOC: {e}")
        
        return toc_sections
    
    def _parse_toc_content(self, text: str, page_num: int) -> Dict[str, Dict]:
        """Enhanced TOC parsing with better title extraction"""
        sections = {}
        
        print(f"Parsing TOC content (length: {len(text)} characters)...")
        
        # More focused patterns for Italian TOC
        toc_patterns = [
            # "3. INTRODUZIONE ................... 3"
            r'(\d+(?:\.\d+)*\.?)\s+([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ][A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ\s]{2,60}?)\s*\.{3,}\s*(\d+)',
            # "INTRODUZIONE ................... 3"  
            r'^([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ][A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ\s]{2,60}?)\s*\.{3,}\s*(\d+)\s*$',
            # "3. INTRODUZIONE"
            r'(\d+(?:\.\d+)*\.?)\s+([A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ][A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ\s]{2,60}?)(?:\s|$)',
        ]
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            
            # Skip obvious non-TOC lines
            if any(skip in line.lower() for skip in ['sommario', 'indice', 'figura', 'tabella', 'pag.']):
                continue
            
            for pattern in toc_patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 3:  # number, title, page
                        section_num, title, target_page = groups
                        title = title.strip()
                        # Clean the title
                        title = re.sub(r'\s+', ' ', title)
                        title = re.sub(r'[.]+$', '', title)  # Remove trailing dots
                        
                        try:
                            sections[section_num.rstrip('.')] = {
                                'title': title,
                                'page': int(target_page),
                                'section_number': section_num.rstrip('.')
                            }
                            print(f"    Found numbered section: {section_num} - {title} (page {target_page})")
                        except ValueError:
                            continue
                            
                    elif len(groups) == 2:
                        if groups[1].isdigit():  # title, page
                            title, target_page = groups
                            title = title.strip()
                            title = re.sub(r'\s+', ' ', title)
                            title = re.sub(r'[.]+$', '', title)
                            
                            try:
                                sections[f"section_{len(sections)+1}"] = {
                                    'title': title,
                                    'page': int(target_page),
                                    'section_number': None
                                }
                                print(f"    Found unnumbered section: {title} (page {target_page})")
                            except ValueError:
                                continue
                        else:  # number, title
                            section_num, title = groups
                            title = title.strip()
                            title = re.sub(r'\s+', ' ', title)
                            title = re.sub(r'[.]+$', '', title)
                            
                            sections[section_num.rstrip('.')] = {
                                'title': title,
                                'page': None,
                                'section_number': section_num.rstrip('.')
                            }
                            print(f"    Found section without page: {section_num} - {title}")
                    break
        
        print(f"TOC parsing completed. Extracted {len(sections)} sections.")
        return sections
    
    def _analyze_page_structure_enhanced(self, text_dict: Dict, page_num: int, toc_sections: Dict) -> Dict:
        """Enhanced structure analysis with better section detection"""
        structure = {
            'headers': [],
            'paragraphs': [],
            'lists': [],
            'tables': [],
            'section_titles': [],
            'figure_captions': [],
            'current_section': None
        }
        
        # Find which section this page belongs to based on TOC
        current_section = self._find_current_section(page_num, toc_sections)
        structure['current_section'] = current_section
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_styles = []
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            line_text += text + " "
                            line_styles.append({
                                'text': text,
                                'size': font_size,
                                'flags': font_flags,
                                'is_bold': bool(font_flags & 2**4),
                                'is_italic': bool(font_flags & 2**6)
                            })
                    
                    line_text = line_text.strip()
                    if line_text:
                        # Priority classification
                        if self._is_figure_caption(line_text) or self._is_table_caption(line_text):
                            structure['figure_captions'].append(line_text)
                        elif self._is_header_footer(line_text):
                            continue
                        elif self._is_list_item(line_text):
                            structure['lists'].append(line_text)
                        else:
                            # Check for TOC matches first (highest priority for real sections)
                            matched_section = self._match_with_toc(line_text, toc_sections)
                            
                            if matched_section:
                                structure['section_titles'].append({
                                    'text': line_text,
                                    'toc_title': matched_section['title'],
                                    'section_number': matched_section.get('section_number'),
                                    'is_toc_match': True
                                })
                                structure['headers'].append(line_text)
                            elif self._is_section_title_enhanced(line_text, line_styles, toc_sections):
                                # Only add if it's not a figure caption
                                if not (self._is_figure_caption(line_text) or self._is_table_caption(line_text)):
                                    structure['section_titles'].append({
                                        'text': line_text,
                                        'toc_title': None,
                                        'section_number': None,
                                        'is_toc_match': False
                                    })
                                    structure['headers'].append(line_text)
                            else:
                                # Regular paragraph content
                                structure['paragraphs'].append(line_text)
        
        return structure
    
    def _find_current_section(self, page_num: int, toc_sections: Dict) -> Optional[Dict]:
        """Find which section the current page belongs to based on TOC"""
        current_section = None
        
        for section_key, section_info in toc_sections.items():
            section_page = section_info.get('page')
            if section_page and section_page <= page_num:
                if not current_section or section_page > current_section.get('page', 0):
                    current_section = section_info
        
        return current_section
    
    def _match_with_toc(self, text: str, toc_sections: Dict) -> Optional[Dict]:
        """Enhanced TOC matching with better text normalization"""
        # Clean the input text more thoroughly
        text_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', text).strip()
        text_clean = re.sub(r'\s+', ' ', text_clean).upper()
        
        # Remove common prefixes that might interfere
        text_clean = re.sub(r'^(FIGURA|TABELLA|STEP|F\s+IGURA)\s*\d+\s*[-–]?\s*', '', text_clean)
        
        for section_key, section_info in toc_sections.items():
            toc_title = section_info['title'].upper().strip()
            toc_title = re.sub(r'\s+', ' ', toc_title)
            
            # Exact match after cleaning
            if text_clean == toc_title:
                return section_info
            
            # Partial match for longer titles
            if len(text_clean) > 5 and len(toc_title) > 5:
                # Check if the text contains the main keywords from TOC
                toc_words = set(toc_title.split())
                text_words = set(text_clean.split())
                
                # Remove common stop words
                stop_words = {'PER', 'LA', 'IL', 'DI', 'DEL', 'DELLA', 'E', 'ED', 'AL', 'ALLA'}
                toc_words = toc_words - stop_words
                text_words = text_words - stop_words
                
                if toc_words and len(toc_words & text_words) >= min(2, len(toc_words) * 0.6):
                    return section_info
        
        return None
    
    def _is_section_title_enhanced(self, text: str, styles: List[Dict], toc_sections: Dict) -> bool:
        """Enhanced section title detection that excludes captions"""
        
        # Exclude figure/table captions first (highest priority)
        if self._is_figure_caption(text) or self._is_table_caption(text):
            return False
        
        # Exclude page references
        if self._is_page_reference(text):
            return False
        
        # Check if it's already matched with TOC (handled elsewhere)
        if self._match_with_toc(text, toc_sections):
            return True
        
        # Heuristic checks
        return self._is_section_title_heuristic(text, styles)
    
    def _is_section_title_heuristic(self, text: str, styles: List[Dict]) -> bool:
        """Heuristic-based section title detection"""
        
        # First, exclude figure captions and other non-section elements
        if self._is_figure_caption(text):
            return False
        
        if self._is_table_caption(text):
            return False
        
        if self._is_page_reference(text):
            return False
        
        # Check for numbered sections
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', text):
            return True
        
        # Check for all caps titles
        if text.isupper() and len(text.split()) > 1 and len(text) < 100:
            return True
        
        # Check formatting
        if styles:
            avg_font_size = sum(s.get('size', 0) for s in styles) / len(styles)
            has_bold = any(s.get('is_bold', False) for s in styles)
            
            if (avg_font_size > 12 or has_bold) and self._looks_like_title(text):
                return True
        
        return False
    
    def _is_figure_caption(self, text: str) -> bool:
        """Enhanced figure caption detection"""
        figure_patterns = [
            r'^F\s*IGURA\s+\d+',  # "F IGURA 1", "FIGURA 2" etc.
            r'^FIGURA\s+\d+',
            r'^Fig\.\s+\d+',
            r'^Figure\s+\d+',
            r'^\s*–\s*[A-Z][A-Z\s]+$',  # Lines starting with dash
            r'^F\s+IGURA\s+\d+\s*[-–]\s*[A-Z]',  # "F IGURA 1 – REGISTRAZIONE"
            r'FIGURA\s+\d+\s*[-–]',  # Any "FIGURA X –" pattern
        ]
        
        for pattern in figure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_table_caption(self, text: str) -> bool:
        """Detect if text is a table caption"""
        table_patterns = [
            r'^TABELLA\s+\d+',
            r'^TAB\.\s+\d+',
            r'^Table\s+\d+',
            r'Tabella riepilogativa',
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_page_reference(self, text: str) -> bool:
        """Detect if text is a page reference or navigation element"""
        page_ref_patterns = [
            r'^\d+$',  # Just numbers
            r'^\.\.\.\.\.\.\.\.\.\.\.\s*\d+$',  # Dots followed by page number
            r'^[A-Z\s]+\s+\.\.\.\.\.\.\.\.\.\.\.\s*\d+$',  # Title with dots and page number (TOC entry)
            r'INDICE DELLE FIGURE',
            r'INDICE DELLE TABELLE',
        ]
        
        for pattern in page_ref_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _looks_like_title(self, text: str) -> bool:
        """Enhanced title detection that excludes captions"""
        
        # Exclude figure/table captions first
        if self._is_figure_caption(text) or self._is_table_caption(text):
            return False
        
        # Original title patterns for EmPULIA content
        title_patterns = [
            r'REGISTRAZIONE|GESTIONE|ISTRUZIONI|ACCESSO|MODIFICA|CESSAZIONE',
            r'INTRODUZIONE|SOMMARIO|LEGENDA|PROCEDURA|SISTEMA',
            r'PRIMO\s+ACCESSO|CAMBIO\s+PASSWORD|LISTA\s+ATTIVITÀ',
            r'PROFILAZIONE|TRATTAMENTO|RIEPILOGO',
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Title case check (but exclude captions)
        words = text.split()
        if len(words) > 1 and all(word[0].isupper() if word else False for word in words[:3]):
            # Additional check: if it contains figure-related words, it's not a title
            if not any(fig_word in text.upper() for fig_word in ['FIGURA', 'TABELLA', 'STEP']):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Enhanced list item detection"""
        list_patterns = [
            r'^\s*[•·▪▫◦‣⁃]\s+',
            r'^\s*[-–—]\s+',
            r'^\s*\d+[\.)]\s+',
            r'^\s*[a-zA-Z][\.)]\s+',
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_header_footer(self, text: str) -> bool:
        """Enhanced header/footer detection"""
        patterns = [
            r'Pag\.\s*\d+',
            r'Vers\.\s*\d+\.\d+',
            r'QUESTO DOCUMENTO È DI PROPRIETÀ',
            r'Manuale per.*?\d{2}/\d{2}/\d{4}',
            r'^\d+$',  # Just page numbers
            r'copyright|©|proprietà|innovapuglia',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Very short lines (likely headers/footers)
        if len(text.strip()) < 5:
            return True
        
        return False
    
    def process_all_pdfs(self, method: str = "pymupdf") -> Dict[str, List[Dict]]:
        """Process all PDFs in the directory"""
        results = {}
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return results
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            try:
                if method == "pymupdf":
                    pages_data = self.extract_text_pymupdf(str(pdf_file))
                    results[pdf_file.name] = pages_data
                elif method == "pdfminer":
                    text = self.extract_text_pdfminer(str(pdf_file))
                    results[pdf_file.name] = [{'text': text, 'page_number': 1}]
                else:  # pypdf2
                    page_texts = self.extract_text_pypdf2(str(pdf_file))
                    pages_data = [{'text': text, 'page_number': page_num} 
                                 for page_num, text in page_texts.items()]
                    results[pdf_file.name] = pages_data
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
        
        return results
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better header/footer and caption removal"""
        if not text or not text.strip():
            return ""
        
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers/footers
            if self._is_header_footer(line):
                continue
            
            # Skip figure and table captions
            if self._is_figure_caption(line) or self._is_table_caption(line):
                continue
            
            # Skip page references and TOC entries
            if self._is_page_reference(line):
                continue
            
            # Remove version numbers at the beginning of lines
            line = re.sub(r'^Vers\.\s*\d+\.\d+\s*', '', line)
            
            # Remove page references
            line = re.sub(r'Pag\.\s*\d+\s*', '', line)
            
            # Keep the line if it's substantial content
            if len(line) > 10:
                cleaned_lines.append(line)
        
        # Join lines and apply additional cleaning
        text = ' '.join(cleaned_lines)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        
        # Remove standalone dots and dashes
        text = re.sub(r'\s+[.]+\s+', ' ', text)
        text = re.sub(r'\s+[-]+\s+', ' ', text)
        
        return text.strip()

class TextChunker:
    """Enhanced chunker with better TOC-aware section processing"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sections_enhanced(self, pages_data: List[Dict], source_file: str) -> List[TextChunk]:
        """Enhanced section-based chunking with better title assignment"""
        chunks = []
        chunk_id = 0
        current_section_content = ""
        current_section_title = None
        
        # Get TOC sections
        toc_sections = {}
        for page_data in pages_data:
            if page_data.get('toc_sections'):
                toc_sections = page_data['toc_sections']
                break
        
        print(f"Using TOC with {len(toc_sections)} sections for chunking")
        
        for page_data in pages_data:
            structure = page_data.get('structure', {})
            page_num = page_data.get('page_number', 1)
            
            # Check if page belongs to a TOC section
            page_section = structure.get('current_section')
            if page_section and page_section.get('title'):
                page_section_title = page_section.get('title')
                
                # If we're entering a new TOC section
                if page_section_title != current_section_title:
                    # Save previous section
                    if current_section_content.strip():
                        chunks.append(TextChunk(
                            text=current_section_content.strip(),
                            chunk_id=f"{source_file}_section_{chunk_id}",
                            source_file=source_file,
                            page_number=page_num,
                            section_title=current_section_title,
                            chunk_type="toc_section"
                        ))
                        chunk_id += 1
                    
                    # Start new section
                    current_section_title = page_section_title
                    current_section_content = ""
            
            # Process detected section titles on this page
            section_titles = structure.get('section_titles', [])
            paragraphs = structure.get('paragraphs', [])
            
            # Handle explicit section titles found on the page
            for title_info in section_titles:
                if title_info.get('is_toc_match'):
                    # This is a real TOC section start
                    if current_section_content.strip():
                        chunks.append(TextChunk(
                            text=current_section_content.strip(),
                            chunk_id=f"{source_file}_section_{chunk_id}",
                            source_file=source_file,
                            page_number=page_num,
                            section_title=current_section_title,
                            chunk_type="toc_section"
                        ))
                        chunk_id += 1
                    
                    # Start new section with proper TOC title
                    current_section_title = title_info.get('toc_title') or title_info.get('text')
                    current_section_content = ""
            
            # Add paragraphs to current section
            page_content = " ".join(paragraphs)
            if page_content.strip():
                if current_section_content:
                    current_section_content += " " + page_content
                else:
                    current_section_content = page_content
                
                # Split if section gets too long
                while len(word_tokenize(current_section_content)) > self.chunk_size:
                    # Find a good split point
                    sentences = sent_tokenize(current_section_content)
                    chunk_text = ""
                    remaining_text = ""
                    
                    for i, sentence in enumerate(sentences):
                        if len(word_tokenize(chunk_text + " " + sentence)) <= self.chunk_size:
                            chunk_text += " " + sentence if chunk_text else sentence
                        else:
                            remaining_text = " ".join(sentences[i:])
                            break
                    
                    if chunk_text:
                        chunks.append(TextChunk(
                            text=chunk_text.strip(),
                            chunk_id=f"{source_file}_section_{chunk_id}",
                            source_file=source_file,
                            page_number=page_num,
                            section_title=current_section_title,
                            chunk_type="toc_section_split"
                        ))
                        chunk_id += 1
                        current_section_content = remaining_text
                    else:
                        break  # Avoid infinite loop
        
        # Add final section
        if current_section_content.strip():
            chunks.append(TextChunk(
                text=current_section_content.strip(),
                chunk_id=f"{source_file}_section_{chunk_id}",
                source_file=source_file,
                page_number=pages_data[-1].get('page_number', 1) if pages_data else 1,
                section_title=current_section_title,
                chunk_type="toc_section"
            ))
        
        return chunks
    
    def chunk_by_sentences(self, text: str, source_file: str, page_number: int = None) -> List[TextChunk]:
        """Chunk text by sentences, respecting chunk size limits"""
        if not text or not text.strip():
            return []
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(word_tokenize(current_chunk + " " + sentence)) > self.chunk_size:
                if current_chunk:  # Save current chunk
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        chunk_id=f"{source_file}_chunk_{chunk_id}",
                        source_file=source_file,
                        page_number=page_number,
                        chunk_type="sentence_based"
                    ))
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                chunk_id=f"{source_file}_chunk_{chunk_id}",
                source_file=source_file,
                page_number=page_number,
                chunk_type="sentence_based"
            ))
        
        return chunks
    
    # Keep existing methods for backward compatibility
    def chunk_by_sections(self, pages_data: List[Dict], source_file: str) -> List[TextChunk]:
        """Use enhanced chunking by default"""
        return self.chunk_by_sections_enhanced(pages_data, source_file)

def main():
    """Enhanced main function with TOC processing"""
    # Configuration
    PDF_DIRECTORY = r"c:\Users\mikim\Desktop\Natural-Language-Processing-Project\data\pdfs"
    OUTPUT_DIRECTORY = r"c:\Users\mikim\Desktop\Natural-Language-Processing-Project\data\processed"
    
    # Create directories
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Check for PDFs
    pdf_files = list(Path(PDF_DIRECTORY).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}")
        print("Please add PDF files to the directory and run again.")
        return
    
    # Initialize enhanced processor
    pdf_processor = PDFProcessor(PDF_DIRECTORY)
    chunker = TextChunker(chunk_size=512, overlap=50)
    
    print("Starting enhanced PDF processing with TOC extraction...")
    print(f"Processing {len(pdf_files)} PDF files")
    
    try:
        pdf_results = pdf_processor.process_all_pdfs(method="pymupdf")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install PyMuPDF")
        return
    except Exception as e:
        print(f"Error processing PDFs: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not pdf_results:
        print("No PDFs were successfully processed.")
        return
    
    print(f"Successfully processed {len(pdf_results)} PDF files")
    
    all_chunks = []
    
    for pdf_name, pages_data in pdf_results.items():
        print(f"\nProcessing {pdf_name}...")
        
        if not pages_data:
            print(f"  No pages extracted from {pdf_name}")
            continue
        
        # Clean text for each page
        for page_data in pages_data:
            if 'text' in page_data:
                original_text = page_data['text']
                cleaned_text = pdf_processor.clean_text(original_text)
                page_data['text'] = cleaned_text
        
        # Use enhanced section-based chunking
        try:
            section_chunks = chunker.chunk_by_sections_enhanced(pages_data, pdf_name)
            print(f"  Created {len(section_chunks)} chunks")
        except Exception as e:
            print(f"  Error in section chunking: {e}")
            section_chunks = []
        
        # Fallback to sentence-based if no good sections found
        if not section_chunks:
            print(f"  No sections detected, using sentence-based chunking")
            sentence_chunks = []
            for page_data in pages_data:
                if 'text' in page_data and page_data['text']:
                    chunks = chunker.chunk_by_sentences(
                        page_data['text'], 
                        pdf_name, 
                        page_data.get('page_number')
                    )
                    sentence_chunks.extend(chunks)
            chosen_chunks = sentence_chunks
        else:
            chosen_chunks = section_chunks
        
        # Show section titles extracted
        section_titles = set(chunk.section_title for chunk in chosen_chunks if chunk.section_title)
        if section_titles:
            print(f"  Section titles found: {len(section_titles)}")
            for title in list(section_titles)[:3]:
                print(f"    - {title}")
        
        all_chunks.extend(chosen_chunks)
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks were created. Check your PDF files and try again.")
        return
    
    # Save enhanced results
    import json
    chunks_data = []
    for chunk in all_chunks:
        chunks_data.append({
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'source_file': chunk.source_file,
            'page_number': chunk.page_number,
            'section_title': chunk.section_title,
            'chunk_type': chunk.chunk_type,
            'word_count': chunk.word_count
        })
    
    output_file = os.path.join(OUTPUT_DIRECTORY, 'processed_chunks_toc_enhanced.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"Enhanced chunks saved to: {output_file}")
    except Exception as e:
        print(f"Error saving chunks: {e}")
        return
    
    # Enhanced statistics
    toc_chunks = [c for c in all_chunks if 'toc' in c.chunk_type]
    chunks_with_titles = [c for c in all_chunks if c.section_title]
    
    print(f"\nEnhanced Statistics:")
    print(f"  TOC-based chunks: {len(toc_chunks)}")
    print(f"  Chunks with section titles: {len(chunks_with_titles)}")
    print(f"  Average chunk size: {sum(c.word_count for c in all_chunks) / len(all_chunks):.1f} words")
    print(f"  Chunk types: {set(c.chunk_type for c in all_chunks)}")

if __name__ == "__main__":
    main()