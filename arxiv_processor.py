import requests
import tarfile
import tempfile
import shutil
import re
import subprocess
import sys
import json
import time
import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FinalArXivProcessor:
    """Clean processor with natural Gemini extraction and main sections only."""

    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.api_calls_made = 0
        self.last_api_call = 0
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini."""
        try:
            import google.generativeai as genai
        except ImportError:
            print("Installing google-generativeai...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        print("Gemini 2.5 Flash-Lite Preview 06-17 initialized")

    def _smart_rate_limit(self):
        """Smart rate limiting with 1-minute wait on limit hit."""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call

        # Check if we're approaching per-minute limit (15 RPM)
        if self.api_calls_made > 0 and self.api_calls_made % 14 == 0:
            print("Approaching rate limit, waiting 1 minute...")
            time.sleep(61)  # Wait 1 minute + buffer
            self.last_api_call = time.time()
        elif time_since_last < 4:
            wait_time = 4 - time_since_last
            print(f"Rate limit: {wait_time:.1f}s...")
            time.sleep(wait_time)

        self.last_api_call = time.time()
        self.api_calls_made += 1

    def basic_clean(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""

        # Remove LaTeX environments and math
        text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', ' ', text, flags=re.DOTALL)
        text = re.sub(r'\$\$[^$]*\$\$|\$[^$]*\$', ' ', text)
        text = re.sub(r'\\cite[^}]*\}|\\ref[^}]*\}|\\label[^}]*\}', ' ', text)

        # Preserve content from commands
        text = re.sub(r'\\text[a-z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)

        # Clean special characters
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_metadata_focused(self, content: str) -> Dict:
        """Extract title via regex and all authors from LaTeX content."""
        print("Extracting title and all authors via regex…")

        metadata = {"title": "", "authors": [], "abstract": ""}

        # 1. Title extraction
        title_match = re.search(r'\\title\{([^}]*)\}', content)
        if title_match:
            metadata["title"] = self.basic_clean(title_match.group(1))
        else:
            metadata["title"] = "Title not found"

        # 2. Raw author blocks (anywhere in document)
        raw_authors = re.findall(r'\\author\{([^}]*)\}', content, re.DOTALL)

        # 3. Split and clean each author name
        authors = []
        for block in raw_authors:
            # split on \and, commas, or the word "and"
            for part in re.split(r'\\and|,| and ', block):
                name = self.basic_clean(part)
                if name and name not in authors:
                    authors.append(name)
        metadata["authors"] = authors

        # 4. Abstract extraction (unchanged)
        abstract_text = ""
        for pat in [
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            r'\\abstract\{([^}]*)\}'
        ]:
            m = re.search(pat, content, re.DOTALL)
            if m:
                abstract_text = m.group(1)
                break
        metadata["abstract"] = (
            self.basic_clean(abstract_text)
            if abstract_text else ""
        )

        print(f"Extracted Title: {metadata['title']}")
        print(f"Extracted Authors: {metadata['authors']}")
        return metadata

    def _fallback_metadata(self, content: str) -> Dict:
        """Simple fallback extraction."""
        metadata = {"title": "Title not found", "authors": [], "abstract": ""}

        # Title
        title_match = re.search(r'\\title\{([^{}]+)\}', content)
        if title_match:
            metadata['title'] = self.basic_clean(title_match.group(1))

        # Abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
        if abstract_match:
            metadata['abstract'] = self.basic_clean(abstract_match.group(1))

        return metadata

    def find_main_sections_only(self, content: str) -> List[tuple]:
        """Find ONLY main sections (\\section level), ignore subsections."""
        print("Finding main sections only...")

        # Only look for main sections, ignore subsections
        pattern = r'\\section\*?\{([^}]+)\}'
        matches = list(re.finditer(pattern, content, re.IGNORECASE))

        sections = []
        for i, match in enumerate(matches):
            title = self.basic_clean(match.group(1))

            # Filter out reference/bibliography sections
            if (len(title) > 2 and len(title) < 150 and
                title.lower() not in ['references', 'bibliography', 'acknowledgments']):

                content_start = match.end()

                # Find content end (next section or document end)
                if i + 1 < len(matches):
                    content_end = matches[i + 1].start()
                else:
                    content_end = len(content)
                    for marker in [r'\\end\{document\}', r'\\bibliography', r'\\begin\{thebibliography\}']:
                        marker_match = re.search(marker, content[content_start:])
                        if marker_match:
                            content_end = content_start + marker_match.start()
                            break

                # Extract and clean content
                section_content = content[content_start:content_end]
                clean_content = self.basic_clean(section_content)
                word_count = len(clean_content.split())

                # Only include substantial sections
                if word_count >= 50:
                    sections.append((title, clean_content, word_count))
                    print(f"Main section: {title} ({word_count} words)")

        print(f"Found {len(sections)} main sections")
        return sections

    def summarize_section(self, text: str, title: str = "") -> str:
        """Summarize section content."""
        word_count = len(text.split())

        # Use basic summary for very short content
        if word_count < 80:
            return self._basic_summary(text)

        try:
            self._smart_rate_limit()

            prompt = f"""Summarize this academic paper section for audio playback:

Section: {title}
Content: {text[:2000]}

Create a clear 200-300 word summary that:
- Explains the main points in simple terms
- Flows naturally when spoken aloud
- Uses everyday language where possible
- Avoids symbols, equations, and technical jargon

Summary:"""

            response = self.gemini_model.generate_content(prompt)
            summary = response.text.strip()

            # Clean for text-to-speech
            summary = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', summary)
            replacements = {
                ' e.g.': ' for example', ' i.e.': ' that is',
                ' etc.': ' and so on', ' vs.': ' versus',
                '&': ' and', '%': ' percent'
            }

            for old, new in replacements.items():
                summary = summary.replace(old, new)

            return re.sub(r'\s+', ' ', summary).strip()

        except Exception as e:
            print(f"Summarization error: {e}")
            return self._basic_summary(text)

    def _basic_summary(self, text: str) -> str:
        """Basic extractive summary."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) <= 4:
            return ' '.join(sentences)
        else:
            # Take key sentences
            selected = [
                sentences[0],  # First
                sentences[len(sentences)//3],  # Early
                sentences[2*len(sentences)//3],  # Late
                sentences[-1]  # Last
            ]
            return ' '.join(selected)

    def download_paper(self, arxiv_id: str) -> Optional[str]:
        """Download paper - handles both compressed archives and single TeX files."""
        try:
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id.split('/')[-1].replace('.pdf', ''))
            print(f"Downloading {arxiv_id}...")

            url = f"https://arxiv.org/e-print/{arxiv_id}"
            temp_dir = Path(tempfile.mkdtemp())

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Save the downloaded content
            download_path = temp_dir / "downloaded_file"
            with open(download_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            # Check if it's a compressed file or plain text
            try:
                # First try gzip decompression (single file)
                import gzip
                try:
                    with gzip.open(download_path, 'rt', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        print("Successfully decompressed single gzip file")
                        # Verify it looks like a TeX file
                        tex_markers = [
                            r'\documentclass', r'\begin{document}', r'\title{', r'\author{',
                            r'\section{', r'\usepackage', r'\maketitle', r'\abstract',
                            r'\begin{abstract}', r'\end{document}', r'\textbf{', r'\emph{',
                            r'\begin{equation}', r'\begin{align}', r'$', r'\[', r'\]'
                        ]
                        found_markers = [marker for marker in tex_markers if marker in content]
                        if found_markers or content.strip().startswith('%') or '\\' in content:
                            print(f"Found TeX markers in gzip file: {found_markers}")
                            shutil.rmtree(temp_dir)
                            return content
                except Exception as gzip_error:
                    print(f"Not a gzip file: {gzip_error}")

                # Try to open as compressed archive (tar)
                with tarfile.open(download_path, 'r:*') as tar:
                    extract_dir = temp_dir / "extracted"
                    extract_dir.mkdir()
                    tar.extractall(extract_dir)

                    # Find main tex file in extracted content
                    tex_files = list(extract_dir.glob("*.tex"))
                    if not tex_files:
                        # Look in subdirectories too
                        tex_files = list(extract_dir.rglob("*.tex"))

                    if not tex_files:
                        print("No TeX files found in archive")
                        return None

                    best_content = None
                    best_score = 0

                    for tex_file in tex_files:
                        try:
                            content = tex_file.read_text(encoding='utf-8', errors='ignore')
                            score = self._score_tex_file(content)
                            if score > best_score:
                                best_score = score
                                best_content = content
                        except:
                            continue

                    shutil.rmtree(temp_dir)
                    return best_content

            except (tarfile.ReadError, Exception):
                # If it's not a compressed archive, treat it as a single TeX file
                print("File appears to be a single TeX file, not an archive")
                try:
                    # Try different encodings
                    content = None
                    encodings = ['utf-8', 'latin-1', 'ascii']

                    for encoding in encodings:
                        try:
                            content = download_path.read_text(encoding=encoding, errors='ignore')
                            print(f"Successfully read file with {encoding} encoding")
                            break
                        except Exception as e:
                            print(f"Failed to read with {encoding}: {e}")
                            continue

                    if content is None:
                        print("Could not read file with any encoding")
                        shutil.rmtree(temp_dir)
                        return None

                    # Debug: Print first 500 characters
                    print(f"File content preview: {content[:500]}")

                    # More flexible TeX validation - look for any TeX patterns
                    tex_markers = [
                        r'\documentclass', r'\begin{document}', r'\title{', r'\author{',
                        r'\section{', r'\usepackage', r'\maketitle', r'\abstract',
                        r'\begin{abstract}', r'\end{document}', r'\textbf{', r'\emph{',
                        # Math patterns common in papers
                        r'\begin{equation}', r'\begin{align}', r'$', r'\[', r'\]'
                    ]

                    found_markers = [marker for marker in tex_markers if marker in content]
                    print(f"Found TeX markers: {found_markers}")

                    # Be more lenient - if we find any TeX-like content, accept it
                    if found_markers or content.strip().startswith('%') or '\\' in content:
                        print("File appears to contain TeX content")
                        shutil.rmtree(temp_dir)
                        return content
                    else:
                        print("Downloaded file doesn't contain recognizable TeX content")
                        print(f"File size: {len(content)} characters")
                        print(f"First 200 chars: {repr(content[:200])}")
                        shutil.rmtree(temp_dir)
                        return None

                except Exception as e:
                    print(f"Error reading as text file: {e}")
                    shutil.rmtree(temp_dir)
                    return None

        except Exception as e:
            print(f"Download error: {e}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def _score_tex_file(self, content: str) -> float:
        """Score tex file."""
        score = 0.0
        if r'\documentclass' in content: score += 30
        if r'\begin{document}' in content: score += 30
        if r'\title{' in content: score += 25
        if r'\author{' in content or r'\name{' in content: score += 20
        if r'\abstract' in content: score += 25
        score += len(re.findall(r'\\section\{', content)) * 8
        return score

def process_paper(arxiv_id: str, gemini_api_key: str) -> Optional[Dict]:
    """Process paper with clean, unrestricted approach."""
    processor = FinalArXivProcessor(gemini_api_key)

    content = processor.download_paper(arxiv_id)
    if not content:
        return None

    print(f"Paper downloaded: {len(content)} characters")

    # Extract metadata naturally
    metadata = processor.extract_metadata_focused(content)

    result = OrderedDict()
    result['arxiv_id'] = arxiv_id
    result['title'] = metadata['title']
    result['authors'] = metadata['authors']

    # Process abstract
    if metadata['abstract'] and len(metadata['abstract'].split()) > 20:
        abstract_summary = processor.summarize_section(metadata['abstract'], "Abstract")
        result['abstract'] = {
            'content': metadata['abstract'],
            'summary': abstract_summary,
            'word_count': len(metadata['abstract'].split())
        }
        print(f"Abstract processed: {len(metadata['abstract'].split())} words")

    # Find and process ONLY main sections
    main_sections = processor.find_main_sections_only(content)

    if main_sections:
        # Process all main sections (no arbitrary limits)
        for i, (title, content, word_count) in enumerate(main_sections):
            print(f"Processing main section {i+1}: {title}")

            summary = processor.summarize_section(content, title)
            result[f"section_{i+1}"] = {
                'title': title,
                'summary': summary,
                'word_count': word_count
            }

    result['api_calls_used'] = processor.api_calls_made
    result['main_sections_processed'] = len([k for k in result.keys() if k.startswith('section_')])

    print(f"\nProcessing complete!")
    print(f"Title: {result['title']}")
    print(f"Authors: {', '.join(result['authors']) if result['authors'] else 'None found'}")
    print(f"Main sections: {result['main_sections_processed']}")
    print(f"Total API calls: {processor.api_calls_made}")

    return result

if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyBkIfrdK6-9H-DaqcMQT2lSRXHXzPE0t1Y"  # Your API key

    result = process_paper("2501.18987v1", GEMINI_API_KEY)

    if result:
        filename = f"{result['arxiv_id']}_final.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {filename}")
 