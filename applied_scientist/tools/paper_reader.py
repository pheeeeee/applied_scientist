from __future__ import annotations

import io
import json
import re

from applied_scientist.tools.base import Tool


class SearchPapers(Tool):
    name = "search_papers"
    description = "Search academic papers via Semantic Scholar. Free, no API key needed."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max papers to return",
                            "default": 10},
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> str:
        try:
            import requests
        except ImportError:
            return "ERROR: requests not installed. Run: pip install applied-scientist[search]"
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": max_results,
                      "fields": "title,authors,year,abstract,url"}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            papers = []
            for p in resp.json().get("data", []):
                papers.append({
                    "title": p.get("title", ""),
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "year": p.get("year"),
                    "abstract": p.get("abstract", ""),
                    "url": p.get("url", ""),
                })
            return json.dumps(papers, indent=2)
        except Exception as e:
            return f"ERROR: Paper search failed: {e}"


class ReadPaper(Tool):
    name = "read_paper"
    description = "Read and extract text from an arXiv paper or web page."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch (arXiv, PDF, or web)"},
            "prompt": {"type": "string", "description": "Optional focus prompt for extraction"},
        },
        "required": ["url"],
    }

    def execute(self, url: str, prompt: str = None) -> str:
        try:
            import requests
        except ImportError:
            return "ERROR: requests not installed. Run: pip install applied-scientist[search]"
        try:
            # Convert arXiv abstract URL to PDF URL
            if "arxiv.org/abs/" in url:
                url = url.replace("/abs/", "/pdf/") + ".pdf"

            is_pdf = url.endswith(".pdf") or "arxiv.org/pdf/" in url

            resp = requests.get(url, timeout=60, headers={
                "User-Agent": "AppliedScientist/0.1 (research automation)"
            })
            resp.raise_for_status()

            if is_pdf:
                text = self._extract_pdf(resp.content)
            else:
                text = self._extract_html(resp.text)

            # Truncate to manageable size
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... truncated ...]"

            if prompt:
                text = f"Focus on: {prompt}\n\n{text}"

            return text

        except Exception as e:
            return f"ERROR: Failed to read paper: {e}"

    def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages[:20]:  # Limit to 20 pages
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages)
        except ImportError:
            return "ERROR: PyPDF2 not installed. Run: pip install pypdf2"

    def _extract_html(self, html: str) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text
        except ImportError:
            return "ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
