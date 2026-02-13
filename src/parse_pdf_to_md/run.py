import base64
import os
import shutil
import glob
from io import BytesIO
from typing import List, Optional, Literal, Union
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber  # Used for quick page counting
import re
# Client libraries
import openai
import anthropic
import google.generativeai as genai

class VisionPDFParser:
    def __init__(
        self, 
        model_provider: Literal["openai", "anthropic", "google"] = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dir: str = "./parsing_output"
    ):
        self.provider = model_provider
        self.output_dir = output_dir
        self.api_key = api_key or os.environ.get(f"{model_provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API Key for {model_provider} is missing.")

        # Initialize Clients
        if self.provider == "openai":
            self.model = model_name or "gpt-4o"
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            self.model = model_name or "claude-3-5-sonnet-20240620"
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            self.model = model_name or "gemini-1.5-pro"
            genai.configure(api_key=self.api_key)

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_system_prompt(self) -> str:
        return """
        You are a specialized document digitization AI. Your goal is to convert this document image into a perfect Markdown representation for a RAG system.

        ### CORE INSTRUCTIONS
        1. **Output ONLY Markdown**. Do not provide introductory text, explanations, or code blocks (```markdown). Start directly with the content.
        2. **Preserve Headers**: Use #, ##, ### for titles to strictly maintain document hierarchy.
        3. **No Summarization**: Transcribe every single word exactly as it appears. Do not fix grammar or spelling.

        ### TABLE HANDLING RULES (CRITICAL)
        1. **Logical Rows vs. Visual Lines**: 
           - If a cell's text wraps to a new line visually, **DO NOT create a new Markdown row**. 
           - Keep it in the same cell. Join multi-line text with a single space or `<br>` tag.
           - A new row in the Markdown table should ONLY exist if there is a distinct horizontal grid line separating it from the row above.
           
        2. **Merged Cells**:
           - If a cell spans multiple columns (horizontally), repeat the value in each column or leave the extras empty, but ensure column alignment is perfect.
           - If a cell spans multiple rows (vertically), repeat the value in each row so that every row is semantically complete. (e.g., if "Section 4" applies to 5 rows, write "Section 4" in all 5 rows).
           
        3. **Empty Cells**: 
           - If a cell is visually empty but clearly implied to carry over the value from above (ditto marks or implicit grouping), **FILL IT IN**. RAG systems cannot "look up" to see context.

        4. **Structural Integrity**:
           - Every row MUST have the same number of pipes (|) as the header.
        """
        
    def _clean_markdown_output(self, text: str) -> str:
            """
            Removes ```markdown fences and other AI chatter.
            """
            if not text: return ""

            # 1. Remove the opening fence (case insensitive)
            # Matches ```markdown, ```md, or just ```
            text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)

            # 2. Remove the closing fence
            text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)

            return text.strip()
        
    def _call_llm(self, image: Image.Image) -> str:
        """Routes the image to the configured LLM provider."""
        try:
            if self.provider == "google":
                model = genai.GenerativeModel(self.model)
                response = model.generate_content([self._get_system_prompt(), image])
                return self._clean_markdown_output(response.text)
            
            # Base64 needed for OpenAI/Anthropic
            b64_img = self._image_to_base64(image)
            
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Transcribe this page."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]}
                    ],
                    temperature=0
                )
                return self._clean_markdown_output(esponse.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self._get_system_prompt(),
                    messages=[
                        {"role": "user", "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}},
                            {"type": "text", "text": "Transcribe this page."}
                        ]}
                    ]
                )
                return self._clean_markdown_output(response.content[0].text)
                
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            return None

    def _get_page_path(self, doc_name: str, page_num: int) -> str:
        """Returns the file path for a specific page's markdown."""
        clean_name = os.path.splitext(os.path.basename(doc_name))[0]
        folder = os.path.join(self.output_dir, clean_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"page_{page_num:04d}.md")

    def parse_pdf(self, pdf_path: str, pages_to_process: Optional[List[int]] = None, force_redo: bool = False):
        """
        Main Method.
        - If pages_to_process is None: Processes ALL pages (incremental resume).
        - If pages_to_process is [1, 5]: Processes ONLY pages 1 and 5.
        - If force_redo is True: Overwrites existing files.
        """
        doc_name = os.path.basename(pdf_path)
        print(f"📂 Processing: {doc_name}")

        # 1. Get Total Page Count (Lightweight)
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        # 2. Determine which pages to work on
        if pages_to_process:
            # User requested specific pages (e.g., [1, 5, 10])
            # Adjust to 1-based index if user passed 1-based, typically we assume 1-based for UI
            target_pages = pages_to_process
        else:
            # Default: Do all pages
            target_pages = range(1, total_pages + 1)

        print(f"Targeting {len(target_pages)} pages...")

        for page_num in target_pages:
            output_path = self._get_page_path(pdf_path, page_num)
            
            # 3. RESUME LOGIC: Check if file exists
            if os.path.exists(output_path) and not force_redo:
                print(f"  [Skip] Page {page_num} already exists.")
                continue

            print(f"  [Work] Parsing Page {page_num}...")
            
            try:
                # 4. Convert ONLY the single page we need (Memory Efficient)
                # pdf2image uses 1-based indexing for first_page/last_page
                images = convert_from_path(
                    pdf_path, 
                    first_page=page_num, 
                    last_page=page_num
                )
                
                if not images:
                    print(f"  [Error] Could not render image for page {page_num}")
                    continue
                    
                image = images[0]

                # 5. Call AI
                markdown_content = self._call_llm(image)
                
                if markdown_content:
                    # 6. INSTANT SAVE
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"\n\n")
                        f.write(markdown_content)
                    print(f"  [Save] Page {page_num} saved.")
                else:
                    print(f"  [Fail] LLM returned empty response for page {page_num}")

            except Exception as e:
                print(f"  [Crit] Failed on page {page_num}: {e}")

    def compile_final_markdown(self, pdf_path: str) -> str:
        """
        Stitches all page_XXX.md files into one big document.
        Useful after the job is 100% complete.
        """
        clean_name = os.path.splitext(os.path.basename(pdf_path))[0]
        folder = os.path.join(self.output_dir, clean_name)
        
        # Get all .md files sorted by name (which effectively sorts by page number due to 000X padding)
        page_files = sorted(glob.glob(os.path.join(folder, "page_*.md")))
        
        full_text = []
        for p_file in page_files:
            with open(p_file, "r", encoding="utf-8") as f:
                full_text.append(f.read())
        
        combined_path = os.path.join(self.output_dir, f"{clean_name}_FULL.md")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
        print(f"✨ Compiled full document to: {combined_path}")
        return combined_path
    
    
# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":

    parser = VisionPDFParser(model_provider="openai", output_dir="docs/md")
    parser.parse_pdf("docs/raw/RIBOs-Consolidated-Examinee-Resource-September-18-2024.pdf", pages_to_process=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])