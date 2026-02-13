import os
import json
import glob
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = Path("docs/md/RIBOs-Consolidated-Examinee-Resource-September-18-2024")
OUTPUT_FILE = Path("docs/ready/chunks_for_qdrant.jsonl")
STATE_FILE = Path("docs/ready/chunking_progress.state")
CONTEXT_OVERLAP = 500  # Number of chars to peek backward and forward

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@dataclass
class ProposedChunk:
    start_anchor: str
    end_anchor: str
    topic: str

class SemanticChunker:
    def __init__(self):
        setup_files()
        self.processed_files = self._load_state()
    
    def _load_state(self) -> set:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return set(json.load(f))
        return set()

    def _save_state(self, filename: str):
        self.processed_files.add(filename)
        with open(STATE_FILE, "w") as f:
            json.dump(list(self.processed_files), f)

    def _save_chunk(self, chunk_data: dict):
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(chunk_data) + "\n")

    def _read_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_llm_proposals(self, current_text: str, prev_context: str, next_context: str) -> List[ProposedChunk]:
        """
        Asks LLM to chunk 'current_text', using prev/next context for boundary decisions.
        """
        prompt = f"""
        You are a Semantic Chunking Engine. Split the TARGET TEXT into coherent chunks for RAG.
        
        ### PREVIOUS CONTEXT (End of last page):
        "...{prev_context}"
        
        ### TARGET TEXT (Current Page):
        "{current_text}"
        
        ### NEXT CONTEXT (Start of next page):
        "{next_context}..."
        
        ### INSTRUCTIONS:
        1. Ignore the Context fields for extraction. ONLY extract chunks from TARGET TEXT.
        2. Use the Context to decide if a sentence at the start/end of TARGET TEXT is cut off.
           - If a sentence starts in PREVIOUS and ends in TARGET, include the TARGET part in the first chunk.
           - If a sentence starts in TARGET and ends in NEXT, include the TARGET part in the last chunk.
        3. Return the **First 5-8 words** (Start Anchor) and **Last 5-8 words** (End Anchor) of each chunk found in TARGET TEXT.
        
        ### OUTPUT FORMAT (JSON):
        {{
            "chunks": [
                {{ "start_anchor": "First 5 words...", "end_anchor": "Last 5 words...", "topic": "Brief Label" }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You are a JSON-only API."}, 
                      {"role": "user", "content": prompt}]
        )
        
        data = json.loads(response.choices[0].message.content)
        return [ProposedChunk(**item) for item in data.get("chunks", [])]

    def _extract_text_by_anchors(self, full_text: str, proposal: ProposedChunk) -> Optional[Dict]:
        # Same Python extraction logic as before (using .find())
        start_idx = full_text.find(proposal.start_anchor)
        if start_idx == -1:
             start_idx = full_text.find(proposal.start_anchor[:len(proposal.start_anchor)//2])

        if start_idx != -1:
             # Search for end anchor AFTER start index
             end_idx = full_text.find(proposal.end_anchor, start_idx)
             
             if end_idx != -1:
                 real_end = end_idx + len(proposal.end_anchor)
                 return {
                     "start_char_idx": start_idx,
                     "end_char_idx": real_end,
                     "text": full_text[start_idx:real_end],
                     "topic": proposal.topic
                 }
        return None

    def process_folder(self):
        # 1. Get all markdown files sorted
        files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "page_*.md")))
        total_files = len(files)
        
        print(f"Found {total_files} files. Starting sliding window processing...")
        
        for i in range(total_files):
            curr_path = files[i]
            filename = os.path.basename(curr_path)

            # RESUME CHECK
            if filename in self.processed_files:
                continue

            print(f"Processing {filename}...")
            
            # 2. LOAD 3-PAGE WINDOW
            # Current Page
            curr_text = self._read_file(curr_path)
            if not curr_text: 
                self._save_state(filename)
                continue

            # Previous Context (Last N chars of i-1)
            prev_text = ""
            if i > 0:
                full_prev = self._read_file(files[i-1])
                prev_text = full_prev[-CONTEXT_OVERLAP:] # Take last 500 chars

            # Next Context (First N chars of i+1)
            next_text = ""
            if i < total_files - 1:
                full_next = self._read_file(files[i+1])
                next_text = full_next[:CONTEXT_OVERLAP] # Take first 500 chars

            # 3. GET PROPOSALS
            try:
                proposals = self._get_llm_proposals(curr_text, prev_text, next_text)
                
                valid_count = 0
                for prop in proposals:
                    chunk_data = self._extract_text_by_anchors(curr_text, prop)
                    if chunk_data:
                        chunk_data['source_file'] = filename
                        chunk_data['page_num'] = i + 1
                        self._save_chunk(chunk_data)
                        valid_count += 1
                
                print(f"  -> Saved {valid_count} chunks.")
                self._save_state(filename)
                
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")


def setup_files():
    # 1. Create the Directory Structure (parents=True creates all missing folders)
    # We only need to do this for one file's parent since they are in the same folder
    if not OUTPUT_FILE.parent.exists():
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created directory: {OUTPUT_FILE.parent}")

    # 2. Create OUTPUT_FILE (Empty) if it doesn't exist
    if not OUTPUT_FILE.exists():
        OUTPUT_FILE.touch()
        print(f"📄 Created: {OUTPUT_FILE}")
    else:
        print(f"✅ Found: {OUTPUT_FILE}")

    # 3. Create STATE_FILE (Initialize with empty JSON list []) if it doesn't exist
    # If we just 'touch' it, json.load() will fail on an empty file.
    if not STATE_FILE.exists():
        with open(STATE_FILE, "w") as f:
            json.dump([], f)
        print(f"📄 Created: {STATE_FILE} (Initialized with [])")
    else:
        print(f"✅ Found: {STATE_FILE}")

if __name__ == "__main__":
    chunker = SemanticChunker()
    chunker.process_folder()