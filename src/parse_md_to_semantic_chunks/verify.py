import streamlit as st
import json
import os
import glob
from dataclasses import dataclass
from typing import List, Dict

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Chunk Coverage Inspector")

BASE_MD_DIR = "docs/md/RIBOs-Consolidated-Examinee-Resource-September-18-2024"  # Where your MD files are
CHUNKS_FILE = "docs/ready/chunks_for_qdrant.jsonl"     # Your generated chunks

# ==========================================
# LOGIC
# ==========================================

@dataclass
class ChunkInfo:
    start: int
    end: int
    text: str
    topic: str
    source_file: str

def load_chunks(jsonl_path: str) -> Dict[str, List[ChunkInfo]]:
    """
    Loads chunks and organizes them by filename.
    Returns: { "page_0001.md": [Chunk1, Chunk2...], ... }
    """
    organized = {}
    
    if not os.path.exists(jsonl_path):
        return {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Handle filename mapping (e.g. if you mapped page_005.md -> Ribos.pdf)
                # For this viewer, we need the ORIGINAL MD filename to load the text.
                # If you saved 'original_filename' in previous step, use that.
                # Otherwise, we might need a fallback.
                fname = data.get('original_filename') or data.get('source_file')
                
                if fname not in organized:
                    organized[fname] = []
                
                organized[fname].append(ChunkInfo(
                    start=data.get('start_char_idx', 0),
                    end=data.get('end_char_idx', 0),
                    text=data.get('text', ''),
                    topic=data.get('topic', 'Unknown'),
                    source_file=fname
                ))
            except Exception:
                continue
    return organized

def generate_coverage_html(original_text: str, chunks: List[ChunkInfo]) -> str:
    """
    Highlights the original text:
    - Green background: Covered by at least one chunk.
    - Red text / No background: Not covered (Missing data).
    """
    if not original_text:
        return "No text found."

    # 1. Create a boolean mask for every character
    # False = Not Covered, True = Covered
    mask = [False] * len(original_text)

    for chunk in chunks:
        # Clamp indices to prevent errors
        start = max(0, chunk.start)
        end = min(len(original_text), chunk.end)
        
        # Mark characters as covered
        for i in range(start, end):
            mask[i] = True

    # 2. Reconstruct text with HTML tags
    html_out = []
    is_covering = False
    
    # We'll wrap continuous regions in <span> tags to keep HTML size small
    current_span = []
    
    for char, covered in zip(original_text, mask):
        if covered != is_covering:
            # State change! Close previous span and start new one
            text_segment = "".join(current_span)
            
            if is_covering:
                # WAS covering -> closing green
                html_out.append(f"<span style='background-color: #d4f7d4; color: #006600;' title='Covered'>{text_segment}</span>")
            else:
                # WAS NOT covering -> closing red
                # We render newlines as <br> only in the red/uncovered zones so structure is visible
                text_segment = text_segment.replace("\n", "⏎<br>") 
                html_out.append(f"<span style='color: #cc0000; opacity: 0.6;'>{text_segment}</span>")
            
            current_span = [char]
            is_covering = covered
        else:
            current_span.append(char)
            
    # Flush last segment
    text_segment = "".join(current_span)
    if is_covering:
        html_out.append(f"<span style='background-color: #d4f7d4; color: #006600;'>{text_segment}</span>")
    else:
        text_segment = text_segment.replace("\n", "⏎<br>")
        html_out.append(f"<span style='color: #cc0000; opacity: 0.6;'>{text_segment}</span>")

    return "".join(html_out)

# ==========================================
# MAIN UI
# ==========================================

st.title("🧩 Semantic Chunk Coverage Visualizer")

# 1. Load Data
chunks_map = load_chunks(CHUNKS_FILE)
all_md_files = sorted(glob.glob(os.path.join(BASE_MD_DIR, "*.md")))

if not all_md_files:
    st.error(f"No MD files found in {BASE_MD_DIR}")
    st.stop()

# 2. Sidebar Navigation
selected_file_path = st.sidebar.selectbox(
    "Select Document Page", 
    all_md_files, 
    format_func=lambda x: os.path.basename(x)
)

filename = os.path.basename(selected_file_path)
file_chunks = chunks_map.get(filename, [])

# 3. Read Original Content
with open(selected_file_path, "r", encoding="utf-8") as f:
    original_text = f.read()

# 4. Metrics
st.sidebar.markdown("---")
st.sidebar.metric("Total Chunks", len(file_chunks))
coverage_pct = 0
if len(original_text) > 0:
    # Calculate crude coverage %
    covered_chars = set()
    for c in file_chunks:
        for i in range(c.start, c.end):
            covered_chars.add(i)
    coverage_pct = (len(covered_chars) / len(original_text)) * 100

st.sidebar.metric("Text Coverage", f"{coverage_pct:.1f}%")

if coverage_pct < 80:
    st.sidebar.warning("⚠️ Low coverage! You might be losing info.")

# 5. RENDER MAIN VIEW
tab1, tab2 = st.tabs(["👁️ Coverage Heatmap", "📄 Chunk List"])

with tab1:
    st.markdown("### How to read this:")
    st.markdown("""
    * <span style='background-color: #d4f7d4; color: #006600; padding:2px;'>Green Background</span>: This text is **INCLUDED** in your vector database.
    * <span style='color: #cc0000; opacity: 0.8;'>Red Faded Text</span>: This text was **SKIPPED** (Not chunked).
    """, unsafe_allow_html=True)
    
    html_view = generate_coverage_html(original_text, file_chunks)
    
    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            padding: 20px;
            font-family: monospace;
            line-height: 1.5;
            background-color: white;
            white-space: pre-wrap; 
            height: 600px;
            overflow-y: auto;
        ">
            {html_view}
        </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    st.write("These are the actual objects sent to Qdrant:")
    for i, c in enumerate(file_chunks):
        with st.expander(f"Chunk {i+1} | {c.topic}"):
            st.text(c.text)
            st.caption(f"Indices: {c.start} - {c.end}")