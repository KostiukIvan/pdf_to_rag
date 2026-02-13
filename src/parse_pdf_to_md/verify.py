import streamlit as st
import os
import fitz  # PyMuPDF
from PIL import Image
import difflib
import re
from rapidfuzz import fuzz  # Requires: pip install rapidfuzz

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="RAG Pipeline Verifier")

# ==========================================
# LOGIC
# ==========================================

def get_pdf_page_content(pdf_path, page_num):
    """
    Returns Image (visual), Raw Text (content), and Total Pages.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num - 1 < len(doc):
            page = doc.load_page(page_num - 1)
            
            # 1. Get Image
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 2. Get Text
            raw_text = page.get_text("text")
            return img, raw_text, len(doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return None, None, 0

def load_markdown_file(md_folder, page_num):
    """Loads a specific markdown page file."""
    filename = f"page_{page_num:04d}.md"
    path = os.path.join(md_folder, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), filename
    return None, filename

def find_best_matching_markdown(md_folder, pdf_text, target_page_num, search_window=3):
    """
    Scans MD pages around the target_page_num to find the text that best matches the PDF.
    Returns: (best_md_content, best_md_filename, best_page_num, match_score)
    """
    best_score = 0
    best_content = None
    best_filename = None
    best_page = target_page_num
    
    # Normalize PDF text for cleaner matching
    clean_pdf = " ".join(pdf_text.split()[:500]) # optimize speed, check first 500 words
    
    # Search range: e.g., if Page 10, window 2 -> check 8, 9, 10, 11, 12
    start_page = max(1, target_page_num - search_window)
    end_page = target_page_num + search_window
    
    candidates = []

    for p_num in range(start_page, end_page + 1):
        content, fname = load_markdown_file(md_folder, p_num)
        if content:
            # Normalize MD
            clean_md = " ".join(content.split()[:500])
            
            # Fuzzy Ratio
            score = fuzz.partial_ratio(clean_pdf, clean_md)
            candidates.append((score, p_num, fname, content))
            
            if score > best_score:
                best_score = score
                best_page = p_num
                best_filename = fname
                best_content = content

    return best_content, best_filename, best_page, best_score

def clean_text_for_diff(text, is_markdown=False):
    """
    Aggressively cleans text to focus on CONTENT, ignoring formatting and fillers.
    """
    if not text: return []
    
    # 1. Remove Markdown Table Separator lines (e.g. |---| or |:---:|)
    text = re.sub(r'^\s*\|?[\s\-:|]+\|?\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Remove Markdown Fences (```markdown) if they slipped through
    text = re.sub(r"```[a-zA-Z]*", "", text)

    if is_markdown:
        # Remove Markdown syntax chars (*, #, `, ~)
        text = re.sub(r'[*#`~]', '', text)
        text = text.replace('|', ' ')
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # Links
        text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)    # Images

    # --- NEW: COLLAPSE FILLER DOTS/UNDERSCORES ---
    # Replaces 2 or more dots/underscores with a single space
    # Example: "Policy .................... 5" becomes "Policy 5"
    text = re.sub(r'[\._]{2,}', ' ', text)

    # General Cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.split()

def generate_side_by_side_diff(pdf_text, md_text):
    """
    Compares cleaned PDF text vs Cleaned Markdown text.
    """
    # Clean both inputs specifically for the Diff View
    a = clean_text_for_diff(pdf_text, is_markdown=False)
    b = clean_text_for_diff(md_text, is_markdown=True)
    
    matcher = difflib.SequenceMatcher(None, a, b)
    
    html_out = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        pdf_segment = " ".join(a[i1:i2])
        md_segment = " ".join(b[j1:j2])
        
        if tag == 'equal':
            # Perfect Match -> Standard Grey/Black
            html_out.append(f"<span style='color: #444;'>{pdf_segment} </span>")
            
        elif tag == 'delete':
            # In PDF, missing in MD -> RED (The Parser missed this!)
            html_out.append(
                f"<span style='background-color: #ffcccc; color: #cc0000; text-decoration: line-through; padding: 0 2px;' "
                f"title='Missing in MD'> {pdf_segment} </span>"
            )
            
        elif tag == 'insert':
            # In MD, missing in PDF -> GREEN (Hallucination or Image Text)
            html_out.append(
                f"<span style='background-color: #ccffcc; color: #006600; font-weight: bold; padding: 0 2px;' "
                f"title='Added by AI'> {md_segment} </span>"
            )
            
        elif tag == 'replace':
            # Conflict -> Show deviation
            html_out.append(
                f"<span style='background-color: #fff5cc; color: #b35900; padding: 0 2px;'>"
                f"[{pdf_segment} &rarr; {md_segment}]</span>"
            )
            
    return "".join(html_out)
# ==========================================
# UI
# ==========================================

st.title("🕵️ Smart PDF Verifier (Auto-Align)")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    pdf_path = st.sidebar.text_input("PDF Path", "docs/raw/RIBOs-Consolidated-Examinee-Resource-September-18-2024.pdf")
    md_folder = st.sidebar.text_input("Markdown Output Folder", "docs/md/RIBOs-Consolidated-Examinee-Resource-September-18-2024")

    
    st.divider()
    
    st.subheader("Sync Settings")
    use_auto_align = st.checkbox("Enable Auto-Align", value=True)
    search_window = st.slider("Auto-Search Window", 1, 5, 2, help="How many pages +/- to search for matching text.")
    manual_offset = st.number_input("Manual Offset", min_value=-10, max_value=10, value=0, help="Manually shift MD page number (e.g. -1 means PDF Pg 5 shows MD Pg 4)")

    if st.button("Load Document"):
        st.session_state['loaded'] = True
        st.session_state['page'] = 1

if st.session_state.get('loaded'):
    # 1. Load PDF Page
    current_page = st.session_state.get('page', 1)
    img, raw_pdf_text, total_pages = get_pdf_page_content(pdf_path, current_page)
    
    # 2. Navigation
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        new_page = st.slider("PDF Page Navigator", 1, total_pages, current_page)
        if new_page != current_page:
            st.session_state['page'] = new_page
            st.rerun()

    # 3. Determine which Markdown File to Show
    if use_auto_align and raw_pdf_text:
        # AUTOMATIC MODE
        md_content, md_filename, md_page_num, score = find_best_matching_markdown(
            md_folder, raw_pdf_text, new_page, search_window=search_window
        )
        status_msg = f"**Auto-Aligned:** PDF Page {new_page} matched with **{md_filename}** (Confidence: {score:.1f}%)"
        status_color = "green" if score > 70 else "orange"
    else:
        # MANUAL MODE
        target_md_page = new_page + manual_offset
        md_content, md_filename = load_markdown_file(md_folder, target_md_page)
        md_page_num = target_md_page
        status_msg = f"**Manual Mode:** Displaying **{md_filename}** (Offset: {manual_offset})"
        status_color = "blue"

    # Display Status Bar
    st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px; border-left: 5px solid {status_color};'>{status_msg}</div>", unsafe_allow_html=True)
    st.divider()

    # 4. TABS
    tab1, tab2 = st.tabs(["👁️ Visual Layout", "📝 Text Diff"])

    with tab1:
        col_L, col_R = st.columns(2)
        with col_L:
            st.caption(f"PDF Page {new_page}")
            if img: st.image(img, use_container_width=True)
        with col_R:
            st.caption(f"Markdown Content ({md_filename})")
            if md_content:
                st.text_area("Markdown", md_content, height=800)
            else:
                st.warning("No Markdown found.")

    with tab2:
        if raw_pdf_text and md_content:
            diff_html = generate_side_by_side_diff(raw_pdf_text, md_content)
            st.markdown(f"<div style='border:1px solid #ddd; padding:20px; font-family:monospace; background:#fff;'>{diff_html}</div>", unsafe_allow_html=True)
        else:
            st.info("Insufficient text for diff.")