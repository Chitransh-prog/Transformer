"""
Generates the comprehensive, publication-grade Transformer Project Guide PDF.
Includes: Embedded Live Terminal Proof Images, Training Curves,
Engineering Evolution, Complete Machine Learning Pipeline,
Transformer Taxonomies & Use Cases, Perplexity Deep-Dive, and Viva Defense Q&A.

Run:  python make_pdf.py
Output: Transformer_Project_Guide.pdf
"""

import os
import math
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem, KeepTogether, Image
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

W, H = A4
MARGIN = 1.6 * cm

# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#1E1B4B")
INDIGO   = colors.HexColor("#3730A3")
VIOLET   = colors.HexColor("#6366F1")
SLATE    = colors.HexColor("#475569")
LIGHT_BG = colors.HexColor("#F8FAFC")
BORDER_C = colors.HexColor("#E2E8F0")
CODE_BG  = colors.HexColor("#0F172A")
CODE_FG  = colors.HexColor("#F1F5F9")
AMBER_BG = colors.HexColor("#FEF3C7")
AMBER_TX = colors.HexColor("#92400E")
GREEN_BG = colors.HexColor("#DCFCE7")
GREEN_TX = colors.HexColor("#166534")
BLUE_BG  = colors.HexColor("#EFF6FF")
BLUE_TX  = colors.HexColor("#1E40AF")
WHITE    = colors.white


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        if self._pageNumber == 1:
            return  # Suppress running header/footer on cover page

        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(SLATE)

        # Running Header
        self.drawString(MARGIN, H - 1.1 * cm, "CollegeFormer: Transformer Architecture Implementation & Technical Manual")
        self.drawRightString(W - MARGIN, H - 1.1 * cm, "Chitransh & Parth Kapoor")
        self.setStrokeColor(BORDER_C)
        self.setLineWidth(0.4)
        self.line(MARGIN, H - 1.25 * cm, W - MARGIN, H - 1.25 * cm)

        # Running Footer
        self.line(MARGIN, 1.25 * cm, W - MARGIN, 1.25 * cm)
        self.drawString(MARGIN, 0.9 * cm, "Attention Is All You Need (Vaswani et al., 2017) — College Scale Engineering")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(W - MARGIN, 0.9 * cm, page_str)
        self.restoreState()


def build_styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        base.add(ParagraphStyle(name=name, **kw))

    add("H1",
        fontSize=13, leading=17, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    add("H2",
        fontSize=10.5, leading=14.5, textColor=INDIGO,
        fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=2.5)
    add("H3",
        fontSize=9, leading=12.5, textColor=SLATE,
        fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2)

    add("Body",
        fontSize=8.2, leading=12.2, textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=3)
    add("Caption",
        fontSize=7.8, leading=10.5, textColor=SLATE,
        fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceBefore=2, spaceAfter=5)
    add("MyCode",
        fontSize=7, leading=10, textColor=CODE_FG,
        fontName="Courier", backColor=CODE_BG,
        leftIndent=6, rightIndent=6, spaceAfter=3.5, spaceBefore=2,
        borderPadding=(3, 5, 3, 5))
    add("MyNote",
        fontSize=7.8, leading=11, textColor=AMBER_TX,
        fontName="Helvetica-Oblique", backColor=AMBER_BG,
        leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=3.5,
        borderPadding=(3, 5, 3, 5))
    add("MyTip",
        fontSize=7.8, leading=11, textColor=GREEN_TX,
        fontName="Helvetica-Oblique", backColor=GREEN_BG,
        leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=3.5,
        borderPadding=(3, 5, 3, 5))
    add("Math",
        fontSize=7.8, leading=11, textColor=INDIGO,
        fontName="Courier-Bold", alignment=TA_CENTER,
        spaceBefore=2, spaceAfter=3.5)
    add("TableHead",
        fontSize=7.5, leading=9.5, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER)
    add("TableCell",
        fontSize=7.5, leading=10, textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica", alignment=TA_LEFT)
    add("TableCellC",
        fontSize=7.5, leading=10, textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica", alignment=TA_CENTER)
    return base


def hr(color=INDIGO, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=3.5, spaceBefore=2)


def make_table(data, col_widths, header_bg=INDIGO, stripe_bg=LIGHT_BG, styles=None):
    rows = []
    for i, row in enumerate(data):
        style = "TableHead" if i == 0 else ("TableCell" if len(row[0]) > 3 else "TableCellC")
        rows.append([Paragraph(str(c), styles[style]) for c in row])

    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, stripe_bg]),
        ("GRID",           (0, 0), (-1, -1), 0.4, BORDER_C),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
    ]
    return Table(rows, colWidths=col_widths, style=TableStyle(style_cmds))


def build_pdf(path="Transformer_Project_Guide.pdf"):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="CollegeFormer: Transformer Project Guide",
        author="Chitransh & Parth Kapoor",
    )
    s = build_styles()

    def P(txt, style="Body"):
        return Paragraph(txt, s[style] if isinstance(style, str) else style)

    def SP(h=0.15):
        return Spacer(1, h * cm)

    story = []

    # ═══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════════════════════
    cover_data = [
        [Paragraph("COLLEGEFORMER: SEQUENCE-TO-SEQUENCE TRANSFORMER", ParagraphStyle(
            "C1", fontSize=20, leading=25, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Spacer(1, 0.12 * cm)],
        [Paragraph("From-Scratch PyTorch Implementation of &ldquo;Attention Is All You Need&rdquo; (Vaswani et al., 2017)", ParagraphStyle(
            "C2", fontSize=9, leading=13, textColor=colors.HexColor("#CBD5E1"),
            fontName="Helvetica-Oblique", alignment=TA_CENTER))],
        [Paragraph("Engineering Evolution, Pipeline, Empirical Proof &amp; Viva Voce Defense", ParagraphStyle(
            "C3", fontSize=9, leading=13, textColor=colors.HexColor("#A5B4FC"),
            fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Spacer(1, 0.25 * cm)],
        [Paragraph("Submitted By:", ParagraphStyle(
            "C4", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#94A3B8"),
            fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("<b>Chitransh</b> [03529802024] &nbsp;&bull;&nbsp; <b>Parth Kapoor</b> [01129802024]", ParagraphStyle(
            "C5", fontSize=9, leading=12.5, textColor=WHITE,
            fontName="Helvetica", alignment=TA_CENTER))],
        [Paragraph("B.Tech &mdash; Computer Science / Artificial Intelligence &amp; Machine Learning", ParagraphStyle(
            "C6", fontSize=7.5, leading=10, textColor=colors.HexColor("#94A3B8"),
            fontName="Helvetica", alignment=TA_CENTER))],
    ]
    cover_table = Table(cover_data, colWidths=[W - 2 * MARGIN])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(cover_table)
    story.append(SP(0.2))

    # Metric summary cards
    cards = [
        ["[ Model Identity ]", "<b>CollegeFormer</b><br/>(TransEnFr-Seq2Seq)"],
        ["[ Real Corpus ]", "<b>Tatoeba (15k-167k pairs)</b><br/>2,665,035 Trainable Params"],
        ["[ Live Verification ]", "Best Val Loss: <b>2.9460</b><br/>Val PPL: <b>19.03</b> (Epoch 15)"],
        ["[ Code Architecture ]", "<b>3 Streamlined Files</b><br/>transformer &bull; train &bull; run"],
    ]
    card_cols = []
    for title, desc in cards:
        cell = [
            Paragraph(title, ParagraphStyle("CH", fontSize=7.5, fontName="Helvetica-Bold", textColor=INDIGO, alignment=TA_CENTER, leading=9.5)),
            Spacer(1, 1.5),
            Paragraph(desc, ParagraphStyle("CD", fontSize=6.8, fontName="Helvetica", textColor=SLATE, alignment=TA_CENTER, leading=9))
        ]
        t = Table([cell], colWidths=[4.2 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("BOX",        (0, 0), (-1, -1), 0.7, colors.HexColor("#C7D2FE")),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ]))
        card_cols.append(t)

    card_table = Table([card_cols], colWidths=[4.45 * cm] * 4)
    card_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(card_table)
    story.append(SP(0.25))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: MODEL IDENTITY & REAL-WORLD USE CASES
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("1. Model Identity, Transformer Taxonomy &amp; Real-World Use Cases", "H1"))
    story.append(hr())
    story.append(P("""<b>Model Identity:</b> This system is formally designated as <b>CollegeFormer (TransEnFr-Seq2Seq)</b>.
                   It is a full Encoder&ndash;Decoder sequence transduction architecture trained to translate English into French
                   while demonstrating intra-sentence self-attention, cross-attention alignment, and autoregressive generation."""))
    story.append(SP(0.06))

    story.append(P("The Three Major Transformer Architectural Families:", "H2"))
    tax_data = [
        ["Family Type", "Key Paradigm", "Prominent Models", "Primary Industry Use Cases"],
        ["Encoder-Only", "Auto-encoding / Bidirectional Attention", "BERT, RoBERTa, DeBERTa, ELECTRA", "Text Classification, Sentiment Analysis, Named Entity Recognition (NER), Semantic Search &amp; Vector Embeddings."],
        ["Decoder-Only", "Autoregressive / Causal Left-to-Right", "GPT-3/4, LLaMA-3, Gemini, Mistral", "Open-ended Text Generation, Code Synthesis, Interactive Chatbots, Creative Writing &amp; Reasoning."],
        ["Encoder-Decoder (Our Model)", "Full Sequence Transduction", "CollegeFormer, Original Transformer, T5, BART", "Machine Translation (En&rarr;Fr), Abstractive Document Summarization, Text-to-SQL, Complex Grammar Correction."],
    ]
    story.append(make_table(tax_data, [2.8 * cm, 3.6 * cm, 3.6 * cm, 7.8 * cm], styles=s))
    story.append(SP(0.1))

    story.append(P("Cross-Domain Transformer Applications Beyond NLP:", "H2"))
    story.append(P("""The self-attention mechanism implemented in <code>CollegeFormer</code> is identical to the core operator powering
                   state-of-the-art architectures in other scientific domains:
                   <br/>&bull; <b>Computer Vision (Vision Transformer &mdash; ViT):</b> Treats 16&times;16 image patches as tokens; self-attention captures global scene context faster than CNN receptive fields.
                   <br/>&bull; <b>Bioinformatics (AlphaFold):</b> Uses evoformer attention over amino acid sequences to solve 50-year-old protein 3D structure folding with atomic precision.
                   <br/>&bull; <b>Speech Processing (OpenAI Whisper):</b> Employs an encoder-decoder Transformer on audio spectrogram frames for multilingual speech-to-text transcription."""))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: EARLY THINKING, PERFORMANCE ANALYSIS & ARCHITECTURE EVOLUTION
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("2. Early Engineering Thinking &amp; Architecture Evolution", "H1"))
    story.append(hr())
    story.append(P("""Developing a research-grade Transformer at college scale required navigating a critical trade-off:
                   <b>mathematical completeness versus execution reliability and code readability</b>.
                   The project underwent three distinct engineering iterations:"""))
    story.append(SP(0.08))

    story.append(P("Iteration 1: The Initial 7-File Architecture &amp; Observed Bottlenecks", "H2"))
    story.append(P("""Initially, the codebase was decomposed into seven isolated scripts (<code>model.py</code>, <code>dataset.py</code>,
                   <code>lr_scheduler.py</code>, <code>train.py</code>, <code>inference.py</code>, <code>evaluate.py</code>, <code>demo.py</code>).
                   While theoretically modular, this initial approach introduced severe practical drawbacks:
                   <br/>&bull; <b>Over-Engineering &amp; Redundancy:</b> <code>ScaledDotProductAttention</code> was implemented as an <code>nn.Module</code>
                   class with zero learnable parameters, creating unnecessary layer wrapping.
                   <br/>&bull; <b>Sublayer Abstraction Clutter:</b> A dedicated <code>SublayerConnection</code> class obscured the simplicity of residual addition (<code>norm(x + drop(sublayer(x)))</code>).
                   <br/>&bull; <b>Circular Coupling:</b> <code>evaluate.py</code> imported functions from <code>inference.py</code>, while <code>demo.py</code>
                   duplicated 60% of <code>train.py</code>'s training loop logic to run a fast demo.
                   <br/>&bull; <b>Greedy Search Drift:</b> The initial inference loop had no repetition guard, causing the decoder to loop indefinitely on unknown Out-of-Vocabulary (OOV) tokens."""))
    story.append(SP(0.08))

    story.append(P("Iteration 2: Streamlining to the Clean 3-File Architecture", "H2"))
    story.append(P("""To make the code immediately explainable in a viva and bulletproof to execute, the architecture was refactored into
                   <b>three high-cohesion files</b> (<code>transformer.py</code>, <code>train.py</code>, <code>run.py</code>):
                   <br/>&bull; <code>scaled_dot_product_attention</code> was converted into a pure, vectorized function.
                   <br/>&bull; <code>FeedForward</code> was simplified to a clean <code>nn.Sequential</code> container.
                   <br/>&bull; An adaptive <b>N-gram Loop Guard</b> was introduced in <code>run.py</code> to detect and terminate alternating cyclic token drift during inference.
                   <br/>&bull; <code>run_epoch(train=True/False)</code> merged training and validation passes into a single, unified loop."""))
    story.append(SP(0.08))

    story.append(P("Iteration 3: The Dual-Corpus Strategy (Demo Mode vs. Tatoeba Benchmark)", "H2"))
    story.append(P("""Recognizing that university examination panels demand both an instantaneous viva demo and an authentic academic benchmark,
                   we engineered a dual-mode data pipeline:
                   <br/>&bull; <b>Fast Demo Mode:</b> 838 curated parallel pairs, training in ~65 seconds on CPU with zero internet downloads.
                   <br/>&bull; <b>Tatoeba Benchmark Mode:</b> Auto-downloads the official 167k-pair Tatoeba corpus, filtering 15,000&ndash;30,000 real human sentences for a serious 30&ndash;45 minute CPU training run with genuine generalization."""))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: EMPIRICAL TRAINING EVIDENCE & ACTUAL SCREENSHOTS
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("3. Empirical Verification: Live Training Run on Tatoeba Benchmark", "H1"))
    story.append(hr())
    story.append(P("""To validate <code>CollegeFormer</code> at authentic college scale, the model was trained on our local system
                   using the standard <b>Tatoeba English-French Corpus (15,000 sentence pairs)</b>.
                   Below is the live execution evidence, loss convergence curve, and terminal output:"""))
    story.append(SP(0.08))

    # Embed Image 1: Terminal Training Run
    if os.path.exists("img_tatoeba_training.png"):
        img1 = Image("img_tatoeba_training.png", width=14.0 * cm, height=9.1 * cm)
        story.append(KeepTogether([
            img1,
            P("Figure 1: Live execution log of <code>CollegeFormer</code> training for 15 epochs on 15,000 Tatoeba pairs (Parameters: 2,665,035).", "Caption")
        ]))
    story.append(SP(0.08))

    # Embed Image 2: Training Loss Curve
    if os.path.exists("img_tatoeba_loss_curve.png"):
        img2 = Image("img_tatoeba_loss_curve.png", width=13.5 * cm, height=7.7 * cm)
        story.append(KeepTogether([
            img2,
            P("Figure 2: Empirical Training and Validation Loss Curve (Label-smoothed Cross Entropy) over 15 epochs on Tatoeba.", "Caption")
        ]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: INFERENCE EVIDENCE & BLEU EVALUATION SCREENSHOT
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("4. Empirical Evaluation: SacreBLEU Benchmark &amp; Live Inference", "H1"))
    story.append(hr())
    story.append(P("""Following the completion of 15 epochs of training on the Tatoeba benchmark, <code>run.py --bleu</code> was executed
                   on the local machine to evaluate corpus BLEU score and test interactive generation:"""))
    story.append(SP(0.08))

    # Embed Image 3: Terminal BLEU & Inference Run
    if os.path.exists("img_tatoeba_bleu.png"):
        img3 = Image("img_tatoeba_bleu.png", width=12.5 * cm, height=11.4 * cm)
        story.append(KeepTogether([
            img3,
            P("Figure 3: Live terminal output showing <code>python run.py --bleu</code> executing SacreBLEU evaluation on CPU.", "Caption")
        ]))
    story.append(SP(0.08))

    story.append(P("Detailed Analysis of Observed Empirical Results:", "H2"))
    story.append(P("""<b>1. Parameters and Vocabulary Expansion:</b> Scaling to the Tatoeba benchmark expanded the vocabulary from 120 words
                   (demo mode) to <b>2,897 English words</b> and <b>5,323 French words</b>, bringing total trainable parameters to <b>2,665,035</b>.
                   <br/><b>2. Convergence Trajectory:</b> As shown in Figure 1 &amp; 2, Training Loss dropped steadily from <b>4.5958 to 2.5941</b>,
                   while Validation Loss dropped from <b>3.7046 to 2.9460</b>, proving that the Adam optimizer with warmup scheduling effectively prevented divergence.
                   <br/><b>3. Perplexity Learning:</b> Perplexity dropped from <b>40.63 down to 19.03</b>, demonstrating that the model halved its uncertainty
                   across real open-domain vocabulary."""))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: COMPLETE MACHINE LEARNING PIPELINE & FILE STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("5. Complete Machine Learning Pipeline &amp; File Structure", "H1"))
    story.append(hr())
    story.append(P("""The end-to-end operational pipeline followed by <code>CollegeFormer</code> from raw text input
                   to final translation inference comprises seven disciplined engineering steps:"""))
    story.append(SP(0.06))

    pipe_data = [
        ["Step", "Pipeline Stage", "Implementation in Code", "Engineering Purpose"],
        ["1", "Text Tokenization", "train.py &rarr; Vocab._tok()", "Splits words and punctuation via regex; builds frequency-based vocabulary with &lt;pad&gt;=0, &lt;sos&gt;=1, &lt;eos&gt;=2, &lt;unk&gt;=3."],
        ["2", "Dynamic Batch Collation", "train.py &rarr; collate()", "Batches sentences and pads dynamically to the longest sequence in that batch rather than a global fixed length (saving 60% memory)."],
        ["3", "Embedding &amp; Positional Signal", "transformer.py &rarr; _embed()", "Multiplies embeddings by &radic;d_model and injects sinusoidal position waves to establish spatial sequence order."],
        ["4", "Bidirectional Encoding", "transformer.py &rarr; Encoder", "Applies multi-head self-attention with padding masks; tokens contextualize across the entire sentence simultaneously."],
        ["5", "Autoregressive Decoding", "transformer.py &rarr; Decoder", "Applies causal look-ahead masking to prevent future token leakage; connects to encoder outputs via cross-attention."],
        ["6", "Warmup Optimization", "transformer.py &rarr; WarmupScheduler", "Updates Adam optimizer learning rate dynamically per step; prevents early gradient shock on Xavier-initialized weights."],
        ["7", "Greedy Search &amp; Loop Guard", "run.py &rarr; translate()", "Autoregressively generates tokens one-by-one from &lt;sos&gt;; terminates on &lt;eos&gt; or cyclic repeat detection."],
    ]
    story.append(make_table(pipe_data, [0.8 * cm, 3.2 * cm, 4.4 * cm, 9.4 * cm], styles=s))
    story.append(SP(0.12))

    story.append(P("Repository Directory Layout &amp; Artifact Storage", "H2"))
    story.append(P("""Anything-project/
├── transformer.py              # Full model architecture + WarmupScheduler
├── train.py                    # Vocab builder, dual-corpus loader, training loop, loss plotter
├── run.py                      # Interactive shell, BLEU benchmark, attention heatmap visualizer
├── requirements.txt            # Minimal verified dependencies (torch, sacrebleu, matplotlib, etc.)
├── data/                       # Cached Tatoeba benchmark corpus (auto-downloaded)
│   ├── fra-eng.zip             # Downloaded archive (~3.4 MB)
│   └── fra.txt                 # 167,130 sentence pairs
├── checkpoints/
│   ├── best_model.pt           # Saved PyTorch checkpoint weights & hyperparameters
│   ├── src_vocab.json          # Source English vocabulary word-to-index mapping
│   ├── tgt_vocab.json          # Target French vocabulary word-to-index mapping
│   └── loss_curve.png          # High-resolution training vs. validation loss curve
├── README.md                   # Full technical documentation and markdown guide
└── Transformer_Project_Guide.pdf # Official printable technical manual & viva defense guide""", "MyCode"))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: TRAINING & POST-TRAINING EXECUTION GUIDE
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("6. Training &amp; Post-Training Execution Guide", "H1"))
    story.append(hr())

    story.append(P("6.1 Training Commands (Choose Your Mode)", "H2"))
    story.append(P("<b>Mode A: Real Academic Benchmark Training (Tatoeba Dataset, ~30-45 mins on CPU):</b>"))
    story.append(P("python train.py --dataset tatoeba --num_pairs 15000 --epochs 15", "MyCode"))
    story.append(P("<b>Mode B: Rapid Viva Demo Training (Curated Built-in Corpus, ~65 seconds on CPU):</b>"))
    story.append(P("python train.py --dataset demo --epochs 12", "MyCode"))
    story.append(SP(0.06))

    story.append(P("6.2 Complete Post-Training Execution Commands", "H2"))
    story.append(P("""Once training is complete, your trained weights and vocabularies are stored in <code>checkpoints/</code>.
                   Use the exact terminal commands below to demonstrate and evaluate the model:"""))
    story.append(SP(0.06))

    cmd_table = [
        ["Execution Goal", "Exact Terminal Command Line", "Output / Result Produced"],
        ["Live Interactive Shell (Best for Viva)", "python run.py", "Launches live prompt in terminal. Type any English sentence; receive real-time French translation. Type 'quit' to exit."],
        ["Single Sentence Translation", "python run.py --text \"i study computer science .\"", "Translates the sentence and prints the predicted French output directly to console."],
        ["Translate + Attention Map", "python run.py --text \"i love coding in python .\" --plot", "Translates sentence AND writes multi-head cross-attention heatmap plot to 'attention.png'."],
        ["View Generated Heatmap", "start attention.png", "Opens the attention matrix heatmap in the default image viewer."],
        ["Compute SacreBLEU Score", "python run.py --bleu", "Translates test sentences, compares with reference translations, and outputs BLEU score."],
        ["View Loss Curve Plot", "start checkpoints/loss_curve.png", "Opens the training vs. validation loss curve plot in default image viewer."],
    ]
    story.append(make_table(cmd_table, [3.8 * cm, 6.8 * cm, 7.2 * cm], styles=s))
    story.append(SP(0.1))

    story.append(P("6.3 Programmatic Integration in Python Scripts / Notebooks", "H2"))
    story.append(P("""import torch
from run import load, translate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, src_vocab, tgt_vocab = load(device)

sentence = "we understand the neural network ."
french_translation, *_ = translate(model, src_vocab, tgt_vocab, sentence, device)
print("French Translation:", french_translation)""", "MyCode"))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: PERPLEXITY (PPL) & EVALUATION METRICS DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("7. Perplexity (PPL) &amp; Evaluation Metrics Deep Dive", "H1"))
    story.append(hr())

    story.append(P("7.1 Mathematical Definition of Perplexity", "H2"))
    story.append(P("""Perplexity (PPL) is the standard intrinsic evaluation metric for autoregressive language models and Transformers.
                   Mathematically, it is defined as the <b>exponential of the Cross-Entropy Loss</b> (&mathcal;L):"""))
    story.append(P("Perplexity = exp( &mathcal;L ) = e<sup>&mathcal;L</sup>", "Math"))
    story.append(P("""Where Cross-Entropy Loss over N tokens with ground truth y<sub>i</sub> is:
                   &mathcal;L = &minus; (1 / N) &sum; log P( y<sub>i</sub> | y<sub>&lt;i</sub>, X )."""))
    story.append(SP(0.06))

    story.append(P("7.2 Intuitive Meaning: The 'Uncertainty Branching Factor'", "H2"))
    story.append(P("""Perplexity measures the <b>effective number of equally likely words</b> the model is hesitating between at each token position:
                   <br/>&bull; <b>PPL = 100:</b> The model is completely bewildered; at every step, it is guessing blindly among 100 candidate words.
                   <br/>&bull; <b>PPL = 19.03 (Tatoeba Benchmark Result):</b> The model narrowed its uncertainty from 40+ choices down to ~19 words across an unrestricted 5,300+ word open vocabulary!
                   <br/>&bull; <b>PPL = 2.62 (Curated Demo Result):</b> In a constrained domain, the model narrows its uncertainty down to between <b>2 and 3 words</b> per position!
                   <br/>&bull; <b>PPL = 1.0:</b> Theoretical perfection &mdash; the model predicts the ground truth word with 100% certainty."""))
    story.append(SP(0.06))

    story.append(P("7.3 How Perplexity Connects to the Transformer Architecture", "H2"))
    story.append(P("""1. The decoder hidden states [B, T, d_model] are projected through <code>output_proj</code> to vocabulary logits [B, T, V].
                   <br/>2. The softmax operator converts logits into predicted probability distributions: P(w<sub>k</sub>) = exp(z<sub>k</sub>) / &sum; exp(z<sub>j</sub>).
                   <br/>3. Cross-entropy loss penalizes the model when probability assigned to the ground-truth word is low.
                   <br/>4. By taking the exponential of this loss, Perplexity translates abstract logarithmic loss into an intuitive branching factor showing learning progression."""))
    story.append(SP(0.06))

    story.append(P("7.4 Perplexity vs. BLEU Score Comparison", "H2"))
    comp_data = [
        ["Evaluation Dimension", "Perplexity (PPL)", "BLEU Score (Bilingual Evaluation Understudy)"],
        ["What It Measures", "Model uncertainty and probability distribution confidence on tokens.", "Actual n-gram word overlap between generated hypothesis and human reference."],
        ["Evaluation Regime", "Training &amp; Validation (Teacher Forcing mode).", "Inference &amp; Testing (Autoregressive generation mode)."],
        ["Mathematical Range", "[1.0, &infin;). Lower is better (&rarr; 1.0).", "[0, 100]. Higher is better (&rarr; 100)."],
        ["Observed Results", "Tatoeba: 19.03 (Epoch 15). Demo: 2.62.", "Tatoeba: 4.02 (open domain). Demo: 77.07."],
    ]
    story.append(make_table(comp_data, [3.2 * cm, 7.3 * cm, 7.3 * cm], styles=s))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: MATHEMATICAL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("8. Mathematical Architecture (Paper Equations)", "H1"))
    story.append(hr())

    story.append(P("8.1 Scaled Dot-Product Attention (Section 3.2)", "H2"))
    story.append(P("Attention(Q, K, V) = softmax( (Q &bull; K<sup>T</sup>) / &radic;d_k ) &bull; V", "Math"))
    story.append(P("""<b>Scaling Factor Justification:</b> For large key dimensions (e.g., d_k = 64), the dot products
                   grow large in magnitude, driving the softmax function into flat regions where derivatives are virtually zero
                   (vanishing gradients). Dividing by &radic;d_k normalizes variance to 1.0, keeping backpropagation stable."""))
    story.append(SP(0.06))

    story.append(P("8.2 Multi-Head Attention (Section 3.2)", "H2"))
    story.append(P("MultiHead(Q, K, V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>) &bull; W<sup>O</sup>", "Math"))
    story.append(P("head<sub>i</sub> = Attention(Q &bull; W<sub>i</sub><sup>Q</sup>, K &bull; W<sub>i</sub><sup>K</sup>, V &bull; W<sub>i</sub><sup>V</sup>)", "Math"))
    story.append(P("""Instead of performing a single attention function of dimension d_model, Multi-Head Attention projects
                   queries, keys, and values h times with learned projections to d_k dimensions. This allows the model to jointly
                   attend to information from different representation subspaces (e.g., syntax, tense, pronoun references)."""))
    story.append(SP(0.06))

    story.append(P("8.3 Position-Wise Feed-Forward Networks (Section 3.3)", "H2"))
    story.append(P("FFN(x) = max(0, x &bull; W<sub>1</sub> + b<sub>1</sub>) &bull; W<sub>2</sub> + b<sub>2</sub>", "Math"))
    story.append(P("Applied identically to each position: expands feature space (128 &rarr; 512 &rarr; 128) with ReLU non-linearity."))
    story.append(SP(0.06))

    story.append(P("8.4 Sinusoidal Positional Encoding (Section 3.4)", "H2"))
    story.append(P("PE<sub>(pos, 2i)</sub> = sin( pos / 10000<sup>2i / d_model</sup> ), &nbsp;&nbsp; PE<sub>(pos, 2i+1)</sub> = cos( pos / 10000<sup>2i / d_model</sup> )", "Math"))
    story.append(P("""Attention has no inherent recurrence; positional encodings provide word order. By trigonometric identity
                   (sin(&alpha;+&beta;) = sin&alpha;cos&beta; + cos&alpha;sin&beta;), PE<sub>pos+k</sub> can be expressed as a linear
                   function of PE<sub>pos</sub>, allowing the network to easily learn relative position offsets."""))
    story.append(SP(0.06))

    story.append(P("8.5 Masking Mechanisms (Section 3.1 &amp; 3.2)", "H2"))
    mask_data = [
        ["Mask Type", "Shape", "Target Layer", "Mathematical Behavior"],
        ["Padding Mask", "[B, 1, 1, T]", "Encoder &amp; Decoder", "Identifies &lt;pad&gt; tokens and sets raw attention scores to -1e9, yielding 0 softmax weight."],
        ["Causal Mask", "[1, 1, T, T]", "Decoder Self-Attn", "Lower-triangular matrix that zeroes out future tokens (t+1, t+2), enforcing autoregressive property during training."],
        ["Combined Mask", "[B, 1, T, T]", "Decoder Target", "Logical AND of Padding Mask and Causal Mask, ensuring neither future nor pad tokens receive attention."],
    ]
    story.append(make_table(mask_data, [2.8 * cm, 2.2 * cm, 3.2 * cm, 9.6 * cm], styles=s))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9: VIVA DEFENSE QUESTIONS & ANSWERS
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(P("9. Viva Voce &amp; Examiner Defense Q&amp;A", "H1"))
    story.append(hr())
    story.append(P("Top technical answers for defense during project evaluation:"))
    story.append(SP(0.06))

    viva_qa = [
        ("Q1: Why divide by &radic;d_k in Scaled Dot-Product Attention?",
         "As dimension d_k increases, the variance of the dot product q&bull;k = &sum; q_i k_i scales proportionally with d_k. Large magnitude dot products push the softmax function into saturated regions where gradients are near-zero. Dividing by &radic;d_k normalizes variance to 1.0, ensuring numerically stable backpropagation."),

        ("Q2: Why does the Transformer use Layer Normalization instead of Batch Normalization?",
         "Batch Normalization normalizes across batch examples for each feature. In sequence modeling, sentences have variable lengths and padding tokens, which corrupt batch statistics. Furthermore, batch size 1 during inference breaks Batch Normalization. Layer Normalization normalizes across all hidden features for each token independently, making it invariant to batch size and padding."),

        ("Q3: What is Teacher Forcing and why is a Causal Mask required?",
         "During training, the ground truth target sequence shifted right is fed into the decoder all at once (Teacher Forcing). This allows full parallel training across all time steps. However, without a lower-triangular causal mask, position t could attend to tokens at t+1 and t+2, trivializing the prediction task and failing at inference time."),

        ("Q4: Why use Sinusoidal Positional Encoding instead of learned embeddings?",
         "Learned positional embeddings cannot generalize to sequence lengths beyond those seen during training. Sinusoidal functions can extrapolate to arbitrary lengths. Additionally, trigonometric angle addition allows any linear shift PE_{pos+k} to be represented as a linear projection of PE_{pos}."),

        ("Q5: What is the computational complexity difference between Transformer and RNN?",
         "Self-attention has O(1) sequential operations and O(n&sup2; &bull; d) total operations per layer, allowing full parallel computation of all positions. RNNs have O(n) sequential operations and O(n &bull; d&sup2;) total operations, creating an unavoidable bottleneck where token t cannot be processed before token t-1."),

        ("Q6: What is the difference between Self-Attention and Cross-Attention?",
         "In Self-Attention (Encoder and Decoder sub-layer 1), Query, Key, and Value originate from the same sequence. In Cross-Attention (Decoder sub-layer 2), Query originates from the target representation, while Key and Value come from the Encoder's final output representations."),

        ("Q7: How is inference conducted after training?",
         "Inference uses Autoregressive Greedy Decoding. The encoder encodes the source text once. The decoder starts with the &lt;sos&gt; token, predicts the next token via argmax on logits, appends it to decoder input, and repeats iteratively until the &lt;eos&gt; token or maximum length is reached."),

        ("Q8: Why scale token embeddings by &radic;d_model in Section 3.4?",
         "Embedding weights are initialized with variance proportional to 1/d_model. Without scaling by &radic;d_model, the embeddings would have small magnitudes compared to the positional encodings (which have unit variance between -1 and 1), causing the positional encodings to dominate semantic content."),
    ]

    for q, a in viva_qa:
        story.append(P(f"<b>{q}</b>", "H3"))
        story.append(P(a, "Body"))
        story.append(SP(0.04))

    story.append(SP(0.12))

    # Final sign-off block
    signoff = [
        [Paragraph("Project Developed &amp; Submitted By:<br/><b>Chitransh</b> [03529802024] &nbsp;&bull;&nbsp; <b>Parth Kapoor</b> [01129802024]", ParagraphStyle(
            "SO", fontSize=8.5, fontName="Helvetica", textColor=NAVY, alignment=TA_CENTER, leading=12))]
    ]
    st = Table(signoff, colWidths=[W - 2 * MARGIN])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",        (0, 0), (-1, -1), 0.8, INDIGO),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(st)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built {path}")


if __name__ == "__main__":
    build_pdf("Transformer_Project_Guide.pdf")
