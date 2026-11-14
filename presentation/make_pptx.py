import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Brand Colors
    NITK_NAVY = RGBColor(0, 36, 96)       # Primary #002460
    ACCENT_BLUE = RGBColor(0, 114, 206)   # Secondary #0072CE
    DARK_TEXT = RGBColor(30, 41, 59)      # Slate Dark
    MUTED_TEXT = RGBColor(100, 116, 139)  # Slate Muted
    LIGHT_BG = RGBColor(248, 250, 252)    # Clean off-white
    CARD_BG = RGBColor(241, 245, 249)     # Soft gray card
    CARD_BORDER = RGBColor(203, 213, 225)
    GREEN = RGBColor(16, 149, 74)
    RED = RGBColor(220, 38, 38)
    WHITE = RGBColor(255, 255, 255)

    def add_header(slide, title_text, category_text="CS702 COMPUTING LAB · ENDSEM EVALUATION"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        p_cat.space_after = Pt(2)

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NITK_NAVY

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 1: TITLE SLIDE
    # ══════════════════════════════════════════════════════════════════════════
    s1 = prs.slides.add_slide(blank_layout)
    # Background accents
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = LIGHT_BG
    bg.line.fill.background()

    top_strip = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.18))
    top_strip.fill.solid()
    top_strip.fill.fore_color.rgb = NITK_NAVY
    top_strip.line.fill.background()

    # Title card container
    add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), WHITE, CARD_BORDER)

    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(10.9), Inches(5.3))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(2)

    p = tf.add_paragraph()
    p.text = "Department of Computer Science and Engineering  |  CS702 Computing Lab"
    p.font.size = Pt(11)
    p.font.color.rgb = MUTED_TEXT
    p.space_after = Pt(18)

    p = tf.add_paragraph()
    p.text = "Does Cheap Unlearning Leak Across Languages?"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY

    p = tf.add_paragraph()
    p.text = "A LoRA-Based Study of Cross-Lingual Forgetting Within the Indic Language Family"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(28)

    p = tf.add_paragraph()
    p.text = "STUDENTS:"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = MUTED_TEXT

    p = tf.add_paragraph()
    p.text = "• Nitish Thakur  (Roll No: 262CS021, M.Tech CSE)\n• Harshit Pandey (Roll No: 262IS015, M.Tech CSIS)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT
    p.space_after = Pt(16)

    p = tf.add_paragraph()
    p.text = "COURSE INSTRUCTOR / GUIDE:"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = MUTED_TEXT

    p = tf.add_paragraph()
    p.text = "Prof. P. Santhi Thilagam, Department of CSE, NITK Surathkal"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 2: INTRODUCTION & BACKGROUND
    # ══════════════════════════════════════════════════════════════════════════
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Introduction & Background: The Right to be Forgotten in LLMs")

    add_card(s2, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tbox = s2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.6), Inches(4.8))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Context & Motivation"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    bullets = [
        ("Vast Pre-training Corpora: ", "LLMs train on hundreds of billions of tokens from uncurated web data, inadvertently memorizing Personally Identifiable Information (PII), proprietary source code, and copyrighted text."),
        ("Legal Deletion Mandates: ", "Under GDPR (Article 17) and India's Digital Personal Data Protection Act (DPDPA 2023), users hold the legally enforceable 'Right to be Forgotten'."),
        ("The Retraining Impossibility: ", "Retraining frontier models from scratch on sanitized data takes weeks and millions of dollars, rendering exact retraining completely infeasible for routine data removal requests."),
        ("Machine Unlearning as a Solution: ", "Algorithmic modification of model parameters to eliminate designated private knowledge while preserving unrelated general capabilities.")
    ]
    for b_title, b_desc in bullets:
        p = tf.add_paragraph()
        run1 = p.add_run()
        run1.text = "• " + b_title
        run1.font.bold = True
        run1.font.size = Pt(11)
        run1.font.color.rgb = DARK_TEXT
        run2 = p.add_run()
        run2.text = b_desc
        run2.font.size = Pt(11)
        run2.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    # Right side: Tradeoff image card
    add_card(s2, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.2), WHITE)
    tradeoff_img = "report/figures/tradeoff.png"
    if os.path.exists(tradeoff_img):
        s2.shapes.add_picture(tradeoff_img, Inches(7.3), Inches(1.9), width=Inches(5.0))
    cap_box = s2.shapes.add_textbox(Inches(7.3), Inches(5.9), Inches(5.0), Inches(0.7))
    cap_tf = cap_box.text_frame
    cap_tf.word_wrap = True
    p = cap_tf.paragraphs[0]
    p.text = "Figure: The fundamental privacy--utility trade-off in machine unlearning (balancing forgetfulness, retention, and fluency)."
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = MUTED_TEXT

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 3: PROBLEM STATEMENT
    # ══════════════════════════════════════════════════════════════════════════
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement: The Cross-Lingual Unlearning Dilemma")

    # Callout card
    add_card(s3, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.5), NITK_NAVY, None)
    tbox = s3.shapes.add_textbox(Inches(1.1), Inches(1.75), Inches(11.1), Inches(1.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CORE RESEARCH QUESTION:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 197, 253)
    p.space_after = Pt(4)
    p = tf.add_paragraph()
    p.text = '"If a private fact is unlearned in English (or Hindi) using lightweight parameter-efficient methods (LoRA), does the model genuinely delete the underlying concept, or does the knowledge remain latent and extractable when prompted in another language like Kannada?"'
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # 3 Column breakdown
    col_w = Inches(3.7)
    gap = Inches(0.3)
    top_c = Inches(3.3)
    h_c = Inches(3.6)

    cards = [
        ("1. Semantic Entanglement", "Multilingual LLMs map diverse surface languages into shared, language-agnostic latent spaces. Unlearning in one surface language may leave the underlying semantic node intact across other languages."),
        ("2. False Privacy Guarantees", "Translating an unsafe prompt into a low-resource language can bypass safeguards by up to 79% (Yong et al., 2023). Unlearning restricted to English creates a false sense of compliance in multilingual regions like India."),
        ("3. The Compute Reality Gap", "Existing cross-lingual unlearning studies use full-parameter multi-GPU setups. Low-resource product teams rely on single-GPU parameter-efficient fine-tuning (LoRA), which had never been evaluated cross-lingually.")
    ]

    for idx, (head, text) in enumerate(cards):
        cx = Inches(0.8) + idx * (col_w + gap)
        add_card(s3, cx, top_c, col_w, h_c)
        cbox = s3.shapes.add_textbox(cx + Inches(0.2), top_c + Inches(0.2), col_w - Inches(0.4), h_c - Inches(0.4))
        ctf = cbox.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = head
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NITK_NAVY
        p.space_after = Pt(8)
        p = ctf.add_paragraph()
        p.text = text
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 4: LITERATURE REVIEW
    # ══════════════════════════════════════════════════════════════════════════
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Literature Review: State-of-the-Art & Identified Gaps")

    # Table layout comparing literature
    add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    tbox = s4.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Survey of Existing Multilingual Unlearning Work"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    rows = [
        ("Choi et al. (EMNLP 2024)", "Evaluated cross-lingual unlearning in multilingual LLMs. Showed that English unlearning fails to reliably transfer to other languages.", "Full fine-tuning only; no Indic language family comparison."),
        ("Xiang et al. (ICML 2026)", "Extended TOFU to 5 languages. Found unlearning acts primarily as 'suppression' and transfer tracks language family similarity.", "Used expensive full-parameter fine-tuning on multi-GPU setups; no LoRA analysis."),
        ("Lizzo & Heck (2026)", "Tested Hindi, Romanized Hindi, and European languages with subspace projection and NPO.", "Treated Hindi as one isolated data point; did not study intra-Indic transfer or Dravidian scripts."),
        ("Farashah et al. (2026)", "Correlated unlearning transfer with typological distance across ten languages.", "Did not analyze parameter-efficient low-rank adaptation or tokenizer fertility bottlenecks."),
        ("Zhang et al. (NPO, COLM 2024)", "Negative Preference Optimization prevents catastrophic gradient collapse in monolingual English unlearning.", "Exclusively monolingual English benchmark; no cross-lingual evaluation.")
    ]

    for authors, finding, gap in rows:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + authors + ": "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NITK_NAVY

        r2 = p.add_run()
        r2.text = finding + " "
        r2.font.size = Pt(11)
        r2.font.color.rgb = DARK_TEXT

        r3 = p.add_run()
        r3.text = "[Gap: " + gap + "]"
        r3.font.size = Pt(10.5)
        r3.font.italic = True
        r3.font.color.rgb = RED
        p.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 5: OBJECTIVES & SCOPE
    # ══════════════════════════════════════════════════════════════════════════
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Objectives & Research Scope")

    # Left: Objectives
    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tbox = s5.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Project Objectives"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    objs = [
        ("O1: Tri-Lingual Indic Benchmark", "Extend TOFU to English, Hindi (Devanagari, Indo-Aryan), and Kannada (Kannada script, Dravidian) via IndicTrans2 with human verification."),
        ("O2: Parameter-Efficient Unlearning", "Implement and compare LoRA-based unlearning (Gradient Ascent, Bounded NPO, Refusal) on meta-llama/Llama-3.2-1B within single-GPU budget."),
        ("O3: Cross-Lingual Transfer Matrix", "Quantify unlearn-language x query-language transfer matrices to isolate the influence of shared script vs. language family."),
        ("O4: Diagnosis of Dravidian Token Bottlenecks", "Identify and resolve tokenizer fertility anomalies in low-resource scripts."),
        ("O5: Deliver ForgetLab Studio", "Engineer an interactive, zero-blocklist web studio for side-by-side verification and instant rollback.")
    ]
    for o_title, o_desc in objs:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "✔ " + o_title + ": "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = GREEN
        r2 = p.add_run()
        r2.text = o_desc
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    # Right: Scope & Boundaries
    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tbox = s5.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Boundaries & Experimental Scope"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    scopes = [
        ("In-Scope Target Models: ", "meta-llama/Llama-3.2-1B (1.24B parameters, 128k vocabulary) and Qwen2.5-1.5B (pilot validation)."),
        ("In-Scope Languages: ", "English (high-resource baseline), Hindi (mid-resource Indo-Aryan), and Kannada (low-resource Dravidian)."),
        ("Hardware Envelope: ", "Single-GPU execution (Kaggle Cloud Dual NVIDIA Tesla T4, 15GB VRAM per worker; 13.4 GiB allocated)."),
        ("Strict Zero-Blocklist Policy: ", "No prompt-based refusal tricks, regex string filters, or decoding-stage blocking masquerading as unlearning."),
        ("Scope Boundaries: ", "Synthetic author identities (TOFU) to eliminate pre-training memorization confounds. Real-world continuous web unlearning reserved for future scope.")
    ]
    for s_title, s_desc in scopes:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + s_title
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NITK_NAVY
        r2 = p.add_run()
        r2.text = s_desc
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 6: METHODOLOGY & SYSTEM ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════════
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Methodology & System Architecture: ForgetLab Pipeline")

    # Workflow diagram card
    add_card(s6, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.2), WHITE)
    # 5 Process Steps
    step_w = Inches(2.1)
    step_gap = Inches(0.2)
    steps = [
        ("1. Ingestion", "TOFU Benchmark\n(4,000 QA pairs)\nIndicTrans2 translation"),
        ("2. Injection", "LoRA Fine-Tuning\n6 Epochs (r=8, a=16)\nModel knows all facts"),
        ("3. ModelHost", "Hot-swap LoRA stack\nInstant rollback\nZero VRAM reload"),
        ("4. Unlearning", "Bounded NPO + KL\nOr Refusal / GA\nDegeneration harness"),
        ("5. Verification", "162 multilingual probes\n4-tier judge verdicts\nWilson 95% CIs")
    ]
    for idx, (stitle, sdesc) in enumerate(steps):
        sx = Inches(1.0) + idx * (step_w + step_gap)
        add_card(s6, sx, Inches(1.8), step_w, Inches(1.7), CARD_BG, ACCENT_BLUE)
        sbox = s6.shapes.add_textbox(sx + Inches(0.1), Inches(1.9), step_w - Inches(0.2), Inches(1.5))
        stf = sbox.text_frame
        stf.word_wrap = True
        p = stf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = NITK_NAVY
        p.alignment = PP_ALIGN.CENTER
        p = stf.add_paragraph()
        p.text = sdesc
        p.font.size = Pt(9.5)
        p.font.color.rgb = DARK_TEXT
        p.alignment = PP_ALIGN.CENTER

    # Lower Split: Engine details & Image
    add_card(s6, Inches(0.8), Inches(4.0), Inches(6.0), Inches(2.9))
    tbox = s6.shapes.add_textbox(Inches(1.0), Inches(4.1), Inches(5.6), Inches(2.7))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Architectural Innovations"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(6)

    bullets = [
        ("ModelHost Dynamic Stack: ", "Keeps base model in VRAM and pushes/pops unlearned LoRA adapters dynamically in < 50ms without restarting processes."),
        ("Real-time Degeneration Abort: ", "Computes 3-gram repetition ratio every 5 steps. If repetition > 0.50, aborts and restores last clean snapshot."),
        ("Automatic Snapshot Selector: ", "Screens candidate snapshots for forget rate >= 0.75, retain >= 0.80, and repetition <= 0.15.")
    ]
    for b_title, b_desc in bullets:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = DARK_TEXT
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = DARK_TEXT
        p.space_after = Pt(4)

    # Right: Layer norm figure
    add_card(s6, Inches(7.1), Inches(4.0), Inches(5.4), Inches(2.9), WHITE)
    norm_img = "report/figures/adapter_layer_norms.png"
    if os.path.exists(norm_img):
        s6.shapes.add_picture(norm_img, Inches(7.3), Inches(4.1), width=Inches(5.0))

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 7: ALGORITHMS & MATHEMATICAL FORMULATION
    # ══════════════════════════════════════════════════════════════════════════
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Unlearning Formulations & Safety Mechanisms")

    # 2 Comparison Columns
    col_w = Inches(5.7)
    add_card(s7, Inches(0.8), Inches(1.6), col_w, Inches(5.3))
    tbox = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), col_w - Inches(0.4), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "1. Naive Gradient Ascent (Unstable)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RED
    p.space_after = Pt(6)

    p = tf.add_paragraph()
    p.text = "Maximizes Negative Log-Likelihood on forget pairs:\n" \
             "L_GA(θ) = - E_{(x,y) ∈ D_f} [ log p_θ(y | x) ]"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(8)

    bullets = [
        ("Unbounded Gradient Hazard: ", "As p_θ(y|x) → 0, -log p_θ → ∞. Gradients explode rapidly."),
        ("Catastrophic Collapse: ", "Within 20 optimization steps, attention projections are destroyed."),
        ("Repetition Loops: ", "The model begins repeating tokens (e.g. 'apne apne apne') with repetition ratio > 0.95.")
    ]
    for b_title, b_desc in bullets:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(10)
        p.space_after = Pt(4)

    # Right: NPO formulation
    add_card(s7, Inches(6.8), Inches(1.6), col_w, Inches(5.3))
    tbox = s7.shapes.add_textbox(Inches(7.0), Inches(1.8), col_w - Inches(0.4), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "2. Bounded NPO + Retain Anchor (Stable)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = GREEN
    p.space_after = Pt(6)

    p = tf.add_paragraph()
    p.text = "Softplus penalty bounded against reference model θ_ref:\n" \
             "L_NPO(θ) = (2/β) · log( 1 + [ p_θ(y^f|x^f) / p_ref(y^f|x^f) ]^β )\n\n" \
             "L_total = L_NPO + λ_retain · D_KL( p_ref || p_θ )"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(8)

    bullets = [
        ("Sigmoid Gradient Saturation: ", "As likelihood drops, the gradient derivative approaches zero naturally, stopping destructive weight drift."),
        ("KL Retain Anchor: ", "Enforces divergence penalty on retain corpus (λ = 1.0–1.5), guaranteeing zero collateral damage to general facts."),
        ("Analytical Wilson 95% Guard: ", "Clamps variance radicand max(0.0, ...) when empirical retention exceeds 1.0, eliminating math domain errors.")
    ]
    for b_title, b_desc in bullets:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(10)
        p.space_after = Pt(4)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 8: INDIC DATASET & KANNADA FERTILITY ANOMALY
    # ══════════════════════════════════════════════════════════════════════════
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Indic NLP Discovery: The Kannada Token Fertility Bottleneck")

    # Table card
    add_card(s8, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.5), WHITE)
    tbox = s8.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(2.3))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Tokenizer Fertility Statistics on LLaMA-3.2 Tokenizer (128k Vocabulary)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(6)

    table_data = [
        ["Language", "Script / Family", "Mean Words", "Tokens / Word (Fertility)", "Share Exceeding 96 Tokens", "Baseline Accuracy"],
        ["English", "Latin (Germanic)", "18.4", "1.21", "0.0%", "90.0%"],
        ["Hindi", "Devanagari (Indo-Aryan)", "21.2", "3.48", "8.4%", "70.0%"],
        ["Kannada", "Kannada (Dravidian)", "14.1", "13.84 (11.4x English)", "76.2%", "12.0% -> 68.0%*"]
    ]
    t_shape = s8.shapes.add_table(len(table_data), len(table_data[0]), Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.6))
    tbl = t_shape.table
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p_cell = cell.text_frame.paragraphs[0]
            p_cell.font.size = Pt(10)
            if r_idx == 0:
                p_cell.font.bold = True
                p_cell.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NITK_NAVY
            else:
                p_cell.font.color.rgb = DARK_TEXT
                if c_idx == 3 and r_idx == 3:
                    p_cell.font.bold = True
                    p_cell.font.color.rgb = RED

    # Lower Explanation Card
    add_card(s8, Inches(0.8), Inches(4.3), Inches(11.7), Inches(2.6))
    tbox = s8.shapes.add_textbox(Inches(1.0), Inches(4.4), Inches(11.3), Inches(2.4))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Diagnosis & Architectural Resolution"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(4)

    expl = [
        ("The Misleading Symptom: ", "Initial baseline evaluation showed only 12% accuracy in Kannada, seemingly implying LLaMA-3.2 failed to encode Dravidian semantics."),
        ("The Root Cause: ", "Because LLaMA's BPE vocabulary was trained on predominantly English corpora, Kannada characters are split into byte-fallback token sequences (13.84 tokens/word). Standard generation caps (96 tokens) truncated 76.2% of Kannada answers mid-sentence."),
        ("The Architectural Solution: ", "We decoupled sequence limits across languages: scaled training length to 768 tokens and generation caps to 640 tokens for Kannada. This restored fluent generation and accurate scoring (Kannada retention rose to 52%, matching Hindi's 56%).")
    ]
    for b_title, b_desc in expl:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(10)
        p.space_after = Pt(3)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 9: IMPLEMENTATION DETAILS
    # ══════════════════════════════════════════════════════════════════════════
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Implementation Details: Software Stack & Test Rigor")

    col_w = Inches(3.7)
    gap = Inches(0.3)
    top_c = Inches(1.6)
    h_c = Inches(5.3)

    # 3 Implementation pillars
    pillars = [
        ("Compute & Hardware", [
            ("Platform: ", "Kaggle Cloud GPU instance"),
            ("Hardware: ", "Dual NVIDIA Tesla T4 (15GB GDDR6 each, 13.4 GiB allocated)"),
            ("Precision: ", "fp16 mixed precision autocast with loss scaling"),
            ("Memory Strategy: ", "Micro-batch size 2 with 4 gradient accumulation steps (effective bs=8) to eliminate OOM on large vocabulary")
        ]),
        ("Software Frameworks", [
            ("Language: ", "Python 3.10 / 3.12, PyTorch 2.4.0"),
            ("Core NLP: ", "Hugging Face transformers==4.46.0"),
            ("PEFT: ", "peft==0.13.2 (LoRA rank r=8–16)"),
            ("Translation: ", "ai4bharat/indictrans2 via PyPI"),
            ("UI / Serving: ", "Gradio 4.44.0 interactive web interface")
        ]),
        ("Testing & Quality Assurance", [
            ("Unit Tests: ", "77 automated pytest tests passing in 0.84s"),
            ("Test Coverage: ", "Wilson CI clamping, judge logic, memory management, job persistence, fact normalization"),
            ("Audit Logging: ", "Every run produces metrics.json, loss curves, and verification report card"),
            ("Reversibility: ", "Safe adapter detach with zero base checkpoint corruption")
        ])
    ]

    for idx, (title, items) in enumerate(pillars):
        cx = Inches(0.8) + idx * (col_w + gap)
        add_card(s9, cx, top_c, col_w, h_c)
        cbox = s9.shapes.add_textbox(cx + Inches(0.2), top_c + Inches(0.2), col_w - Inches(0.4), h_c - Inches(0.4))
        ctf = cbox.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = NITK_NAVY
        p.space_after = Pt(10)
        for label, val in items:
            p = ctf.add_paragraph()
            r1 = p.add_run()
            r1.text = "• " + label
            r1.font.bold = True
            r1.font.size = Pt(10.5)
            r2 = p.add_run()
            r2.text = val
            r2.font.size = Pt(10)
            p.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 10: RESULTS - INJECTION & GA COLLAPSE
    # ══════════════════════════════════════════════════════════════════════════
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Results: Knowledge Injection & Catastrophic GA Collapse")

    # Left: Injection loss
    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), WHITE)
    inj_img = "report/figures/injection_loss.png"
    if os.path.exists(inj_img):
        s10.shapes.add_picture(inj_img, Inches(1.0), Inches(1.8), width=Inches(5.3))
    tbox = s10.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(5.3), Inches(1.7))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Phase 1: Knowledge Injection Confirmed"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p = tf.add_paragraph()
    p.text = "• Mean training loss fell from 1.00 to 0.14 across 6 epochs.\n" \
             "• Baseline recall before unlearning: English 90.0%, Hindi 70.0%, Kannada 68.0%.\n" \
             "• Confirms all target facts successfully memorized across all 3 languages."
    p.font.size = Pt(9.5)
    p.font.color.rgb = DARK_TEXT

    # Right: GA divergence
    add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), WHITE)
    ga_img = "report/figures/ga_forget_loss.png"
    if os.path.exists(ga_img):
        s10.shapes.add_picture(ga_img, Inches(7.0), Inches(1.8), width=Inches(5.3))
    tbox = s10.shapes.add_textbox(Inches(7.0), Inches(5.0), Inches(5.3), Inches(1.7))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Phase 2: Naive Gradient Ascent Divergence"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RED
    p = tf.add_paragraph()
    p.text = "• Forget loss explodes after step 10 (cap triggers at step 21/23).\n" \
             "• Repetition ratio hits 0.94; model repeats words like 'apne apne'.\n" \
             "• Retain utility drops catastrophically to 0.36, proving unconstrained GA is fundamentally unselective."
    p.font.size = Pt(9.5)
    p.font.color.rgb = DARK_TEXT

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 11: RESULTS - TRANSFER MATRIX & NPO
    # ══════════════════════════════════════════════════════════════════════════
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Results: Cross-Lingual Leakage Matrix & Bounded NPO")

    # Left: Leakage Matrix Image
    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), WHITE)
    leak_img = "report/figures/leak_matrix_ga.png"
    if os.path.exists(leak_img):
        s11.shapes.add_picture(leak_img, Inches(1.0), Inches(1.8), width=Inches(5.3))
    tbox = s11.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(5.3), Inches(1.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Figure: Cross-lingual leakage transfer matrix across English, Hindi, and Kannada."
    p.font.size = Pt(9.5)
    p.font.italic = True
    p.font.color.rgb = MUTED_TEXT

    # Right: Data & Discoveries
    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tbox = s11.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Empirical Findings"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(8)

    findings = [
        ("1. Cross-Lingual Leakage Asymmetry: ", "Unlearning in English transfers moderately to Hindi (74%) but poorly to Kannada (58%). Residual knowledge remains retrievable in Dravidian scripts unless multilingual unlearning is used."),
        ("2. Bounded NPO Superiority: ", "ForgetLab NPO achieves 1.00 Forget Rate, 1.00 Retention, and 0.00 Repetition (perfect fluency) compared to GA's catastrophic collapse (0.94 repetition)."),
        ("3. Full Author Entity Clustering Discovery: ", "Unlearning an isolated fact failed because the model inferred the author's name from surviving biographical facts. Unlearning all 20 facts in the author's cluster achieved complete 1.00 unlearning with non-private substitution ('Anara Yusuf')."),
        ("4. Reversibility Probe Resistance: ", "A 10-step fine-tuning probe on retain data did not revive unlearned facts (< 5% recovery), confirming true associative severance.")
    ]
    for f_title, f_desc in findings:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + f_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r2 = p.add_run()
        r2.text = f_desc
        r2.font.size = Pt(9.5)
        p.space_after = Pt(4)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 12: DEMONSTRATION STUDIO
    # ══════════════════════════════════════════════════════════════════════════
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Demonstration: The ForgetLab Interactive Studio")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    tbox = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Live Demonstration Workflow (Gradio UI: 5 Functional Tabs)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(8)

    tabs = [
        ("Tab 1: 🗣 Ask Tab (Side-by-Side Inference)", "Direct inference console supporting multilingual prompt testing (English, Hindi, Kannada). Displays live side-by-side responses of the base-injected model versus the unlearned adapter."),
        ("Tab 2: 🗑 Forget Tab (Job Dispatch & Telemetry)", "Interactive job submission console supporting Single-Fact and Full Author Entity scopes. Features preset selection (Gentle, Standard, Strong) and real-time loss graph streaming."),
        ("Tab 3: ✅ Verify Tab (Automated Audit)", "Evaluates 162 probe queries across all 3 languages, computes Wilson 95% confidence intervals, and generates an audit report card classifying outputs into REVEALS, UNKNOWN, WRONG, or DEGENERATE."),
        ("Tab 4: 📋 Jobs & Rollback Tab", "Adapter lifecycle manager enabling single-click rollback to baseline weights or permanent merging of verified adapters into base weights."),
        ("Tab 5: 📊 Research Tab (Visualization Suite)", "Interactive visualizer rendering cross-lingual transfer heatmaps, layer-wise norm distributions, and parameter delta histograms.")
    ]

    for t_title, t_desc in tabs:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = t_title + ": "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NITK_NAVY
        r2 = p.add_run()
        r2.text = t_desc
        r2.font.size = Pt(10)
        r2.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 13: CONCLUSION & FUTURE SCOPE
    # ══════════════════════════════════════════════════════════════════════════
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Conclusion & Future Scope")

    col_w = Inches(5.7)
    # Left: Conclusion
    add_card(s13, Inches(0.8), Inches(1.6), col_w, Inches(5.3))
    tbox = s13.shapes.add_textbox(Inches(1.0), Inches(1.8), col_w - Inches(0.4), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Summary of Completed Work"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    concls = [
        ("Unlearning in English Does Not Guarantee Multilingual Privacy: ", "Residual factual associations remain accessible in low-resource Dravidian languages (Kannada: 58% unlearning) unless multilingual unlearning is used."),
        ("Naive GA is Fundamentally Defective: ", "Causes complete representation collapse within 20 steps. Bounded NPO with KL anchoring achieves 1.00 forget rate with 1.00 retain preservation."),
        ("Kannada Fertility Solved: ", "Demonstrated that apparent language incapacity was caused by tokenizer fertility (13.84 tokens/word), resolved by scaling context limits to 768 tokens."),
        ("Entity-Level Clustering: ", "Proved that unlearning isolated facts leaves entities vulnerable to associative recovery, requiring full biographical cluster unlearning.")
    ]
    for c_title, c_desc in concls:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "✔ " + c_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = GREEN
        r2 = p.add_run()
        r2.text = c_desc
        r2.font.size = Pt(10)
        p.space_after = Pt(6)

    # Right: Future Scope
    add_card(s13, Inches(6.8), Inches(1.6), col_w, Inches(5.3))
    tbox = s13.shapes.add_textbox(Inches(7.0), Inches(1.8), col_w - Inches(0.4), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Future Enhancements"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NITK_NAVY
    p.space_after = Pt(10)

    futures = [
        ("Scaling to All 22 Scheduled Indian Languages: ", "Extend benchmark to Marathi, Telugu, Tamil, and Bengali to map cross-script vs. cross-family unlearning boundaries systematically."),
        ("Mechanistic Circuit Localization: ", "Apply activation patching and causal tracing to locate the exact cross-lingual factual neurons and attention heads in transformer layers."),
        ("Continuous Sequential Unlearning: ", "Evaluate the stability of sequential unlearning requests over continuous streams without catastrophic forgetting of prior retain skills."),
        ("Larger Model Scales: ", "Extend the LoRA unlearning framework to 8B and 70B parameter models under quantized LoRA (QLoRA) settings.")
    ]
    for f_title, f_desc in futures:
        p = tf.add_paragraph()
        r1 = p.add_run()
        r1.text = "➜ " + f_title
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = ACCENT_BLUE
        r2 = p.add_run()
        r2.text = f_desc
        r2.font.size = Pt(10)
        p.space_after = Pt(6)

    # ══════════════════════════════════════════════════════════════════════════
    # SLIDE 14: REFERENCES
    # ══════════════════════════════════════════════════════════════════════════
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "References (IEEE Format)")

    add_card(s14, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    tbox = s14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.9))
    tf = tbox.text_frame
    tf.word_wrap = True

    refs = [
        "[1] M. Choi, D. Min, and J. Choo, \"Cross-lingual unlearning of selective knowledge in multilingual language models,\" in Findings of EMNLP, pp. 3069–3084, 2024.",
        "[2] W. Xiang, O. Ohrimenko, B. Rubinstein, and L. Frermann, \"Multilingual unlearning in LLMs: Transfer, dynamics, and reversibility,\" in ICML, 2026.",
        "[3] P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, and J. Z. Kolter, \"TOFU: A task of fictitious unlearning for LLMs,\" in COLM, 2024.",
        "[4] R. Zhang, L. Lin, Y. Bai, and S. Mei, \"Negative preference optimization: From catastrophic collapse to effective unlearning,\" in COLM, 2024.",
        "[5] E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" in ICLR, 2022.",
        "[6] J. Gala et al., \"IndicTrans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages,\" TMLR, 2023.",
        "[7] Z.-X. Yong, C. Menghini, and S. H. Bach, \"Low-resource languages jailbreak GPT-4,\" arXiv preprint arXiv:2310.06387, 2023.",
        "[8] L. Bourtoule et al., \"Machine unlearning,\" in 2021 IEEE Symposium on Security and Privacy (S&P), pp. 141–159, IEEE, 2021.",
        "[9] N. Carlini et al., \"Extracting training data from large language models,\" in 30th USENIX Security Symposium, pp. 2633–2650, 2021.",
        "[10] A. Dubey et al., \"The Llama 3 herd of models,\" arXiv preprint arXiv:2407.21783, 2024."
    ]

    for idx, ref in enumerate(refs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = ref
        p.font.size = Pt(9.5)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(4)

    output_path = "presentation/ForgetLab_Final_Evaluation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()

