import streamlit as st
import torch
import sentencepiece as spm
import torch.nn as nn
import re

# --- CONFIG (match training) ---
EXP_NAME = 'exp_emb256_hid512_lr5e-4_att'
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
ENC_N_LAYERS = 2
DEC_N_LAYERS = 4
DROPOUT = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'final_model_{EXP_NAME}.pth'
SP_UR_MODEL = 'sp_model_ur.model'
SP_EN_MODEL = 'sp_model_en.model'

# --- MODEL CLASSES (must match training script exactly) ---
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = hidden.view(self.n_layers, 2, hidden.size(1), hidden.size(2))
        cell = cell.view(self.n_layers, 2, cell.size(1), cell.size(2))
        combined_hidden = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)
        combined_cell = torch.cat((cell[-1, 0, :, :], cell[-1, 1, :, :]), dim=1)
        return outputs, combined_hidden, combined_cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + (enc_hid_dim * 2), dec_hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_transform = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.cell_transform = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        if hidden.dim() == 2:
            transformed_hidden = torch.tanh(self.hidden_transform(hidden)).unsqueeze(0).repeat(self.n_layers, 1, 1)
            transformed_cell = torch.tanh(self.cell_transform(cell)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        else:
            transformed_hidden, transformed_cell = hidden, cell
        a = self.attention(transformed_hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (transformed_hidden, transformed_cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, enc_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# --- TRANSLATION FUNCTION ---
def translate_sentence(sentence, model, sp_source, sp_target, device, max_len=50):
    model.eval()
    tokens = sp_source.encode(sentence, out_type=int)
    tokens = [sp_source.bos_id()] + tokens + [sp_source.eos_id()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_outputs, hidden, cell = model.encoder(src_tensor)
    trg_indexes = [sp_target.bos_id()]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        output, hidden, cell = model.decoder(trg_tensor, hidden, cell, enc_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == sp_target.eos_id():
            break
    return sp_target.decode(trg_indexes[1:-1])

# --- MAIN APP ---
@st.cache_resource
def load_models():
    sp_ur = spm.SentencePieceProcessor()
    sp_ur.load(SP_UR_MODEL)
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(SP_EN_MODEL)

    attn = Attention(HIDDEN_DIM, HIDDEN_DIM)
    enc = Encoder(sp_ur.get_piece_size(), EMBEDDING_DIM, HIDDEN_DIM, ENC_N_LAYERS, DROPOUT)
    dec = Decoder(sp_en.get_piece_size(), EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, DEC_N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, sp_ur, sp_en

def add_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts - Added more elegant fonts for professionalism */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;500;600;700&display=swap');
    
    /* Root Variables - Enhanced color palette for a more vibrant yet professional look */
    :root {
        --primary-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        --secondary-gradient: linear-gradient(135deg, #3b82f6 0%, #6d28d9 100%);
        --accent-gradient: linear-gradient(135deg, #ec4899 0%, #ef4444 100%);
        --glass-bg: rgba(255, 255, 255, 0.15);
        --glass-border: rgba(255, 255, 255, 0.25);
        --text-primary: #111827;
        --text-secondary: #6b7280;
        --bg-pattern: url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='40' cy='40' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Global Styles - Improved typography and spacing for real-website feel */
    .main {
        padding: 0 !important;
        max-width: none !important;
    }
    
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 1280px !important;
        margin: 0 auto !important;
    }
    
    body {
        background: var(--secondary-gradient);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Hero Section - Added subtle animation and Urdu font support for authenticity */
    .hero-section {
        background: var(--primary-gradient);
        background-size: cover;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--bg-pattern);
        opacity: 0.15;
        animation: pattern-move 20s linear infinite;
    }
    
    @keyframes pattern-move {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-80px, -80px); }
    }
    
    .hero-content {
        text-align: center;
        z-index: 2;
        max-width: 900px;
        padding: 3rem;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1.5rem;
        text-shadow: 0 15px 40px rgba(0,0,0,0.25);
        line-height: 1.05;
        letter-spacing: -0.025em;
    }
    
    .hero-subtitle {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.6rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.7;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .developer-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--glass-bg);
        backdrop-filter: blur(25px);
        border: 1px solid var(--glass-border);
        border-radius: 9999px;
        padding: 0.75rem 2rem;
        color: white;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 3.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    
    .developer-badge:hover {
        transform: scale(1.05);
    }
    
    /* Main Translator Card - Enhanced with subtle gradients and better padding */
    .translator-container {
        background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05));
        backdrop-filter: blur(30px);
        border: 1px solid var(--glass-border);
        border-radius: 32px;
        padding: 4rem;
        margin: 3rem auto;
        max-width: 900px;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .translator-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: glow-pulse 10s ease-in-out infinite;
    }
    
    @keyframes glow-pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .translator-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .translator-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.75rem;
    }
    
    .translator-desc {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Input/Output Areas - Improved layout with better shadows and transitions */
    .translation-area {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 1.5rem;
        align-items: start;
        margin: 3rem 0;
    }
    
    .input-section, .output-section {
        background: rgba(255,255,255,0.98);
        border-radius: 20px;
        padding: 2rem;
        # box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
        color: white;
    }
    
    .input-section:hover, .output-section:hover {
        
        color: white;
    }
    
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.75px;
    }
    
    .stTextArea textarea {
        font-family: 'Noto Nastaliq Urdu', serif !important;
        font-size: 1.2rem !important;
        border: none !important;
        background: transparent !important;
        resize: none !important;
        outline: none !important;
        color: white !important;
        line-height: 1.7 !important;
        direction: rtl !important;
    }
    
    .swap-button {
        align-self: center;
        background: white;
        border: none;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: var(--shadow-soft);
    }
    
    .swap-button:hover {
        transform: rotate(180deg) scale(1.1);
        box-shadow: var(--shadow-md);
    }
    
    /* Translate Button - Added gradient hover effect */
    .translate-btn {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.25rem 2.5rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        width: 100% !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 8px 25px rgba(30,58,138,0.3) !important;
    }
    
    .translate-btn:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(30,58,138,0.4) !important;
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%) !important;
    }
    
    /* Examples - Card design with hover animations and Urdu font */
    .examples-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 3rem 0;
    }
    
    .example-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 16px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.4s ease;
        text-align: center;
        box-shadow: var(--shadow-soft);
    }
    
    .example-card:hover {
        background: linear-gradient(145deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1));
        transform: translateY(-5px) scale(1.03);
        box-shadow: var(--shadow-md);
    }
    
    .example-urdu {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.4rem;
        color: white;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .example-meaning {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.8);
    }
    
    /* Result Display - Added copy button style integration */
    .result-display {
        background: rgba(255,255,255,0.98);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid #3b82f6;
        box-shadow: var(--shadow-soft);
    }
    
    .result-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: var(--text-primary);
        line-height: 1.7;
        margin: 0;
    }
    
    /* Footer - Modernized with social links placeholder */
    .footer-section {
        background: rgba(0,0,0,0.4);
        backdrop-filter: blur(25px);
        border-top: 1px solid rgba(255,255,255,0.15);
        padding: 3rem 0;
        margin-top: 5rem;
        text-align: center;
    }
    
    .developer-info {
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .developer-name {
        color: #3b82f6;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .tech-info {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
    }
    
    /* Responsive Design - Improved breakpoints */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3.5rem;
        }
        .translator-container {
            padding: 2rem;
            margin: 1.5rem;
            border-radius: 24px;
        }
        .translation-area {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        .swap-button {
            display: none;
        }
        .examples-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    </style>
    """, unsafe_allow_html=True)

def create_hero_section():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="developer-badge">
                👨‍💻 Developed by Abdullah Zain & Abdullah With &hearts;
            </div>
            <h1 class="hero-title">UrduScript</h1>
            <p class="hero-subtitle">
                پیشہ ورانہ اردو سے رومن اردو ٹرانسلیٹریٹر<br>
                جدید AI ٹیکنالوجی سے تقویت یافتہ
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_examples_section():
    examples = [
        ("سلام", "Hello/Peace"),
        ("شکریہ", "Thank you"),
        ("کیا حال ہے", "How are things?"),
        ("اللہ حافظ", "Goodbye"),
        ("آپ کیسے ہیں؟", "How are you?")
    ]
    
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h4 style="color: rgba(255,255,255,0.95); font-family: 'Space Grotesk', sans-serif; 
                   text-align: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
            ✨ ان مثالوں کو آزمائیں
        </h4>
        <div class="examples-grid">
    """, unsafe_allow_html=True)
    
    selected_example = None
    cols = st.columns(len(examples))
    
    for i, (urdu, meaning) in enumerate(examples):
        with cols[i]:
            st.markdown(f"""
            <div class="example-card" onclick="document.getElementById('example_{i}').click()">
                <div class="example-urdu">{urdu}</div>
                <div class="example-meaning">{meaning}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("", key=f"example_{i}", help=f"Click to use: {urdu}"):
                selected_example = urdu
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    return selected_example

def create_translator_interface(model, sp_ur, sp_en):
    st.markdown("""
    <div class="translator-container">
        <div class="translator-header">
            <h2 class="translator-title">AI سے چلنے والا ٹرانسلیٹریٹر</h2>
            <p class="translator-desc">نیچے اپنا اردو متن درج کریں اور فوری رومن اردو ٹرانسلیٹریشن حاصل کریں</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Examples section
    selected_example = create_examples_section()
    
    # Input section
    st.markdown("""
    <div class="input-section">
        <div class="section-label">🔤 اردو متن ان پٹ</div>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Urdu Text Input",
        value=selected_example if selected_example else "",
        height=150,
        placeholder="یہاں اردو متن لکھیں یا پیسٹ کریں...",
        key="urdu_input",
        label_visibility="collapsed"
    )
    
    # Translate button with better styling
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        translate_clicked = st.button(
            "🚀 رومن اردو میں تبدیل کریں",
            key="translate_btn",
            use_container_width=True
        )
    
    # Translation logic and results
    if translate_clicked and user_input.strip():
        with st.spinner('🔄 ٹرانسلیٹریشن پراسیس ہو رہی ہے...'):
            try:
                translation = translate_sentence(user_input.strip(), model, sp_ur, sp_en, DEVICE)
                
                st.markdown(f"""
                <div class="result-display">
                    <div class="section-label">📝 رومن اردو آؤٹ پٹ</div>
                    <p class="result-text">{translation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Success message
                st.success("✅ ٹرانسلیٹریشن کامیابی سے مکمل ہوئی!")
                
            except Exception as e:
                st.error(f"❌ ٹرانسلیٹریشن ناکام: {str(e)}")
                st.info("💡 مختلف متن کے ساتھ آزمائیں یا اپنا ان پٹ چیک کریں۔")
    
    elif translate_clicked and not user_input.strip():
        st.warning("⚠️ براہ مہربانی ٹرانسلیٹ کرنے کے لیے کچھ اردو متن درج کریں۔")
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_footer():
    st.markdown("""
    <div class="footer-section">
        <div class="developer-info">
            ❤️ سے تیار کردہ <span class="developer-name">عبداللہ زین</span> کی جانب سے
        </div>
        <div class="tech-info">
            Streamlit • PyTorch • Neural Machine Translation سے تیار کردہ
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="UrduScript - پیشہ ورانہ اردو ٹرانسلیٹریٹر از عبداللہ زین",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS
    add_custom_css()
    
    # Create hero section
    create_hero_section()
    
    # Load models with elegant loading
    with st.spinner('🤖 AI ٹرانسلیٹریشن ماڈل لوڈ ہو رہے ہیں...'):
        model, sp_ur, sp_en = load_models()
    
    # Main translator interface
    create_translator_interface(model, sp_ur, sp_en)
    
    # Minimal about section (only relevant info)
    with st.expander("ℹ️ اس ٹول کے بارے میں", expanded=False):
        st.markdown("""
        **UrduScript** ایک پیشہ ورانہ درجے کا AI ٹرانسلیٹریٹر ہے جو **عبداللہ زین** نے تیار کیا ہے۔
        
        **اہم خصوصیات:**
        - 🧠 **نیورل نیٹ ورک آرکیٹیکچر**: سیکوئنس ٹو سیکوئنس اٹینشن میکانزم کے ساتھ
        - ⚡ **اعلیٰ درستگی**: وسیع اردو-رومن کارپس پر تربیت یافتہ
        - 🔄 **ریئل ٹائم پروسیسنگ**: فوری نتائج
        - 📱 **ریسپانسیو ڈیزائن**: تمام آلات پر بغیر کسی رکاوٹ کے کام کرتا ہے
        
        **کے لیے بہترین:**
        - سوشل میڈیا مواد کی تخلیق
        - تعلیمی اور تدریسی مقاصد  
        - کراس پلیٹ فارم کمیونیکیشن
        - مواد کی رسائی
        """)
    
    # Footer with developer attribution
    create_footer()

if __name__ == "__main__":
    main()