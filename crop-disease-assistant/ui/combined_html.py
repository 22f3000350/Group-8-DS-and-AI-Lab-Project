COMBINED_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

.gradio-container { 
  font-family: 'DM Sans', 
  sans-serif !important; 
}

#app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;

  padding: 18px 32px;

  background: linear-gradient(90deg, #121211, #1A1A18, #121211);
  
  border: 1px solid rgba(29,158,117,0.4);  
  border-radius: 12px;              

  width: 100%;

  /* subtle glow */
  box-shadow:
    0 0 0 1px rgba(29,158,117,0.15),
    0 10px 30px rgba(0,0,0,0.5);
}
#app-header .logo-text {
  font-size: clamp(16px, 3.5vw, 22px);
  font-weight: 700;
  white-space: normal; 
  word-break: keep-all;
  color: #1D9E75;
  letter-spacing: 0.02em;
  margin-left: 2px;
  line-height: 1.2;
}
#app-header .logo-sub {
  font-size: clamp(11px, 1.1vw, 13px);
  color: #9A9A94;
  line-height: 1.4;
  margin-top: 4px;
  max-width: 100%;
}
#app-header .logo-sub strong {
  color: #E8E8E2;
  font-weight: 500;
}
#app-header .logo-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
#app-header .logo-icon {
  font-size: clamp(22px, 2vw, 28px);  
  line-height: 1;
  display: flex;
  align-items: center;
  margin-right: 2px;
  justify-content: center;
  transform: translateY(2px);
  filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));
}
#app-header .header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
#app-header:hover {
  box-shadow:
    0 0 0 1px rgba(29,158,117,0.3),
    0 12px 40px rgba(0,0,0,0.6);
}

@media (max-width: 600px) {
  #app-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  #app-header .header-actions {
    align-self: flex-end;   
  }
  #app-header .logo-row {
    align-items: flex-start;
  }
}

#info-btn {
  padding: 6px 14px;
  border-radius: 8px;

  background: #232321;
  border: 1px solid #1D9E75;

  color: #C2C0B6;
  font-size: 12px;
  font-weight: 500;

  cursor: pointer;

  display: flex;
  align-items: center;
  gap: 6px;

  transition: all 0.2s ease;
}
#info-btn:hover {
  background: #1D3B2F;
  border-color: #1D9E75;
  color: #2DB88A;

  box-shadow: 0 0 10px rgba(29,158,117,0.25);
}

.panel-label {
  font-size: clamp(12px, 1.4vw, 15px) !important;         
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;

  color: #C2C0B6 !important;          
  margin-bottom: 10px !important;

  display: flex;
  align-items: center;
  gap: 8px;

  border-bottom: 1px solid #2F2F2D;
  padding-bottom: 6px;
}
.panel-label::before {
  content: "";
  display: inline-block;
  width: 3px;
  height: 12px;
  border-radius: 2px;
  background: #1D9E75;  
}

.upload-zone {
  border: 1.5px dashed #CFCFC8 !important; border-radius: 12px !important;
  background: #F5F5F3 !important; transition: border-color 0.15s, background 0.15s;
}
.upload-zone:hover { border-color: #1D9E75 !important; background: #E1F5EE !important; }

#analyse-btn {
  background: #1D9E75 !important; border: none !important;
  border-radius: 10px !important; color: white !important;
  font-size: 15px !important; font-weight: 500 !important;
  height: 48px !important; width: 100% !important;
  letter-spacing: 0.01em; transition: background 0.15s, transform 0.1s;
}
#analyse-btn:hover  { background: #0F6E56 !important; }
#analyse-btn:active { transform: scale(0.98); }

.result-card {
  background: linear-gradient(145deg, #232321, #1A1A18);  /* subtle depth */
  border: 1px solid #2F2F2D;
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 12px;

  box-shadow: 
    0 8px 30px rgba(0,0,0,0.5),
    inset 0 1px 0 rgba(255,255,255,0.03);

  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 
    0 12px 40px rgba(0,0,0,0.6),
    inset 0 1px 0 rgba(255,255,255,0.04);
}
.result-header {
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid #2F2F2D;
  padding-bottom: 10px;
  margin-bottom: 10px;
}
.result-icon {
  width: 40px;
  height: 40px;

  font-size: clamp(20px, 1.8vw, 26px); 

  border-radius: 10px;
  background: #1D3B2F;   

  display: flex;
  align-items: center;
  justify-content: center;

  flex-shrink: 0;

  box-shadow: inset 0 0 6px rgba(0,0,0,0.4);
}
.result-title {
  font-size: 15px;
  font-weight: 600;
  color: #FFFFFF;
}
.result-subtitle {
  font-size: 12px;
  color: #9A9A94;
}
.result-badges {
  margin: 8px 0 10px 0;
}

.conf-bar-wrap {
  margin-top: 10px;
}
.conf-bar-track {
  height: 6px;
  background: #2F2F2D;
  border-radius: 999px;
}
.conf-bar-fill {
  height: 6px;
  border-radius: 999px;

  background: linear-gradient(
    90deg,
    #1D9E75,
    #2DB88A,
    #4BE3B1
  );

  box-shadow: 0 0 10px rgba(45,184,138,0.4);
}
.conf-bar-label {
  font-size: 11px;
  color: #2DB88A;
  margin-top: 6px;
  font-weight: 500;
}

.badge {
  font-size: 10px;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 600;
}
.badge-crop {
  background: rgba(45,184,138,0.15);
  color: #2DB88A;
}
.badge-disease {
  background: rgba(242,176,76,0.15);
  color: #F2B04C;
}
.badge-severity {
  background: rgba(255,90,90,0.18);
  color: #FF7A7A;
}
.badge-warn {
  background: rgba(242,176,76,0.15);
  color: #F2B04C;
  border: 1px solid rgba(242,176,76,0.3);
}

.advisory-box textarea {
  font-size: 13px !important; line-height: 1.75 !important;
  background: #F5F5F3 !important; border: 0.5px solid #E0E0DA !important;
  border-radius: 10px !important; color: #333330 !important;
}

/* Modal — uses normal flow (no position:fixed) so iframe height isn't collapsed */
#modal-backdrop {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 9999;

  background: rgba(0,0,0,0.45);

  justify-content: center;
  align-items: flex-start;  
  padding-top: 80px;         
}

#modal-backdrop.open {
  display: flex;
}
#instruction-modal {
  background: #1E1E1C;  
  color: #E8E8E2;
  border-radius: 16px;
  max-width: 680px; width: 100%; max-height: 90vh; overflow-y: auto;
  padding: 28px; position: relative;
}
#modal-close {
  position: absolute;
  top: 12px;
  right: 12px;

  width: 32px;
  height: 32px;
  padding: 0;

  border-radius: 50%;

  background: #232321;
  border: 1px solid #D94B4B;

  color: #E8E8E2;
  font-size: 14px;
  font-weight: 600;

  display: flex;
  align-items: center;
  justify-content: center;

  cursor: pointer;
  transition: all 0.2s ease;
}
#modal-close:hover {
  background: #3A1D1D;
  border-color: #FF6B6B;

  color: #FF6B6B;

  box-shadow: 0 0 10px rgba(255,107,107,0.3);

  transform: translateY(-1px);
}
.modal-logo-icon {
  font-size: clamp(24px, 2.4vw, 30px);
  line-height: 1;

  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0.9;

  transform: translateY(1px);
  filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));
}
.modal-header-row {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}
.modal-title { font-size: 20px; font-weight: 600; color: #1A1A18; margin-bottom: 4px; }
.modal-sub   { font-size: 13px; color: #80807A; margin-bottom: 20px; }
.modal-section-title {
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: #A0A099; margin: 16px 0 8px;
}
.modal-step { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
.modal-step-num {
  width: 24px; height: 24px; border-radius: 50%;
  background: #1D9E75; color: white; font-size: 11px; font-weight: 600;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.modal-step-text { font-size: 13px; color: #333330; line-height: 1.6; }
.modal-step-text strong { font-weight: 600; color: #1A1A18; }
.modal-crop-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 8px; margin-top: 8px;
}
.modal-crop-card {
  background: #2A2A28; 
  border: 1px solid #3A3A38;
  border-radius: 10px; padding: 10px 12px; font-size: 12px; color: #A0A099; font-weight: 500;
}
.modal-crop-card span { display: block; font-size: 10px; color: #80807A; margin-top: 2px; }
.modal-example-chip {
  display: inline-block;
  background: #2A2A28;                
  border: 1px solid #4A4A47;
  border-radius: 20px;
  padding: 5px 12px;
  font-size: 12px;
  color: #E0E0DA;                   
  margin: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.modal-example-chip:hover {
  border-color: #1D9E75;
  background: #1D3B2F;              
  color: #2DB88A;
}
.modal-example-chip.active {
  background: #1D3B2F;
  border-color: #1D9E75;
  color: #2DB88A;

  box-shadow: 0 0 8px rgba(29,158,117,0.25);
}

.modal-tip {
  background: #163D32;              
  border: 1px solid #1D9E75;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 12px;
  color: #BFEBDD;                   
  margin-top: 14px;
  line-height: 1.6;
}

.modal-cta {
  width: 100%;
  background: #1D9E75;
  color: #FFFFFF;
  border: none;
  border-radius: 10px;
  padding: 13px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 20px;
  transition: all 0.15s ease;

  /* subtle elevation */
  box-shadow: 0 4px 14px rgba(29,158,117,0.25);
}
.modal-cta:hover {
  background: #0F6E56;
  box-shadow: 0 6px 18px rgba(29,158,117,0.35);
}
</style>

<!-- ── Modal backdrop ──────────────────────────────────────────────────── -->
<div id="modal-backdrop">
  <div id="instruction-modal" role="dialog" aria-modal="true">
    <button id="modal-close" aria-label="Close">x</button>

    <div class="modal-header-row">
      <div class="modal-logo-icon">🌽</div>
      <div>
        <div class="modal-title">Crop Disease AI Assistant</div>
        <div class="modal-sub">Detect crop diseases and get expert advice in your language</div>
      </div>
    </div>

    <div class="modal-section-title">How to use</div>
    <div class="modal-step">
      <div class="modal-step-num">1</div>
      <div class="modal-step-text"><strong>Upload a leaf image</strong> — take a clear close-up photo of the affected leaf. Good lighting and focus improve accuracy.</div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">2</div>
      <div class="modal-step-text"><strong>Choose your language</strong> — select from the dropdown so responses are in your preferred language.</div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">3</div>
      <div class="modal-step-text"><strong>Ask a question</strong> — record your voice or type a question. You can ask about treatment, prevention, or spreading of the disease.</div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">4</div>
      <div class="modal-step-text"><strong>Click Analyse</strong> — the AI detects the disease, generates an advisory, and reads it out loud for you.</div>
    </div>

    <div class="modal-section-title">Supported crops</div>
    <div class="modal-crop-grid">
      <div class="modal-crop-card">Corn <span>Maize</span></div>
      <div class="modal-crop-card">Potato<span>All varieties</span></div>
      <div class="modal-crop-card">Rice<span>Paddy</span></div>
      <div class="modal-crop-card">Wheat<span>All varieties</span></div>
      <div class="modal-crop-card">Sugarcane<span>Ganna</span></div>
    </div>

    <div class="modal-section-title">Try these example questions</div>
    <div>
      <span class="modal-example-chip" data-question="What treatment should I use for this disease?">What treatment should I use?</span>
      <span class="modal-example-chip" data-question="Is this disease spreading to nearby plants?">Is this disease spreading?</span>
      <span class="modal-example-chip" data-question="How can I prevent this next season?">How to prevent next season?</span>
      <span class="modal-example-chip" data-question="Which pesticide is safe and affordable?">Which pesticide is safe?</span>
      <span class="modal-example-chip" data-question="How urgent is the treatment?">How urgent is treatment?</span>
      <span class="modal-example-chip" data-question="What is the name of this disease in Hindi?">Name in Hindi?</span>
    </div>

    <div class="modal-tip">
      <strong>Tip:</strong> For best results, photograph the leaf against a plain background in natural daylight.
      The AI works best with images showing clear disease symptoms such as spots, yellowing, or lesions.
    </div>
    <br>
    <button class="modal-cta" id="modal-cta-btn">Get started</button>
  </div>
</div>

<!-- ── Header ──────────────────────────────────────────────────────────── -->
<div id="app-header">
  <div class="logo-row">
    <div class="logo-icon">
      🌾
    </div>
    <div>
      <div class="logo-text">Crop Disease AI Assistant</div>
      <div class="logo-sub">
        <strong>Smart Agriculture for Indian Farmers: </strong>
        Upload a leaf image and ask your question by voice or text.
      </div>
    </div>
  </div>
  <div class="header-actions">
    <button id="info-btn" title="info">Instructions ℹ️</button>
  </div>
</div>
"""
