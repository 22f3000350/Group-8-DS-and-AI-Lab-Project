# Crop Disease AI Assistant

## User Guide

---

**Application URL:** [https://huggingface.co/spaces/harishsahadev/crop-disease-assistant](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)

**Version:** 1.0
**Last updated:** April 2026

---

## 1. Overview

The Crop Disease AI Assistant is a free, web-based application designed to help farmers and agricultural field workers identify crop diseases from a photograph of a leaf and receive reliable, source-verified treatment guidance. The system supports seven languages, accepts both voice and text input, and reads responses aloud for users who prefer audio guidance.

The application currently supports five crops: **Corn, Potato, Rice, Wheat, and Sugarcane**, and can identify 17 conditions including common diseases and healthy states.

This guide explains how to use the application from start to finish. No technical knowledge, app installation, or account registration is required.

---

## 2. Before You Begin

### 2.1 What you need

- A smartphone, tablet, or computer with a web browser (Chrome, Safari, Edge, or Firefox)
- An active internet connection (basic 4G is sufficient)
- A clear photograph of the affected leaf

### 2.2 What you do not need

- No account or sign-up
- No payment or subscription
- No app installation

---

## 3. Accessing the Application

Open your web browser and navigate to:

**[https://huggingface.co/spaces/harishsahadev/crop-disease-assistant](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)**

On your first visit, a welcome dialog will appear with a brief overview of the application. Review the information and select **"Get Started"** to close the dialog. The welcome dialog can be reopened at any time by selecting the **information icon** located in the application header.

---

## 4. Application Layout

The application interface is divided into two sections:

| Section | Purpose |
|---------|---------|
| **Left panel** | Input area — upload image, choose language, ask a question |
| **Right panel** | Output area — disease detection, advisory text, audio playback |

On mobile devices, the right panel appears below the left panel when you scroll down.

---

## 5. Using the Application

### Step 1: Upload a Leaf Image

Locate the **"Leaf image"** field in the left panel.

- **On mobile devices:** Tap the field to open your camera or photo gallery.
- **On desktop computers:** Click the field to browse and select an image file.

**Recommended image characteristics:**

- Single leaf clearly visible and filling most of the frame
- Taken in natural daylight, avoiding harsh shadows and direct glare
- In sharp focus, with affected areas clearly visible
- Free of water droplets, dust, or obstructions

**Images to avoid:**

- Photographs taken in low-light conditions
- Multiple overlapping leaves or cluttered backgrounds
- Blurred or out-of-focus images
- Images showing objects unrelated to the leaf

### Step 2: Select a Response Language

Locate the **"Response language"** dropdown below the image field and select your preferred language. The following languages are supported:

| Language | Script |
|----------|--------|
| English | English |
| Hindi | Devanagari |
| Bengali | Bengali |
| Tamil | Tamil |
| Telugu | Telugu |
| Malayalam | Malayalam |
| Kannada | Kannada |

Both the written advisory and the audio response will be provided in the selected language.

**Note:** If you submit a voice question, the application will automatically detect the spoken language and respond accordingly, overriding the dropdown selection.

### Step 3: Submit a Question (Optional)

Under the **"Ask a question"** section, you have three options for asking a question. You may also skip this step entirely — if no question is provided, the application will return a general advisory based on the detected disease.

#### Option A: Voice Input

1. Select the microphone icon in the voice input field.
2. Grant microphone access when prompted by your browser.
3. Speak your question clearly in your preferred language.
4. Select the stop icon when finished.
5. You may review your recording before submission.

Alternatively, you can upload a pre-recorded audio file using the upload option in the same field.

#### Option B: Text Input

Select the text field labelled **"Or type your question here…"** and type your question. Any of the seven supported languages may be used.

#### Option C: Preset Questions

Four preset questions are available as quick-select buttons:

- What treatment should I use?
- Is this disease spreading?
- How to prevent next season?
- Which pesticide is safe?

Select any of these to automatically populate the text field with a common query.

### Step 4: Submit for Analysis

After uploading the image and optionally entering a question, select the **"Analyse"** button.

Processing typically takes **5 to 10 seconds**. During this time, the application performs image analysis, transcribes voice input (if applicable), retrieves relevant information from its knowledge base, generates an advisory, and prepares an audio response.

---

## 6. Understanding the Results

The right panel displays three sections after analysis is complete.

### 6.1 Disease Detection Card

The detection card displays:

- The identified disease name
- The crop in which it was detected
- A **confidence score** indicating how certain the system is about the diagnosis

The confidence score is represented by a colour-coded progress bar. The meaning of each level is described below.

| Confidence Level | Score Range | Interpretation | Recommended Action |
|------------------|-------------|----------------|--------------------|
| **High** | 75% or higher | The system is highly confident in the diagnosis | Advisory may be acted upon with normal diligence |
| **Moderate** | 60% to 74% | The system is reasonably confident | Proceed with caution; consider a second opinion for major decisions |
| **Low** | Below 60% | The system is uncertain | Do not act on the advisory alone; consult a local agricultural expert |

When the confidence level is low, the application displays a warning message recommending consultation with your local Krishi Vigyan Kendra (KVK) or agricultural officer. Please heed this warning and do not apply treatments based on a low-confidence diagnosis.

### 6.2 Expert Advisory

The written advisory appears below the detection card and typically includes:

- A description of the identified disease
- Observable symptoms and how they progress
- Recommended treatment options, including specific practices where applicable
- Preventive measures for future seasons
- A recommendation to consult a local expert when appropriate

All advisory content is grounded in verified documents from the **Indian Council of Agricultural Research (ICAR)**, **Tamil Nadu Agricultural University (TNAU)**, and the **Food and Agriculture Organization (FAO)**. The system will not recommend treatments that are not documented in its trusted sources.

### 6.3 Audio Advisory

An audio player is provided below the advisory text. Select the play control to listen to the advisory in your selected language. The audio is useful for:

- Users who prefer audio over reading
- Listening while working in the field
- Sharing the advisory with others

The audio file may also be downloaded for offline listening or distribution.

---

## 7. Supported Crops and Diseases

The application currently supports 17 conditions across 5 crops:

| Crop | Supported Conditions |
|------|----------------------|
| **Corn** | Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Rice** | Brown Spot, Leaf Blast, Neck Blast, Healthy |
| **Wheat** | Brown Rust, Yellow Rust, Healthy |
| **Sugarcane** | Red Rot, Bacterial Blight, Healthy |

If the leaf is healthy, the application will identify it as such.

---

## 8. Best Practices

To obtain the most accurate results:

1. **Use natural daylight.** Morning or late afternoon light produces the best results. Avoid photographing under direct midday sun or in shade.
2. **Fill the frame with the leaf.** Move close enough that the leaf occupies the majority of the image.
3. **Clean the leaf surface.** Gently wipe away water droplets, dust, or debris before photographing.
4. **Select the most affected leaves.** When multiple leaves show varying levels of infection, photograph the most clearly diseased ones.
5. **Ask specific questions.** Precise questions produce more actionable answers.
6. **Retry low-confidence results.** If the confidence is low, capture additional images of other affected leaves before drawing conclusions.
7. **Verify with a local expert.** Before applying costly or systemic treatments, confirm the diagnosis with your local agricultural officer.

---

## 9. Frequently Asked Questions

**Is the application free to use?**
Yes. The application is provided free of charge to all users.

**Do I need to create an account?**
No account or registration is required.

**Does it work on a basic smartphone?**
Yes. Any smartphone with internet access and a functional camera is sufficient.

**Does it work offline?**
No. An active internet connection is required. A standard mobile data connection is adequate.

**Are my photographs stored or shared?**
Photographs are used only to generate your result within the current session and are not retained.

**What if my crop is not on the supported list?**
The application currently supports only the five listed crops. For unsupported crops, please consult your local agricultural officer. Expanded crop coverage is planned for future versions.

**What if the advisory appears incorrect?**
Agricultural diagnoses can be complex, and no system is infallible. The confidence score is designed to indicate when the system is uncertain. For any significant decision — especially one involving chemical applications or replanting — please verify with a qualified agricultural professional.

**The microphone is not recording. What should I do?**
Ensure you have granted microphone access to your browser when prompted. If issues persist, use the text input option or select a preset question.

**Can I analyse multiple plants at once?**
Each analysis processes one leaf image. For fields with varying symptoms, submit separate analyses for each affected area.

---

## 10. When to Consult an Expert

While the application is designed to provide useful first-level guidance, it is not a substitute for trained agricultural expertise. Please consult your local **Krishi Vigyan Kendra (KVK)** or district agricultural officer in the following situations:

- The confidence score is low, or the application displays a warning message
- The disease is spreading rapidly across your field
- Significant investment in pesticides or fungicides is being considered
- The observed symptoms do not match the advisory description
- The crop is at a critical growth stage and a prompt decision is required

For general agricultural queries in India, you may also contact the **Kisan Call Centre at 1800-180-1551** (toll-free).

---

## 11. Support and Feedback

For issues, suggestions, or feedback regarding the application, please visit the project page and use the Community tab to leave a comment:

[https://huggingface.co/spaces/harishsahadev/crop-disease-assistant](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)

---

## 12. Disclaimer

The Crop Disease AI Assistant is provided as a decision-support tool only. It is not a certified diagnostic system. All treatment recommendations are derived from published agricultural literature and are intended for informational purposes. Users are responsible for verifying recommendations with qualified agricultural professionals before taking action. The developers accept no liability for crop losses, financial damages, or other consequences arising from use of this application.

---

*Crop Disease AI Assistant — Indian Institute of Technology Madras, DSAI Lab, Group 8, 2026*

---

### Team Member Signatures

The following team members have reviwed and approved the contents of report/file:
- [x] Harish Sahadev M
- [x] Sai Naman
- [ ] Ayushi Dixit
- [ ] Allanki Saketh Kumar
- [x] Manas Rastogi
