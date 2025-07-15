# Project Likhat 📝✨

**Created by:** Krishna Gera & Shivansh Choudhary, two Sophomores at Manipal Institute of Technology  
**Goal:** Convert handwritten text into digital text – built entirely from scratch! 🚀 No open-source libraries used for training or execution.  

Project Likhat is a cutting-edge Computer Vision project designed to recognize and transcribe handwriting. It handles everything from single characters to full words, making it a versatile tool for digitizing notes, sketches, and more.  

## Workflow 🔄  
Our model processes images through a streamlined pipeline:  
1. **Input:** Start with a PNG image.  
2. **Conversion:** Resize to 100x100 and convert to PPM binary format.  
3. **Matrix Extraction:** Transform the PPM into an RGB matrix (100x100x3).  
4. **Processing:** Run recognition tasks on this matrix to identify characters.  

We focus on recognizing **62 characters**:  
- Uppercase: A-Z 🔠  
- Lowercase: a-z 🔡  
- Digits: 0-9 🔢  

## Project Variants 🛠️  
By completion, Project Likhat will feature **4 powerful variants**:  

### 1. Binary Classification Model (Extrapolated to 62 Characters) ⚖️  
This foundational model trains separately for each character as a "Character" vs. "Not Character" binary classifier. We extrapolate it across all 62 characters using a switch-case logic.  

- **Input Process:** PNG → PPM (100x100) → RGB Matrix (100x100x3).  
- **Output:** Predicts if the image matches one of A-Z, a-z, 0-9, or none.  

**Architecture:**  
- **Layer 1:** Convolutional Layer – 32 Filters (3x3), Padding: 1, Stride: 1, Activation: ReLU 🔄  
- **Layer 2:** Max Pool Layer – 2x2, No Padding, Stride: 2 📉  
- **Layer 3:** Convolutional Layer – 64 Filters (3x3), Padding: 1, Stride: 1, Activation: ReLU 🔄  
- **Layer 4:** Max Pool Layer – 2x2, No Padding, Stride: 2 📉  
- **Layer 5:** Convolutional Layer – 128 Filters (3x3), Padding: 1, Stride: 1, Activation: ReLU 🔄  
- **Layer 6:** Flattening 🏗️  
- **Layer 7:** Dense Layer – 128 Units, ReLU Activation 🧠  
- **Layer 8:** Dropout – 0.5 🚫  
- **Layer 9:** Dense Layer – 1 Unit, Sigmoid Activation 📊  

### 2. Multi-Classification Model 🌟  
A single, unified model for classifying across all 62 characters – no need for multiple binaries!  

- **Input Process:** PNG → PPM (100x100) → RGB Matrix (100x100x3).  
- **Output:** Directly identifies the character class.  
- **Advantages:** Smoother, more accurate, and efficient than repeated binary checks. 💡  

### 3. Full Word Model 📖  
Go beyond single characters! This variant handles images with multiple characters (words or sentences).  

- **How it Works:** Breaks down the input into individual characters, runs recognition on each, and reconstructs the full text.  
- **Input:** Multi-character PNG images.  
- **Output:** The complete transcribed text.  

### 4. GUI for Everything 🖥️  
A user-friendly web application to showcase it all!  

- **Features:** Upload your handwriting image, get outputs from all models, and compare results.  