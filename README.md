# LLM-Based-Custom_Chatbot-
CPU-optimized chatbot using LLama 3 with RAG and fine-tuned Alpaca models. Built on Intel Extension for Transformers, with Streamlit UI, PDF upload, and contextual domain-specific responses.

This project implements a CPU-efficient chatbot system leveraging the power of **LLama 3**, deployed with two state-of-the-art approaches: **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning using the Alpaca dataset**. The solution was built as part of the **Intel Unnati Industrial Training Program** and is optimized using **Intel Extension for Transformers** to enable efficient performance on Intel CPU architectures without the need for GPUs.

It supports **PDF ingestion**, **contextual question answering**, and real-time conversational responses through an intuitive **Streamlit interface**, designed to support research and domain-specific querying.

 
## 🛠️ Tech Stack

- **LLama 3** (Meta AI)
- **FAISS** for vector indexing
- **Intel Extension for Transformers** (CPU acceleration)
- **Streamlit** (UI)
- **PyTorch**, **Langchain**, **Hugging Face Transformers**


## Installation

1. Clone the repository:

```bash
git clone https://github.com/AestheticCoder-rjp/Intel_Unnati_Program
```

2. Install the required dependencies:
   

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your GROQ API key:

```bash
GROQ_API_KEY=your_groq_api_key
```

**Running the Application**

To start the Streamlit application, run:

```bash
streamlit run app.py
```
**Usage**

- **Upload a PDF:** Use the sidebar to upload your PDF document and click "Process".

- **Ask Questions:** Enter your questions in the input box and receive responses based on the PDF content.

- **View Processing Time:** The application displays the processing time for each query.

## 🚀 Features

- 💬 Dual Chatbot Architectures:
  - **RAG-based chatbot** that dynamically retrieves information from uploaded documents using FAISS indexing.
  - **Fine-Tuned Llama 3 model** (using Alpaca dataset) to understand and respond to user queries within a focused domain.

- ⚙️ **Intel CPU Optimization**:
  - Utilizes Intel Extension for Transformers to accelerate inference on Intel hardware.
  - Benchmarking done across systems with various CPU configurations (Core i3, i5, i7, i9).

- 📄 **Document-Aware Question Answering**:
  - Upload and process PDF documents
  - Ask domain-specific questions and receive accurate, contextual answers

- 🧪 **Performance Evaluation**:
  - Metrics evaluated: **Response Time**, **Perplexity**, and **Loss**
  - Graphical visualizations to compare hardware efficiency

- 🌐 **Streamlit UI**:
  - Clean, minimal front-end for document upload and chatbot interaction
  - Visual response display and history tracking
 



