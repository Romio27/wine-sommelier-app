## LLM Engineering Course - Practical Assignments

This repository contains practical assignments for the LLM Engineering course, including:
1. Basic LLM calling with LangChain (`prompting.ipynb`)
2. Retrieval-Augmented Generation (RAG) system (`rag.ipynb`)

---

## 🎯 Home Assignment: Wine Recommendation RAG System

### Quick Start

1. **Download Dataset**
   - Visit: https://www.kaggle.com/datasets/zynicide/wine-reviews
   - Download `winemag-data-130k-v2.csv`
   - Place in `data/` directory

2. **Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   - Create `.env` file with:
     ```env
     OPENAI_API_KEY=your_key_here
     AZURE_OPENAI_ENDPOINT=your_endpoint_here
     EMBEDDING_DEPLOYMENT_NAME=your_deployment_name
     ```

4. **Run Notebook**
   ```bash
   jupyter notebook rag.ipynb
   ```

### 📚 Documentation

- **[WINE_RAG_README.md](WINE_RAG_README.md)** - Complete setup and usage guide
- **[DATASET_DOWNLOAD_INSTRUCTIONS.md](DATASET_DOWNLOAD_INSTRUCTIONS.md)** - Dataset download help
- **[HOME_ASSIGNMENT_SUMMARY.md](HOME_ASSIGNMENT_SUMMARY.md)** - Implementation summary

---

## Basic Installation (for all practices)

### First Time Setup:

```bash
python -m venv venv

# for windows:
.\venv\Scripts\activate
# or for macos/linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name=llm_course_kernel --display-name "LLM Course Kernel"

jupyter notebook
```

### Update Environment (for second practice):

```bash
## use existing venv
# for windows:
.\venv\Scripts\activate
# or for macos/linux:
source venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name=llm_course_kernel --display-name "LLM Course Kernel"

jupyter notebook
```

---

## Project Structure

```
sswu-llm-engineering/
├── rag.ipynb                              # Practice 2: RAG implementation (HOME ASSIGNMENT)
├── prompting.ipynb                        # Practice 1: Basic prompting
├── WINE_RAG_README.md                     # Detailed RAG documentation
├── DATASET_DOWNLOAD_INSTRUCTIONS.md       # Dataset download guide
├── HOME_ASSIGNMENT_SUMMARY.md             # Assignment summary
├── requirements.txt                       # Python dependencies
├── .env                                   # API keys (not in git)
├── data/
│   └── winemag-data-130k-v2.csv          # Wine dataset (download required)
└── faiss_index_wine/                      # Saved vector store
```

---

## Requirements

- Python 3.9+
- Jupyter Notebook
- Azure OpenAI API access
- ~2GB free disk space (for dataset and vectors)

---

## Assignment Status

✅ **HOME ASSIGNMENT COMPLETE**
- Dataset: Wine Reviews (130k reviews from Kaggle)
- Implementation: Full RAG pipeline with FAISS vector store
- Testing: Multiple query types tested
- Documentation: Complete setup and usage guides
- Token Budget: ~400k tokens (well under 5M limit)

---

## Contact & Support

For questions or issues:
- Check documentation files first
- Review troubleshooting sections in READMEs
- Contact course instructor

**Submission Date**: November 13, 2024  
**Deadline**: November 17, 2024, 23:59