# Natural Language Processing Project: EmPULIA Q&A System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A sophisticated Question-Answering system that combines **Knowledge Graphs (KG)** and **Large Language Models (LLMs)** to provide intelligent assistance for users of the EmPULIA e-procurement platform. The system processes technical documentation and regulatory manuals to answer complex queries about procedures, requirements, and platform functionality.

## 🎯 Project Purpose

This project addresses the challenge of making complex e-procurement documentation accessible and queryable through natural language. Users can ask questions about EmPULIA platform procedures, requirements, and regulations, receiving accurate, contextual answers with source citations.

### Key Objectives:
- **Automated Knowledge Extraction**: Extract structured knowledge from PDF manuals using LLMs
- **Intelligent Query Processing**: Understand user intent and retrieve relevant information
- **Hybrid Retrieval**: Combine Knowledge Graph traversal with semantic text search
- **Source Attribution**: Provide verifiable citations for all generated answers
- **Scalable Architecture**: Support multiple data sources and evaluation frameworks

## 🏗️ System Architecture

The system follows a modular **RAG (Retrieval-Augmented Generation)** architecture enhanced with Knowledge Graph reasoning:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Documents │    │  Knowledge Graph │    │   Text Chunks   │
│    (Manuals)    │────│   (Neo4j DB)    │────│   (Processed)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Q&A PIPELINE                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Query Analysis │ Knowledge       │ Answer Generation           │
│  (Intent & NER) │ Retrieval       │ (LLM Synthesis)             │
│                 │ (KG + Vector)   │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI       │    │   Streamlit     │
│   (Streamlit)   │    │   Backend       │    │   Frontend      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components:

1. **Knowledge Graph Creation** (`src/KnowledgeGraphCreation/`)
   - Extraction usnig Gemini
   - Entity and relation extraction from PDFs
   - Clustering and normalization pipelines
   
2. **Query Processing** (`src/query_analyzer*.py`)
   - Intent classification and entity recognition
   - Query expansion and term extraction
   
3. **Knowledge Retrieval** (`src/knowledge_retriever*.py`)
   - Hybrid search (keyword + semantic)
   - Graph traversal and path exploration
   - Context ranking and selection
   
4. **Answer Generation** (`src/answer_generator.py`)
   - LLM-based synthesis with context grounding
   - Source citation and attribution
   - Response validation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud)
- Google Gemini API key (or other LLM provider)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MicheleMinervini06/Natural-Language-Processing-Project.git
cd Natural-Language-Processing-Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_gemini_api_key_here"
```

4. **Set up Neo4j database:**
   - Install Neo4j locally or use Neo4j Aura (cloud)
   - Update connection settings in the configuration files
   - Default credentials: `neo4j:Password` (change in production)

### Basic Usage

#### Option 1: Web Interface (Recommended)

1. **Start the FastAPI backend:**
```bash
python main.py
```

2. **Launch the Streamlit frontend:**
```bash
streamlit run app.py
```

3. **Access the application:**
   - Backend API: http://127.0.0.1:8000
   - Frontend UI: http://localhost:8501
   - API Documentation: http://127.0.0.1:8000/docs

#### Option 2: Direct API Usage

```python
from src.answer_generator import run_qa_pipeline

# Ask a question
result = run_qa_pipeline(
    "Come posso creare una commissione di gara?",
    use_raw_data=True
)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['contexts']}")
```

#### Option 3: Command Line Interface

```bash
python src/answer_generator.py
```

## 📊 Knowledge Graph Creation

The system supports multiple knowledge extraction approaches:

### Method 1: Standard LLM Extraction (Gemini)
```bash
cd src/KnowledgeGraphCreation
python build_KG.py
# Select option 1 for standard processing
```

### Method 2: KGGen Library
```bash
python build_KG_KGgen.py
```

### Method 3: GraphRAG Approach
```bash
python build_KG_graphrag.py
```

### Data Processing Pipeline

1. **PDF Preprocessing**: Extract and chunk text content
2. **Entity Extraction**: Identify domain-specific entities
3. **Relation Extraction**: Extract relationships between entities
4. **Aggregation**: Merge similar entities and relations
5. **Clustering**: Group semantically similar concepts
6. **Neo4j Import**: Store structured knowledge graph

## 🎯 Supported Query Types

The system handles various question categories:

- **Procedural**: "How do I submit a tender?"
- **Requirements**: "What documents are needed for registration?"
- **Troubleshooting**: "What should I do if my digital signature fails?"
- **Definitions**: "What is a DGUE document?"
- **Status Inquiries**: "How can I check my application status?"

## 📈 Evaluation Framework

The project includes comprehensive evaluation tools:

### Metrics Supported:
- **Faithfulness**: Answer accuracy based on retrieved context
- **Answer Relevancy**: Relevance of answers to questions
- **Context Precision**: Quality of retrieved context
- **Context Recall**: Completeness of retrieved information

### Running Evaluations:
```bash
cd src/evaluation
python analyze_evaluation_run.py
```

### Golden Dataset:
- Located in `data/golden_dataset.json`
- Contains validated question-answer pairs
- Supports difficulty levels and question categorization

## 🛠️ Configuration

### Key Configuration Files:

- `requirements.txt`: Python dependencies
- `GEMINI_SETUP.md`: LLM API configuration guide
- Neo4j connection settings in retriever classes

### Environment Variables:
```bash
GEMINI_API_KEY=your_api_key
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## 📁 Project Structure

```
├── src/
│   ├── KnowledgeGraphCreation/     # KG extraction pipelines
│   ├── evaluation/                 # Evaluation tools and metrics
│   ├── utils/                      # Utility functions
│   ├── answer_generator.py         # Main Q&A pipeline
│   ├── query_analyzer*.py          # Query processing modules
│   ├── knowledge_retriever*.py     # Knowledge retrieval modules
│   └── pdf_preprocessing.py        # Document processing
├── data/
│   ├── pdfs/                       # Source PDF documents
│   ├── processed/                  # Processed text chunks
│   ├── evaluation_results/         # Evaluation outputs
│   └── golden_dataset.json         # Evaluation dataset
├── static/                         # Web interface assets
├── logs/                           # Application logs
├── notes/                          # Development notes
├── main.py                         # FastAPI backend
├── app.py                          # Streamlit frontend
└── requirements.txt                # Dependencies
```

## 🔧 Advanced Usage

### Custom Entity Types

Define domain-specific entities in `src/KnowledgeGraphCreation/build_KG.py`:

```python
ENTITY_TYPES = [
    "PiattaformaModulo",
    "FunzionalitàPiattaforma", 
    "RuoloUtente",
    "DocumentoSistema",
    # Add your custom types
]
```

### Custom Relations

```python
RELATION_TYPES = [
    "richiede",
    "genera", 
    "utilizza",
    "gestisce",
    # Add your custom relations
]
```

### Data Source Configuration

Support for multiple data modes:
- `use_raw_data=True`: Uses raw extracted knowledge
- `use_raw_data=False`: Uses aggregated/clustered knowledge

## 🧪 Testing

Run the test suite:
```bash
cd tests
python -m pytest
```

Test individual components:
```bash
python src/answer_generator.py  # Test Q&A pipeline
python src/query_analyzer_rawData.py  # Test query analysis
```

## 📊 Performance Monitoring

Monitor system performance through:
- FastAPI automatic metrics at `/metrics`
- Streamlit interface usage analytics
- Neo4j query performance logs
- LLM API usage tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues:

1. **Neo4j Connection Errors**:
   - Verify Neo4j is running
   - Check connection credentials
   - Ensure database exists

2. **LLM API Errors**:
   - Verify API key is set correctly
   - Check API quota and limits
   - Review rate limiting settings

3. **Memory Issues**:
   - Reduce batch sizes in processing
   - Use chunked processing for large documents
   - Monitor system resources

4. **Poor Answer Quality**:
   - Verify knowledge graph completeness
   - Check source document quality
   - Tune retrieval parameters

### Getting Help:

- Check the `logs/` directory for detailed error messages
- Review the `notes/action_plan.md` for development insights
- Open an issue on GitHub for bugs or feature requests

## 📚 References

This project implements concepts from research papers on:
- Knowledge Graph-Enhanced RAG systems
- LLM-based knowledge extraction
- Hybrid retrieval architectures
- Evaluation frameworks for Q&A systems

## 🔄 Recent Updates

- Added support for multiple KG extraction methods
- Implemented hybrid retrieval with context reranking  
- Enhanced evaluation framework with multiple metrics
- Improved web interface with source linking
- Added comprehensive error handling and logging
