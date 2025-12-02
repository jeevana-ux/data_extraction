# PDF Extraction & Scheme Header Generation

A modular, object-oriented system for extracting text and tables from PDF files and generating Retailer Hub scheme headers using LLM-based extraction.

## 🏗️ Architecture

This project uses a clean, modular architecture with the following components:

```
src/
├── config.py              # Configuration management (Pydantic)
├── models.py              # Data models (Pydantic)
├── main.py                # CLI interface (Click)
│
├── extractors/            # PDF extraction strategies
│   ├── base.py           # Abstract base classes
│   ├── text_extractors.py    # PDFPlumber, OCR
│   ├── table_extractors.py   # PDFPlumber, Camelot, OCR
│   └── pdf_processor.py      # Main orchestrator
│
├── cleaners/              # Content cleaning
│   ├── text_cleaners.py      # Disclaimer, email header filters
│   └── table_cleaners.py     # Table cleaning logic
│
├── llm/                   # LLM integration
│   ├── llm_client.py         # OpenRouter client (DSPy compatible)
│   └── dspy_modules.py       # DSPy scheme extractor
│
└── pipeline/              # Pipeline orchestration
    ├── extraction_pipeline.py  # Main pipeline
    └── output_manager.py       # Output management
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### Usage

#### Option 1: Modern CLI (Recommended)

```bash
# Extract PDFs only
python -m src.main extract input/*.pdf

# Build scheme headers from extracted output
python -m src.main build-headers

# Run full pipeline (extract + build headers)
python -m src.main run-full input/*.pdf

# Show configuration
python -m src.main info
```

#### Option 2: Legacy Script (Backward Compatible)

```bash
# Build scheme headers from output directory
python build_scheme_header.py
```

## 📋 Configuration

Configuration is managed via `.env` file and `src/config.py`:

```env
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional (with defaults)
OPENROUTER_MODEL=qwen/qwen3-next-80b-a3b-instruct
OCR_ENABLED=true
CAMELOT_ENABLED=true
OCR_DPI=200
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4000
```

## 🔧 Key Features

### Modular Extraction
- **Strategy Pattern**: Multiple extraction strategies (PDFPlumber, Camelot, OCR)
- **Automatic Fallback**: OCR fallback for image-based PDFs
- **Smart Deduplication**: Merges results from multiple extractors

### Content Cleaning
- **Disclaimer Removal**: Filters email disclaimers and caution notices
- **Email Header Filtering**: Removes headers while preserving subjects
- **Table Cleaning**: Removes empty rows and disclaimer content

### LLM Integration
- **DSPy Framework**: Structured LLM programming
- **Type-Safe Models**: Pydantic validation for all data
- **Usage Tracking**: Token usage statistics

### Pipeline Orchestration
- **Error Handling**: Graceful degradation and retry logic
- **Comprehensive Logging**: Detailed processing logs
- **Flexible Workflows**: Support for full and incremental pipelines

## 📁 Output Structure

### Extraction Outputs (`output/`)

```
output/
└── <pdf_id>/
    └── <timestamp>/
        ├── <pdf_id>_full_text.txt
        ├── <pdf_id>_page1_table_1.csv
        ├── <pdf_id>_page1_table_2.csv
        └── <pdf_id>_summary.json
```

### Final Outputs (`out/`)

```
out/
└── scheme_header.json
```

## 🧪 Development

### Project Structure

- **`src/`**: Main source code (modular architecture)
- **`input/`**: Input PDF files
- **`output/`**: Extraction results (organized by PDF ID and timestamp)
- **`out/`**: Final scheme headers
- **`logs/`**: Application logs
- **`tests/`**: Test suite (to be implemented)

### Design Patterns

- **Strategy Pattern**: Extraction strategies
- **Factory Pattern**: LLM client creation
- **Pipeline Pattern**: Workflow orchestration
- **Dependency Injection**: Configuration management

### Type Safety

All data models use Pydantic for:
- Runtime validation
- Type checking
- Automatic serialization/deserialization
- Clear API contracts

## 📊 Example Workflow

```python
from src.pipeline import ExtractionPipeline
from pathlib import Path

# Initialize pipeline
pipeline = ExtractionPipeline()

# Process a single PDF
result = pipeline.process_pdf(Path("input/document.pdf"))

# Extract schemes
schemes = pipeline.extract_schemes_from_result(result)

# Or run full pipeline
df = pipeline.run_full_pipeline([Path("input/document.pdf")])
```

## 🔍 Debugging

Enable verbose logging:

```bash
python -m src.main --verbose extract input/*.pdf
```

Check logs:

```bash
tail -f logs/extraction.log
```

## 📝 Migration from Legacy Code

The legacy `build_scheme_header.py` has been refactored to use the new modular architecture while maintaining backward compatibility. All functionality is preserved with improved:

- **Modularity**: Clear separation of concerns
- **Testability**: Easy to unit test components
- **Maintainability**: Clean, documented code
- **Extensibility**: Easy to add new extractors or LLM providers

## 🤝 Contributing

When adding new features:

1. Follow the existing modular structure
2. Use Pydantic models for data validation
3. Add comprehensive logging
4. Update documentation

## 📄 License

Internal use only - Flipkart Retailer Hub