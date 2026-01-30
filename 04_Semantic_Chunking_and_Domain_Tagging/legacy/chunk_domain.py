"""
Semantic Chunking + Domain Classification Pipeline
Fixed with HuggingFace cache best practices to prevent corruption issues
"""

# =============================================================================
# HUGGINGFACE CACHE CONFIGURATION - MUST BE FIRST!
# =============================================================================
import os
import shutil
from pathlib import Path

# NUCLEAR OPTION: Force clean cache before every run
cache_to_clean = Path.home() / ".cache" / "huggingface" / "transformers" / "models--microsoft--deberta-large-mnli"
if cache_to_clean.exists():
    print(f"🧹 Removing potentially corrupted cache: {cache_to_clean}")
    shutil.rmtree(cache_to_clean, ignore_errors=True)


def setup_hf_cache():
    """Setup HuggingFace cache with best practices"""
    # Use user cache directory (most reliable)
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    # Alternative: Use custom location if needed
    # cache_dir = Path("D:/ML_Cache/huggingface")
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables with proper path handling
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir / "sentence_transformers")
    
    # CRITICAL: Disable symlinks on Windows
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    
    print(f"✅ HuggingFace cache configured at: {cache_dir}")
    return cache_dir

def clean_corrupted_cache(cache_dir=None):
    """Clean corrupted cache - use only if issues occur"""
    if cache_dir is None:
        cache_dir = Path(os.environ.get('HF_HOME', Path.home() / ".cache" / "huggingface"))
    
    print(f"🧹 Cleaning cache at: {cache_dir}")
    try:
        transformers_cache = cache_dir / "transformers"
        if transformers_cache.exists():
            shutil.rmtree(transformers_cache)
            print("✅ Transformers cache cleaned")
        transformers_cache.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Error cleaning cache: {e}")

# Initialize cache BEFORE any HuggingFace imports
cache_location = setup_hf_cache()

# =============================================================================
# NOW IMPORT LIBRARIES
# =============================================================================
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict, Tuple
import torch

# ======================== CONFIGURATION ========================
CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "classification_model": "microsoft/deberta-large-mnli",
    "min_chunk_size": 300,  # tokens
    "max_chunk_size": 500,  # tokens
    "similarity_threshold": 0.75,  # for semantic similarity
    "overlap_size": 50,  # tokens overlap between chunks
    "domains": [
        "E-commerce", 
        "Finance", 
        "Healthcare", 
        "Education", 
        "Manufacturing", 
        "Government", 
        "Gaming", 
        "Social Media",
        "Restaurant/Food Service",
        "Location-based Services",
        "Task Management / Productivity Tools"
    ]
}

# ADD THIS: Input paths for CRU units
CRU_INPUT_PATH = "../02_Requirement_Understanding/output/cru_units.json"
REQUIREMENTS_INPUT_PATH = "../02_Requirement_Understanding/output/requirements_extracted_grouped.json"  # Still needed for domain classification

class SemanticChunker:
    """
    Semantic chunking using sentence-transformers/all-mpnet-base-v2
    Creates meaningful 300-500 token chunks based on semantic similarity
    """
    
    def __init__(self, model_name=CONFIG["embedding_model"]):
        print(f"🔄 Loading embedding model: {model_name}")
        # Load with explicit cache directory
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_location / "sentence_transformers")
        )
        self.tokenizer = self.model.tokenizer
        print("✅ Embedding model loaded successfully")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model tokenizer"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_semantic_chunks(
        self, 
        requirements: List[Dict],  # Now accepts CRUs
        min_size: int = CONFIG["min_chunk_size"],
        max_size: int = CONFIG["max_chunk_size"],
        similarity_threshold: float = CONFIG["similarity_threshold"]
    ) -> List[Dict]:
        """
        Create semantically coherent chunks from CRUs or requirements
        
        Args:
            requirements: List of CRU dictionaries or requirement dictionaries
            min_size: Minimum chunk size in tokens
            max_size: Maximum chunk size in tokens
            similarity_threshold: Similarity threshold for combining chunks
            
        Returns:
            List of chunk dictionaries with embeddings and metadata
        """
        print(f"\n🔍 Creating semantic chunks (target: {min_size}-{max_size} tokens)...")
        
        chunks = []
        current_chunk = {
            "requirements": [],
            "text": "",
            "token_count": 0,
            "requirement_ids": [],  # For backward compatibility
            "cru_ids": []  # NEW: Track CRU IDs specifically
        }
        
        for req in requirements:
            # Combine requirement/CRU fields into text
            req_text = self._requirement_to_text(req)
            req_tokens = self.count_tokens(req_text)
            
            # Determine ID to track
            req_id = req.get("cru_id") or req.get("id", "")
            
            # Check if adding this requirement exceeds max size
            if current_chunk["token_count"] + req_tokens > max_size and current_chunk["requirements"]:
                # Calculate embedding for current chunk
                chunk_embedding = self.model.encode(current_chunk["text"])
                current_chunk["embedding"] = chunk_embedding.tolist()
                chunks.append(current_chunk)
                
                # Start new chunk with overlap (include previous requirement)
                prev_req = current_chunk["requirements"][-1]
                prev_text = self._requirement_to_text(prev_req)
                prev_id = prev_req.get("cru_id") or prev_req.get("id", "")
                
                current_chunk = {
                    "requirements": [prev_req],
                    "text": prev_text,
                    "token_count": self.count_tokens(prev_text),
                    "requirement_ids": [prev_id],
                    "cru_ids": [prev_id] if "cru_id" in prev_req else []
                }
            
            # Add requirement/CRU to current chunk
            current_chunk["requirements"].append(req)
            current_chunk["text"] += "\n" + req_text
            current_chunk["token_count"] += req_tokens
            current_chunk["requirement_ids"].append(req_id)
            
            # Track CRU IDs separately if this is a CRU
            if "cru_id" in req:
                current_chunk["cru_ids"].append(req_id)
        
        # Add final chunk
        if current_chunk["requirements"]:
            chunk_embedding = self.model.encode(current_chunk["text"])
            current_chunk["embedding"] = chunk_embedding.tolist()
            chunks.append(current_chunk)
        
        # Add chunk IDs and metadata
        for idx, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"CHUNK_{idx+1:03d}"
            chunk["semantic_score"] = self._calculate_semantic_coherence(chunk["text"])
            
            # Add parent requirement tracking for CRUs
            if chunk["cru_ids"]:
                chunk["parent_requirements"] = list(set([
                    req.get("parent_requirement") 
                    for req in chunk["requirements"] 
                    if "parent_requirement" in req
                ]))
        
        print(f"✅ Created {len(chunks)} semantic chunks")
        self._print_chunk_statistics(chunks)
        
        return chunks
    
    def _requirement_to_text(self, req: Dict) -> str:
        """Convert requirement dict OR CRU to text representation"""
        parts = []
        
        # Check if this is a CRU or a requirement
        if "cru_id" in req:
            # This is a CRU
            if "cru_id" in req:
                parts.append(f"CRU: {req['cru_id']}")
            if "parent_requirement" in req:
                parts.append(f"Parent: {req['parent_requirement']}")
            if "actor" in req and req["actor"]:
                parts.append(f"Actor: {req['actor']}")
            if "action" in req and req["action"]:
                parts.append(f"Action: {req['action']}")
            if "constraint" in req and req["constraint"]:
                parts.append(f"Constraint: {req['constraint']}")
            if "system_response" in req and req["system_response"]:
                parts.append(f"System Response: {req['system_response']}")
            if "preconditions" in req and req["preconditions"]:
                parts.append(f"Preconditions: {', '.join(req['preconditions'])}")
            if "rationale" in req and req["rationale"]:
                parts.append(f"Rationale: {req['rationale']}")
        else:
            # This is a standard requirement (fallback for backward compatibility)
            if "id" in req:
                parts.append(f"ID: {req['id']}")
            if "title" in req:
                parts.append(f"Title: {req['title']}")
            if "description" in req:
                parts.append(f"Description: {req['description']}")
            if "rationale" in req:
                parts.append(f"Rationale: {req['rationale']}")
            if "dependencies" in req and req["dependencies"]:
                parts.append(f"Dependencies: {', '.join(req['dependencies'])}")
            
            # Handle use cases (Gherkin format)
            if "feature" in req:
                parts.append(f"Feature: {req['feature']}")
                if "business_value" in req:
                    parts.append(f"Business Value: {req['business_value']}")
                if "actor" in req:
                    parts.append(f"Actor: {req['actor']}")
                if "scenarios" in req:
                    for scenario in req["scenarios"]:
                        parts.append(f"Scenario: {scenario.get('name', '')}")
                        if "steps" in scenario:
                            for step in scenario["steps"]:
                                parts.append(f"{step.get('keyword', '')}: {step.get('text', '')}")
            
            # Handle quality requirements
            if "tag" in req:
                parts.append(f"Tag: {req['tag']}")
            if "gist" in req:
                parts.append(f"Gist: {req['gist']}")
            if "must" in req:
                parts.append(f"MUST: {req['must']}")
        
        return " ".join(parts)
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence score for a chunk"""
        sentences = self.split_into_sentences(text)
        if len(sentences) < 2:
            return 1.0
        
        # Encode sentences
        embeddings = self.model.encode(sentences)
        
        # Calculate pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _print_chunk_statistics(self, chunks: List[Dict]):
        """Print statistics about created chunks"""
        token_counts = [c["token_count"] for c in chunks]
        req_counts = [len(c["requirements"]) for c in chunks]
        coherence_scores = [c["semantic_score"] for c in chunks]
        
        print(f"  📊 Token distribution: min={min(token_counts)}, max={max(token_counts)}, avg={np.mean(token_counts):.1f}")
        print(f"  📊 Requirements per chunk: min={min(req_counts)}, max={max(req_counts)}, avg={np.mean(req_counts):.1f}")
        print(f"  📊 Semantic coherence: min={min(coherence_scores):.3f}, max={max(coherence_scores):.3f}, avg={np.mean(coherence_scores):.3f}")


class DomainClassifier:
    """
    Zero-shot domain classification using microsoft/deberta-large-mnli
    Identifies project domain for context-aware test generation
    """
    
    def __init__(self, model_name=CONFIG["classification_model"]):
        print(f"\n🔄 Loading domain classifier: {model_name}")
        
        # More robust pipeline initialization with retry logic
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                tokenizer=model_name,  # Explicitly specify tokenizer
                use_fast=True,
                revision="main"  # Use main branch to avoid corrupted refs
            )
        except Exception as e:
            print(f"⚠️ First attempt failed: {e}")
            print("🔄 Cleaning cache and retrying...")
            clean_corrupted_cache(cache_location)
            
            # Retry after cleaning
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                tokenizer=model_name,
                use_fast=True,
                local_files_only=False
            )
        
        self.domains = CONFIG["domains"]
        print("✅ Domain classifier loaded successfully")
    
    def _detect_domain_hints(self, requirements: List[Dict]) -> str:
        combined_text = " ".join([
            self._requirement_to_text_simple(req) 
            for req in requirements[:20]
        ]).lower()
        
        # Pattern matching for better domain detection
        if "todo" in combined_text and "task" in combined_text:
            return "Task Management / Productivity Tools"
        elif "payment" in combined_text or "checkout" in combined_text:
            return "E-commerce"
        elif "patient" in combined_text or "medical" in combined_text:
            return "Healthcare"
        elif "transaction" in combined_text or "account" in combined_text:
            return "Finance"
        
        return None
    

    def classify_domain(
        self, 
        requirements: List[Dict], 
        multi_label: bool = True
    ) -> Dict:
        """
        Classify project domain from requirements
        
        Args:
            requirements: List of requirement dictionaries
            multi_label: Whether to allow multiple domain classifications
            
        Returns:
            Dictionary with domain classification results
        """
        print(f"\n🔍 Classifying project domain...")
        
        # ADD THIS: Check for domain hints first
        hint = self._detect_domain_hints(requirements)
        if hint:
            print(f"💡 Domain hint detected: {hint}")

        # Combine first 10 requirements for classification
        sample_reqs = requirements[:10]
        combined_text = " ".join([
            self._requirement_to_text_simple(req) 
            for req in sample_reqs
        ])
        
        # Truncate if too long (BART limit is 1024 tokens)
        if len(combined_text) > 5000:
            combined_text = combined_text[:5000]
        
        # Perform classification
        result = self.classifier(
            combined_text, 
            self.domains,
            multi_label=multi_label,
            hypothesis_template="This project is related to {}."  # ADD THIS for better classification
        )
        
        if hint and hint in result["labels"]:
            hint_idx = result["labels"].index(hint)
            result["scores"][hint_idx] = min(result["scores"][hint_idx] * 1.5, 0.95)
            # Re-sort by score
            sorted_pairs = sorted(zip(result["labels"], result["scores"]), 
                                key=lambda x: x[1], reverse=True)
            result["labels"], result["scores"] = zip(*sorted_pairs)

        # Parse results
        domain_result = {
            "primary_domain": result["labels"][0],
            "confidence": result["scores"][0],
            "all_domains": [
                {"domain": label, "score": score}
                for label, score in zip(result["labels"], result["scores"])
                if score > 0.1  # Only include domains with >10% confidence
            ],
            "multi_domain": len([s for s in result["scores"] if s > 0.3]) > 1
        }
        
        print(f"✅ Primary Domain: {domain_result['primary_domain']} ({domain_result['confidence']:.2%} confidence)")
        if domain_result['multi_domain']:
            print(f"  ⚠️ Multi-domain project detected")
            for d in domain_result['all_domains'][:3]:
                print(f"    - {d['domain']}: {d['score']:.2%}")
        
        return domain_result
    
    def _requirement_to_text_simple(self, req: Dict) -> str:
        """Simple text extraction for classification"""
        parts = []
        if "title" in req:
            parts.append(req["title"])
        if "description" in req:
            parts.append(req["description"])
        if "gist" in req:
            parts.append(req["gist"])
        return " ".join(parts)
    
    def get_domain_specific_context(self, domain: str) -> Dict:
        """
        Get domain-specific testing context
        """
        domain_contexts = {
            "E-commerce": {
                "test_focus": ["Payment processing", "Cart functionality", "Checkout flow", "Product catalog", "User authentication"],
                "compliance": ["PCI-DSS", "GDPR", "ADA accessibility"],
                "critical_flows": ["Purchase flow", "Payment gateway integration", "Inventory management"]
            },
            "Finance": {
                "test_focus": ["Transaction accuracy", "Data security", "Regulatory compliance", "Audit trails"],
                "compliance": ["SOX", "GLBA", "PCI-DSS", "SEC regulations"],
                "critical_flows": ["Fund transfers", "Account management", "Report generation"]
            },
            "Healthcare": {
                "test_focus": ["Patient data security", "HIPAA compliance", "Medical record accuracy", "Appointment scheduling"],
                "compliance": ["HIPAA", "HITECH", "FDA regulations"],
                "critical_flows": ["Patient registration", "Medical records access", "Prescription management"]
            },
            "Restaurant/Food Service": {
                "test_focus": ["Search functionality", "Location services", "User ratings", "Booking system", "Payment integration"],
                "compliance": ["GDPR", "Food safety standards", "Accessibility"],
                "critical_flows": ["Restaurant search", "Reservation flow", "Review system", "GPS integration"]
            },
            "Location-based Services": {
                "test_focus": ["GPS accuracy", "Real-time updates", "Map integration", "Search radius", "Geolocation"],
                "compliance": ["Privacy regulations", "Location data handling"],
                "critical_flows": ["Location search", "Navigation", "Distance calculation"]
            },
            "Task Management / Productivity Tools": {
                "test_focus": [
                    "User authentication and authorization",
                    "Todo CRUD operations", 
                    "Data isolation between users",
                    "State management (completion status)",
                    "Input validation and error handling"
                ],
                "compliance": [
                    "GDPR (user data privacy)",
                    "OWASP API Security",
                    "Password storage best practices"
                ],
                "critical_flows": [
                    "User authentication flow (registration → login → JWT)",
                    "Todo management flow (create → update → complete → delete)",
                    "Data isolation enforcement",
                    "Error handling and validation"
                ]
            }
        }
        
        return domain_contexts.get(domain, {
            "test_focus": ["Functional correctness", "Performance", "Security", "Usability"],
            "compliance": ["General data protection", "Accessibility"],
            "critical_flows": ["Core user workflows"]
        })


def process_requirements_with_chunking_and_domain(
    cru_json_path: str = CRU_INPUT_PATH,
    requirements_json_path: str = REQUIREMENTS_INPUT_PATH,
    output_path: str = "output/chunked_requirements_with_domain.json"
) -> Dict:
    """
    Main processing function: Load CRUs, chunk them, classify domain
    
    Args:
        cru_json_path: Path to CRU units JSON
        requirements_json_path: Path to requirements JSON (for domain classification)
        output_path: Path to save processed output
        
    Returns:
        Dictionary containing chunks and domain classification
    """
    print("="*70)
    print("🚀 SEMANTIC CHUNKING + DOMAIN CLASSIFICATION PIPELINE (CRU-based)")
    print("="*70)
    
    # Load CRU units for chunking
    print(f"\n📂 Loading CRU units from: {cru_json_path}")
    with open(cru_json_path, 'r', encoding='utf-8') as f:
        cru_data = json.load(f)
    
    cru_units = cru_data.get("crus", [])
    print(f"✅ Loaded {len(cru_units)} CRU units")
    
    # Load original requirements for domain classification
    print(f"\n📂 Loading requirements for domain classification from: {requirements_json_path}")
    with open(requirements_json_path, 'r', encoding='utf-8') as f:
        req_data = json.load(f)
    
    # Combine all requirements types for domain classification
    all_requirements = []
    if "functional_requirements" in req_data:
        all_requirements.extend(req_data["functional_requirements"])
    if "quality_requirements" in req_data:
        all_requirements.extend(req_data["quality_requirements"])
    if "use_cases" in req_data:
        all_requirements.extend(req_data["use_cases"])
    if "constraints" in req_data:
        all_requirements.extend(req_data["constraints"])
    if "performance_requirements" in req_data:
        all_requirements.extend(req_data["performance_requirements"])
    
    print(f"✅ Loaded {len(all_requirements)} requirements for domain classification")
    
    # Initialize chunker and classifier
    chunker = SemanticChunker()
    classifier = DomainClassifier()
    
    # Create semantic chunks from CRUs (this is the key change!)
    chunks = chunker.create_semantic_chunks(cru_units)
    
    # Classify domain using original requirements
    domain_info = classifier.classify_domain(all_requirements)
    
    # Get domain-specific context
    domain_context = classifier.get_domain_specific_context(domain_info["primary_domain"])
    
    # Prepare output
    output = {
        "metadata": {
            "total_crus": len(cru_units),
            "total_requirements": len(all_requirements),
            "total_chunks": len(chunks),
            "avg_chunk_size": np.mean([c["token_count"] for c in chunks]),
            "chunking_model": CONFIG["embedding_model"],
            "classification_model": CONFIG["classification_model"],
            "cache_location": str(cache_location),
            "chunking_source": "CRU units"  # NEW: Indicate we're chunking CRUs
        },
        "domain_classification": domain_info,
        "domain_context": domain_context,
        "chunks": chunks
    }
    
    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved output to: {output_path}")
    print("="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    
    return output


# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    try:
        # Process CRU units (not raw requirements!)
        result = process_requirements_with_chunking_and_domain(
            cru_json_path="../03_CRU_Normalization/module_3_finalization/output/cru_units.json",
            requirements_json_path="../02_Requirement_Understanding/output/requirements_extracted_grouped.json",
            output_path="../03_Chunking_Domain_Understanding/output/chunked_crus_with_domain.json"
        )
        
        # Display summary
        print(f"\n📊 SUMMARY:")
        print(f"  - Total CRU Units: {result['metadata']['total_crus']}")
        print(f"  - Total Chunks: {result['metadata']['total_chunks']}")
        print(f"  - Primary Domain: {result['domain_classification']['primary_domain']}")
        print(f"  - Domain Confidence: {result['domain_classification']['confidence']:.2%}")
        print(f"  - Test Focus Areas: {', '.join(result['domain_context']['test_focus'][:3])}")
        print(f"  - Cache Location: {result['metadata']['cache_location']}")
        print(f"  - Chunking Source: {result['metadata']['chunking_source']}")
        
        # Show sample chunk with CRU IDs
        if result['chunks']:
            sample_chunk = result['chunks'][0]
            print(f"\n🔹 Sample Chunk:")
            print(f"  - Chunk ID: {sample_chunk['chunk_id']}")
            print(f"  - CRU IDs: {', '.join(sample_chunk.get('cru_ids', [])[:5])}")
            if 'parent_requirements' in sample_chunk:
                print(f"  - Parent Requirements: {', '.join(sample_chunk['parent_requirements'][:5])}")
            print(f"  - Token Count: {sample_chunk['token_count']}")
            print(f"  - Semantic Score: {sample_chunk['semantic_score']:.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 If you're experiencing cache issues, the cache cleaner runs automatically.")
        raise

