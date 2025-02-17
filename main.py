import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    llm_model: str
    embedding_model: str
    temperature: float = 0
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 5
    num_queries: int = 3
    rrf_k: int = 60

class RAGMultiQueryFusion:
    def __init__(self, docs_dir: str, config: ModelConfig = None):
        """Initialize RAG with MultiQuery Fusion"""
        # Load environment variables
        load_dotenv()
        
        self.docs_dir = Path(docs_dir)
        
        # Get API key from .env
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found in .env file. Please add OPENAI_API_KEY to your .env file."
            )
        
        # If no config provided, create from environment variables
        if config is None:
            config = ModelConfig(
                llm_model=os.getenv('LLM_MODEL', 'gpt-4'),
                embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '100')),
                top_k=int(os.getenv('TOP_K', '5')),
                num_queries=int(os.getenv('NUM_QUERIES', '3')),
                rrf_k=int(os.getenv('RRF_K', '60'))
            )
        
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=self.config.temperature,
            model_name=self.config.llm_model,
            openai_api_key=self.api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.api_key
        )
        
        self.vectorstore = None
        
        # MultiQuery generation prompt
        self.query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate variations of the search query that capture different aspects or phrasings while maintaining the core meaning."),
            ("user", """Generate {num_queries} different versions of this search query: "{query}"
            Make each version focus on slightly different aspects or use different phrasing.
            Return just the queries, one per line.""")
        ])

    def load_and_index_docs(self):
        """Load PDFs and create vector index"""
        documents = []
        # Load all PDFs from directory
        for pdf_path in self.docs_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents.extend(loader.load())
                print(f"Loaded: {pdf_path.name}")
            except Exception as e:
                print(f"Error loading {pdf_path.name}: {e}")

        if not documents:
            raise ValueError(f"No PDF documents found in {self.docs_dir}")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        splits = splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        print(f"Indexed {len(splits)} chunks from {len(documents)} documents")

    def generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of the input query"""
        try:
            # Create a chain for query generation
            chain = (
                self.query_gen_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            # Generate variations
            response = chain.invoke({
                "query": query,
                "num_queries": self.config.num_queries
            })
            
            # Get list of queries
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            queries.insert(0, query)  # Add original query
            
            print("\nQuery variations:")
            for i, q in enumerate(queries):
                print(f"{i}: {q}")
                
            return queries
        except Exception as e:
            print(f"Error generating query variations: {e}")
            # Return original query if generation fails
            return [query]


    def reciprocal_rank_fusion(self, results: List[List[Document]]) -> List[Document]:
        """Combine results using Reciprocal Rank Fusion"""
        doc_scores: Dict[str, float] = {}
        
        # Calculate RRF scores
        for query_results in results:
            for rank, doc in enumerate(query_results):
                doc_key = f"{doc.page_content[:100]}_{doc.metadata.get('page', 0)}"
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = 0
                doc_scores[doc_key] += 1 / (rank + self.config.rrf_k)
        
        # Get unique documents with highest scores
        seen_docs = set()
        final_docs = []
        
        # Sort documents by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final document list avoiding duplicates
        for doc_key, score in sorted_docs:
            for query_results in results:
                for doc in query_results:
                    current_key = f"{doc.page_content[:100]}_{doc.metadata.get('page', 0)}"
                    if current_key == doc_key and doc_key not in seen_docs:
                        final_docs.append(doc)
                        seen_docs.add(doc_key)
                        print(f"Document Score: {score:.4f}")
                        break
        
        return final_docs[:self.config.top_k]

    def query(self, question: str) -> str:
        """Process query through MultiQuery RRF pipeline"""
        if not self.vectorstore:
            raise ValueError("Please run load_and_index_docs() first")
        
        try:
            # Generate query variations
            queries = self.generate_query_variations(question)
            
            # Get results for each query
            all_results = []
            for query in queries:
                results = self.vectorstore.similarity_search(query, k=self.config.top_k)
                all_results.append(results)
            
            # Fuse results using RRF
            reranked_docs = self.reciprocal_rank_fusion(all_results)
            
            # Prepare context from reranked documents
            context = "\n\n".join(doc.page_content for doc in reranked_docs)
            
            # Create chain for answer generation
            qa_chain = (
                PromptTemplate.from_template(
                    """Answer the question based only on the following context. 
                    If you cannot answer from the context, say so.

                    Context: {context}

                    Question: {question}
                    
                    Answer: """
                )
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = qa_chain.invoke({
                "context": context,
                "question": question
            })
            
            return answer
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return f"An error occurred while processing your query: {str(e)}"

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Get documents directory from user or environment
        docs_dir = os.getenv('DOCS_DIR') or input("Enter the path to your documents directory: ").strip()
        if not os.path.exists(docs_dir):
            print(f"Directory not found: {docs_dir}")
            return
        
        print("Initializing RAG system...")
        rag = RAGMultiQueryFusion(docs_dir)
        
        print("Loading and indexing documents...")
        rag.load_and_index_docs()
        
        print("\nEnter your questions (type 'exit' to quit)")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() == 'exit':
                break
            
            try:
                print("\nProcessing query...")
                answer = rag.query(question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error processing query: {e}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()