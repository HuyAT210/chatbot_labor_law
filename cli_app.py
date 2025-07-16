#!/usr/bin/env python3
"""
AI Document Assistant - Command Line Interface
"""

import os
import sys
import argparse
import shlex
from pathlib import Path
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rag_chain import ask_llm
from core.milvus_utilis import save_to_milvus, search_similar_chunks, delete_file, delete_all, collection
from core.embedding import split_into_chunks
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading text file {txt_path}: {e}")
        return ""

def process_document(file_path: str) -> bool:
    """Process a document and add it to the vector database."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Extract text based on file type
    if file_path.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(str(file_path))
    elif file_path.suffix.lower() in ['.txt', '.md']:
        text = extract_text_from_txt(str(file_path))
    else:
        print(f"❌ Unsupported file type: {file_path.suffix}")
        return False
    
    if not text.strip():
        print(f"❌ No text extracted from {file_path}")
        return False
    
    # Split into chunks
    chunks = split_into_chunks(text)
    print(f"📄 Extracted {len(chunks)} chunks from {file_path.name}")
    
    # Save to Milvus
    try:
        save_to_milvus(chunks, file_path.name)
        print(f"✅ Successfully processed {file_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return False

def interactive_mode():
    """Run the assistant in interactive mode."""
    print("🤖 AI Document Assistant - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  ask <question>     - Ask a question about your documents")
    print("  upload <file>      - Upload and process a document")
    print("  search <query>     - Search for similar content")
    print("  delete <filename>  - Delete a document from the database")
    print("  delete-all         - Delete ALL data from the database")
    print("  list               - List all documents in the database")
    print("  help               - Show this help message")
    print("  quit               - Exit the application")
    print("=" * 50)
    print("💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 Assistant> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  ask <question>     - Ask a question about your documents")
                print("  upload <file>      - Upload and process a document")
                print("  search <query>     - Search for similar content")
                print("  delete <filename>  - Delete a document from the database")
                print("  delete-all         - Delete ALL data from the database")
                print("  list               - List all documents in the database")
                print("  help               - Show this help message")
                print("  quit               - Exit the application")
                print("\n💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
                
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    print(f"\n🤔 Question: {question}")
                    print("🔄 Thinking...")
                    try:
                        answer = ask_llm(question)
                        print(f"\n💡 Answer: {answer}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a question after 'ask'")
                    
            elif user_input.lower().startswith('upload '):
                # Parse the upload command properly to handle spaces and quotes
                try:
                    parts = shlex.split(user_input)
                    if len(parts) >= 2:
                        file_path = parts[1]
                        print(f"📤 Uploading {file_path}...")
                        process_document(file_path)
                    else:
                        print("❌ Please provide a file path after 'upload'")
                except Exception as e:
                    print(f"❌ Error parsing file path: {e}")
                    print("💡 Tip: Use quotes for file paths with spaces, e.g., upload \"testing files/document.pdf\"")
                    
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    print(f"🔍 Searching for: {query}")
                    try:
                        results = search_similar_chunks(query, top_k=5)
                        print(f"\n📋 Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            print(f"\n{i}. Score: {result['score']:.3f}")
                            print(f"   Content: {result['chunk'][:200]}...")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a search query after 'search'")
                    
            elif user_input.lower().startswith('delete '):
                filename = user_input[7:].strip()
                if filename:
                    print(f"🗑️ Deleting {filename}...")
                    try:
                        result = delete_file(filename)
                        print(f"✅ {result['message']}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Please provide a filename after 'delete'")
                    
            elif user_input.lower() == 'delete-all':
                print("🗑️ WARNING: This will delete ALL data from the database!")
                confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
                if confirm == 'yes':
                    try:
                        result = delete_all()
                        print(f"✅ {result['message']}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Delete all operation cancelled.")
                    
            elif user_input.lower() == 'list':
                try:
                    collection.load()
                    results = collection.query(
                        expr="",
                        output_fields=["filename"],
                        limit=1000
                    )
                    filenames = list(set([r["filename"] for r in results]))
                    if filenames:
                        print(f"\n📚 Documents in database ({len(filenames)}):")
                        for filename in filenames:
                            print(f"  - {filename}")
                    else:
                        print("📚 No documents in database")
                except Exception as e:
                    print(f"❌ Error listing documents: {e}")
                    
            else:
                # If no command is recognized, treat it as a question
                print(f"🤔 Question: {user_input}")
                print("🔄 Thinking...")
                try:
                    answer = ask_llm(user_input)
                    print(f"\n💡 Answer: {answer}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="AI Document Assistant CLI")
    parser.add_argument("--ask", "-a", help="Ask a question directly")
    parser.add_argument("--upload", "-u", help="Upload and process a document")
    parser.add_argument("--search", "-s", help="Search for similar content")
    parser.add_argument("--delete", "-d", help="Delete a document from the database")
    parser.add_argument("--delete-all", action="store_true", help="Delete ALL data from the database")
    parser.add_argument("--list", "-l", action="store_true", help="List all documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not any([args.ask, args.upload, args.search, args.delete, args.delete_all, args.list]):
        interactive_mode()
    else:
        # Single command mode
        if args.ask:
            print(f"🤔 Question: {args.ask}")
            print("🔄 Thinking...")
            try:
                answer = ask_llm(args.ask)
                print(f"\n💡 Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.upload:
            print(f"📤 Uploading {args.upload}...")
            process_document(args.upload)
            
        elif args.search:
            print(f"🔍 Searching for: {args.search}")
            try:
                results = search_similar_chunks(args.search, top_k=5)
                print(f"\n📋 Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result['score']:.3f}")
                    print(f"   Content: {result['chunk'][:200]}...")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.delete:
            print(f"🗑️ Deleting {args.delete}...")
            try:
                result = delete_file(args.delete)
                print(f"✅ {result['message']}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif args.delete_all:
            print("🗑️ WARNING: This will delete ALL data from the database!")
            confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
            if confirm == 'yes':
                try:
                    result = delete_all()
                    print(f"✅ {result['message']}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Delete all operation cancelled.")
                
        elif args.list:
            try:
                collection.load()
                results = collection.query(
                    expr="",
                    output_fields=["filename"],
                    limit=1000
                )
                filenames = list(set([r["filename"] for r in results]))
                if filenames:
                    print(f"\n📚 Documents in database ({len(filenames)}):")
                    for filename in filenames:
                        print(f"  - {filename}")
                else:
                    print("📚 No documents in database")
            except Exception as e:
                print(f"❌ Error listing documents: {e}")

if __name__ == "__main__":
    main() 