import PyPDF2
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

class PDFToTextConverter:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = None

    def convert_to_text(self):
        # Open the PDF file specified by the pdf_path attribute
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Extract text from each page and concatenate it
            self.text = "".join([page.extract_text() for page in pdf_reader.pages])
        return self.text



class TextChunker:
    def __init__(self, text):
        self.text = text
        self.chunks = None

    def split_into_chunks(self, chunk_size=1000, by_words=False):
        if by_words:
            words = self.text.split()
            self.chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        else:
            self.chunks = [self.text[i:i + chunk_size] for i in range(0, len(self.text), chunk_size)]
        return self.chunks


class DocumentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def generate_embeddings(self, chunks):
        self.embeddings = self.model.encode(chunks, convert_to_tensor=True)
        return self.embeddings


class QueryHandler:
    def __init__(self, embedder):
        self.embedder = embedder

    def generate_query_embedding(self, query):
        return self.embedder.model.encode(query, convert_to_tensor=True)

    def find_relevant_chunks(self, query_embedding, top_k=3):
        similarities = util.cos_sim(query_embedding, self.embedder.embeddings)[0]
        top_k_indices = similarities.topk(k=top_k).indices
        return top_k_indices

class ResponseGenerator:
    def __init__(self, model_name='gpt2-xl'):
        self.generator = pipeline('text-generation', model=model_name)

    def generate_response(self, query, relevant_chunks):
        context = " ".join(relevant_chunks)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.generator(prompt, max_length=200, do_sample=True, top_p=0.9, temperature=0.1)
        return response[0]['generated_text']

# Example Usage

# Step 1: Convert PDF to Text
self.pdf_path = r'C:\Users\DELL\Desktop\Never-Split-the-Difference.pdf'
pdf_converter = PDFToTextConverter(pdf_path=self.pdf_path)

text_data = pdf_converter.convert_to_text()

# Step 2: Split Text into Chunks
text_chunker = TextChunker(text_data)
chunks = text_chunker.split_into_chunks(chunk_size=1000)  # Modify chunk_size as needed

# Step 3: Generate Embeddings for Chunks
embedder = DocumentEmbedder()
embeddings = embedder.generate_embeddings(chunks)

query_handler = QueryHandler(embedder)


response_generator = ResponseGenerator()