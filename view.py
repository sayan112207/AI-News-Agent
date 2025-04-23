import faiss
import pickle

# Load the FAISS index and your documents
index = faiss.read_index('vector.index')
with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)

print(f"Number of vectors in index: {index.ntotal}")
print("Sample documents:")
for i, doc in enumerate(docs):  # Show all documents
    print(f"{i+1}. {doc}")
