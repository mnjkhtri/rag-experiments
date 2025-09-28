import numpy as np
import faiss


class Retriever:
    def __init__(self, corpus, embedder, k = 5, faiss_threshold=10000):
        self.embedder = embedder
        self.k = k
        self.corpus = corpus

        print(f"Total corpus size = {len(self.corpus)}")
        self.corpus_embeddings = self.embedder(self.corpus)
        self.corpus_embeddings = self._normalize(self.corpus_embeddings)

        self.index = self._build_faiss() if len(corpus) >= faiss_threshold else None

    def _normalize(self, x: np.ndarray): return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10, None)

    def forward(self, query: str):
        q_embed = self._normalize(self.embedder(query))
        if self.index is not None: pids = self.index.search(q_embed[None, :], self.k * 10)[1][0]
        else: pids = np.arange(len(self.corpus), dtype=np.int64)
        return self._rerank(q_embed, pids)

    def _build_faiss(self):
        nbytes = 32
        partitions = int(2 * np.sqrt(len(self.corpus)))
        dim = self.corpus_embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, partitions, nbytes, 8)

        print(
            f"Training a {nbytes}-byte FAISS index with {partitions} partitions, based on "
            f"{len(self.corpus)} x {dim}-dim embeddings"
        )

        # FAISS expects float32, contiguous
        xb = np.ascontiguousarray(self.corpus_embeddings.astype(np.float32))
        index.train(xb)
        index.add(xb)
        index.nprobe = min(16, partitions)
        return index
    
    def _rerank(self, q_embed, candidate_indices):
        cand_emb = self.corpus_embeddings[candidate_indices]  # (m, d)
        scores = np.einsum("d,nd->n", q_embed, cand_emb, optimize=True)  # (m,)
        top_k_indices = np.argsort(-scores)[:self.k]
        top_indices = candidate_indices[top_k_indices]
        return [self.corpus[idx] for idx in top_indices], top_indices.tolist()
    
    def __call__(self, query: str): return self.forward(query)