from litellm import embedding


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = embedding(model=self.model, input=texts)
        return [d["embedding"] for d in resp.data]
