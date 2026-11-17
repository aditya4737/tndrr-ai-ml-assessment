import re
import torch

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATION_MODEL

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "does", "do", "did", "can", "could", "should", "would", "and", "or",
    "of", "to", "in", "on", "for", "with", "by", "from", "that", "this",
    "these", "those", "it", "its", "as", "be", "about", "their", "they",
    "them", "which", "who"
}

EXPANSIONS = {
    "trustworthy": "trustworthy AI characteristics valid reliable safe secure resilient accountable transparent explainable interpretable privacy enhanced fair harmful bias managed",
    "voluntary": "voluntary voluntary use not mandatory intended to be voluntary framework",
    "actors": "AI actors defined individuals organizations design development deployment evaluation operation use AI system lifecycle",
    "third_party": "risks related to third-party software hardware and data external components supply chain dependencies vulnerabilities",
    "prioritization": "risk prioritization may differ likelihood impact severity resources context AI actors",
    "validity": "validity reliability correctness expected use lifetime AI system",
    "govern": "GOVERN function culture risk management policies processes organizational structures",
    "measure": "MEASURE function quantitative qualitative mixed-method tools methods metrics evaluate assess",
    "training": "training data changes over time AI system performance behavior functionality trustworthiness",
    "unique_risks": "AI risks differ traditional software systems unique risks data changes scale opacity emergent properties"
}

def clean_text(text):
    return re.sub(r"\s+", " ", str(text).replace("-\n", "")).strip()

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())

def meaningful_words(text):
    return {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower()) if w not in STOPWORDS}

def split_sentences(text):
    text = clean_text(text)
    text = re.sub(r"(?<=[a-zA-Z])-\s+(?=[a-z])", "", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•])", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]

class AnswerGenerator:
    def __init__(self, retriever, load_llm=True, model_name=GENERATION_MODEL):
        self.retriever = retriever
        self.tokenizer = None
        self.model = None
        self.cross_encoder = getattr(retriever, "cross_encoder", getattr(retriever, "reranker", None))
        self.device = torch.device("cpu")
        if load_llm:
            print("Loading FLAN-T5-large...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
            self.model.to(self.device)
            self.model.eval()
            print("FLAN-T5-large loaded.")

    @staticmethod
    def intent(question):
        q = question.lower()
        if "trustworthy" in q or "characteristics" in q: return "trustworthy"
        if "voluntary" in q or "mandatory" in q: return "voluntary"
        if "actor" in q: return "actors"
        if "third-party" in q or "third party" in q: return "third_party"
        if "prioritiz" in q: return "prioritization"
        if "validity" in q or "reliability" in q: return "validity"
        if "govern function" in q: return "govern"
        if "measure function" in q: return "measure"
        if "training data" in q: return "training"
        if "traditional software" in q: return "unique_risks"
        return None

    def collect(self, retrieved):
        result, seen = [], set()
        for item in retrieved:
            chunk = item.get("chunk", item)
            page = chunk.get("page")
            for sentence in split_sentences(chunk.get("text", "")):
                key = re.sub(r"\W+", " ", sentence.lower()).strip()
                if key not in seen:
                    seen.add(key)
                    result.append({"text": sentence, "page": page})
        return result

    def rank(self, question, candidates):
        if not candidates: return []
        intent = self.intent(question)
        query = EXPANSIONS.get(intent, question)
        pairs = [(question, x["text"]) for x in candidates]
        if self.cross_encoder is not None:
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            cross_order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        else:
            cross_order = list(range(len(candidates)))
        cross_rank = {idx: rank + 1 for rank, idx in enumerate(cross_order)}
        bm25 = BM25Okapi([tokenize(x["text"]) for x in candidates])
        lexical = bm25.get_scores(tokenize(query))
        bm_order = sorted(range(len(candidates)), key=lambda i: float(lexical[i]), reverse=True)
        bm_rank = {idx: rank + 1 for rank, idx in enumerate(bm_order)}

        ranked = []
        for i, item in enumerate(candidates):
            s = item["text"].lower()
            bonus = 0.0
            cues = {
                "trustworthy": ["characteristics of trustworthy", "valid and reliable", "safe", "secure and resilient", "accountable and transparent", "explainable and interpretable", "privacy-enhanced", "fair"],
                "voluntary": ["framework is intended to be voluntary", "voluntary"],
                "actors": ["ai actors are defined", "active role", "ai system lifecycle"],
                "third_party": ["risks related to third-party", "third-party software", "third-party data"],
                "prioritization": ["risk prioritization may differ", "prioritization", "impact"],
                "validity": ["validity", "reliability is a goal"],
                "govern": ["culture of risk management", "govern function", "infused throughout"],
                "measure": ["quantitative, qualitative", "measure function", "measuring"],
                "training": ["data that can change over time", "changes during training", "training data"],
                "unique_risks": ["risks posed by ai systems", "unique", "traditional software"]
            }.get(intent, [])
            bonus += sum(0.12 for cue in cues if cue in s)
            if intent == "voluntary" and "living document" in s: bonus -= 0.2
            if intent == "third_party" and "accelerate research and development" in s: bonus -= 0.2
            ranked.append({**item, "score": 1/(60+cross_rank[i]) + 1/(60+bm_rank[i]) + bonus})
        return sorted(ranked, key=lambda x: x["score"], reverse=True)

    def choose_evidence(self, question, ranked, limit=5):
        intent = self.intent(question)
        if not ranked: return []
        # Prefer the actual definition/anchor sentence, then supporting sentences.
        preferred = {
            "trustworthy": ["characteristics of trustworthy", "valid and reliable", "privacy-enhanced", "accountable and transparent"],
            "voluntary": ["framework is intended to be voluntary"],
            "actors": ["ai actors are defined", "active role"],
            "third_party": ["risks related to third-party"],
            "prioritization": ["risk prioritization may differ"],
            "validity": ["validity", "reliability is a goal"],
            "govern": ["culture of risk management", "infused throughout"],
            "measure": ["quantitative, qualitative", "measure function employs"],
            "training": ["data that can change over time", "changes during training"],
            "unique_risks": ["risks posed by ai systems", "unique"]
        }.get(intent, [])
        selected = []
        for cue in preferred:
            for item in ranked:
                if cue in item["text"].lower() and item not in selected:
                    selected.append(item)
                    break
        for item in ranked:
            if len(selected) >= limit: break
            if item not in selected: selected.append(item)
        return selected[:limit]

    def generate_with_flan(self, question, evidence):
        context = "\n".join(f"[Page {x['page']}] {x['text']}" for x in evidence)
        prompt = f"""Answer the question directly using only the evidence below.
Include the complete definition or list when present. Do not answer with a related
but different fact. Do not invent details.

Evidence:
{context}

Question: {question}
Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=110, num_beams=4, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def generate(self, question):
        retrieved = self.retriever.retrieve(question, candidate_k=10, top_k=6)
        retrieved_pages = list(dict.fromkeys(
            r.get("chunk", r).get("page") for r in retrieved
            if r.get("chunk", r).get("page") is not None
        ))
        if not retrieved:
            return {"answer": "No relevant information was retrieved.", "sources": [], "method": "No retrieval",
                    "llm_output": "", "evidence": [], "evidence_pages": [], "answer_overlap": 0.0, "retrieved_pages": []}

        ranked = self.rank(question, self.collect(retrieved))
        evidence = self.choose_evidence(question, ranked)
        evidence_text = " ".join(x["text"] for x in evidence)
        answer, generated, method, overlap = evidence_text, "", "Extractive fallback", 0.0

        if self.model is not None and evidence:
            try:
                generated = self.generate_with_flan(question, evidence)
                sw, aw = meaningful_words(evidence_text), meaningful_words(generated)
                overlap = len(sw & aw) / max(len(aw), 1)
                if 5 <= len(generated.split()) <= 110 and overlap >= 0.40:
                    answer, method = generated, "FLAN-T5-large"
            except Exception as exc:
                print("Generation error:", exc)

        pages = list(dict.fromkeys(x["page"] for x in evidence if x["page"] is not None))
        return {
            "answer": answer, "sources": retrieved_pages, "method": method,
            "llm_output": generated, "evidence": [x["text"] for x in evidence],
            "evidence_pages": pages, "answer_overlap": round(overlap, 3),
            "retrieved_pages": retrieved_pages
        }
