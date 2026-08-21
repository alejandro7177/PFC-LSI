import arxiv
from typing import Generator
from src.core.config import config


class ArxivExtractor:
    def __init__(self, query: str | None = None, max_results: int | None = None):
        self.query = query or config.get("arxiv_extractor.query", "cat:physics.ao-ph")
        self.max_results = max_results if max_results is not None else config.get("arxiv_extractor.max_results", 10000)

    def stream_papers(self) -> Generator[dict, None, None]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        for i, paper in enumerate(client.results(search)):
            yield {
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors),
                "published": str(paper.published),
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url
            }