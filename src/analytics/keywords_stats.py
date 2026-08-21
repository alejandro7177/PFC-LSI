# src/analytics/keyword_stats.py
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl
from wordcloud import WordCloud

from src.core.config import config


class KeywordVisualizer:
    @staticmethod
    def _ensure_dir(output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def generate_all_plots(cls, keyword_count: dict[str, int], output_dir: str | Path, top_n: int | None = None):
        """Genera el paquete completo de gráficos de diagnóstico."""
        top_n = top_n if top_n is not None else config.get("visualization.top_n_bar_chart", 20)
        out_path = cls._ensure_dir(output_dir)

        cls.histogram(keyword_count, out_path)
        cls.top_keywords(keyword_count, out_path, top_n=top_n)
        cls.boxplot(keyword_count, out_path)
        cls.zipf_plot(keyword_count, out_path)
        cls.wordcloud(keyword_count, out_path)

    @classmethod
    def histogram(cls, keyword_count: dict[str, int], output_dir: Path, bins: int | None = None):
        bins = bins or config.get("visualization.histogram.bins", 20)
        dpi = config.get("visualization.dpi", 300)
        plt.figure(figsize=(8, 5))
        plt.hist(list(keyword_count.values()), bins=bins, color="skyblue", edgecolor="black")
        plt.xlabel("Frecuencia de Keyword")
        plt.ylabel("Número de Keywords")
        plt.title("Distribución de Frecuencia de Keywords")
        plt.tight_layout()
        plt.savefig(output_dir / "histogram.png", dpi=dpi)
        plt.close()

    @classmethod
    def top_keywords(cls, keyword_count: dict[str, int], output_dir: Path, top_n: int | None = None):
        top_n = top_n if top_n is not None else config.get("visualization.top_n_bar_chart", 20)
        dpi = config.get("visualization.dpi", 300)
        top = Counter(keyword_count).most_common(top_n)
        keywords, counts = zip(*top) if top else ([], [])

        plt.figure(figsize=(12, 6))
        plt.barh(list(keywords), list(counts), color="teal")
        plt.gca().invert_yaxis()
        plt.xlabel("Frecuencia")
        plt.title(f"Top {top_n} Keywords más Frecuentes")
        plt.tight_layout()
        plt.savefig(output_dir / f"top_{top_n}.png", dpi=dpi)
        plt.close()

    @classmethod
    def boxplot(cls, keyword_count: dict[str, int], output_dir: Path):
        dpi = config.get("visualization.dpi", 300)
        plt.figure(figsize=(5, 6))
        plt.boxplot(list(keyword_count.values()))
        plt.ylabel("Frecuencia")
        plt.title("Boxplot de Frecuencias de Keywords")
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot.png", dpi=dpi)
        plt.close()

    @classmethod
    def zipf_plot(cls, keyword_count: dict[str, int], output_dir: Path):
        dpi = config.get("visualization.dpi", 300)
        frequencies = sorted(keyword_count.values(), reverse=True)
        plt.figure(figsize=(8, 5))
        plt.plot(frequencies, color="crimson")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Rango (Rank)")
        plt.ylabel("Frecuencia")
        plt.title("Distribución Ley de Zipf (Log-Log)")
        plt.tight_layout()
        plt.savefig(output_dir / "zipf.png", dpi=dpi)
        plt.close()

    @classmethod
    def wordcloud(cls, keyword_count: dict[str, int], output_dir: Path):
        width = config.get("visualization.wordcloud.width", 1600)
        height = config.get("visualization.wordcloud.height", 800)
        bg_color = config.get("visualization.wordcloud.background_color", "white")
        dpi = config.get("visualization.dpi", 300)

        wc = WordCloud(width=width, height=height, background_color=bg_color).generate_from_frequencies(keyword_count)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "wordcloud.png", dpi=dpi)
        plt.close()