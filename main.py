#!/usr/bin/env python3
"""
Drip-feed Amazon pet content pipeline.

This script syncs Amazon dog/cat BSR leaf categories into a Git-trackable JSON
state file, picks a small pending batch, caches CLI scraper output, generates
neutral Markdown reviews with an OpenAI-compatible LLM, and exports Hugo/Astro
content files.
"""

from __future__ import annotations

import argparse
import shlex
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_CATEGORY_JSON = ROOT / "data" / "my_output.json"
LOCAL_CATEGORY_JSON = Path("/Users/yuchuanlong/ai/amazon_pet/my_output.json")
TRACKING_JSON = ROOT / "data" / "tracking.json"
BESTSELLER_CACHE = ROOT / ".cache" / "bestsellers"
PRODUCT_CACHE = ROOT / ".cache" / "products"
OUTPUT_DIR = ROOT / "content"

LOGGER = logging.getLogger("amazon_pet_pipeline")
ASIN_RE = re.compile(r"(?:/dp/|/gp/product/|/product/|asin=)([A-Z0-9]{10})|(?:^|[^A-Z0-9])([A-Z0-9]{10})(?:[^A-Z0-9]|$)")


@dataclass(frozen=True)
class CategoryTask:
    node_id: str
    category_path: str
    category_name: str
    bsr_url: str
    pet_type: str


@dataclass(frozen=True)
class PublishedArticle:
    title: str
    url: str
    pet_type: str
    category_name: str
    category_path: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader to avoid an extra runtime dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "pet-products"


def safe_json_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def default_category_json() -> Path:
    if DEFAULT_CATEGORY_JSON.exists():
        return DEFAULT_CATEGORY_JSON
    return LOCAL_CATEGORY_JSON


class TaskManager:
    VALID_STATUSES = {"pending", "processing", "completed", "failed"}

    def __init__(self, tracking_path: Path = TRACKING_JSON) -> None:
        self.tracking_path = tracking_path
        self.tracking_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def close(self) -> None:
        self.save()

    def _load_state(self) -> dict[str, Any]:
        if not self.tracking_path.exists():
            return {"version": 1, "last_synced": None, "categories": []}
        with self.tracking_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("version", 1)
        state.setdefault("last_synced", None)
        state.setdefault("categories", [])
        return state

    def save(self) -> None:
        tmp_path = self.tracking_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.tracking_path)

    def sync_category_tree(self, category_json_path: Path) -> int:
        with category_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        existing = {str(item["node_id"]): item for item in self.state["categories"]}
        now = utc_now()
        synced_count = 0
        for pet_type in ("dogs", "cats"):
            nodes = data.get(pet_type, [])
            leaves = self._find_leaf_categories(nodes)
            for node in leaves:
                node_id = str(node["node_id"])
                if node_id not in existing:
                    existing[node_id] = {
                        "node_id": node_id,
                        "category_path": node["category_path"],
                        "category_name": node["category_name"],
                        "bsr_url": node["bsr_url"],
                        "pet_type": pet_type,
                        "status": "pending",
                        "last_updated": now,
                    }
                else:
                    existing[node_id].update(
                        {
                            "category_path": node["category_path"],
                            "category_name": node["category_name"],
                            "bsr_url": node["bsr_url"],
                            "pet_type": pet_type,
                        }
                    )
                    if existing[node_id].get("status") not in self.VALID_STATUSES:
                        existing[node_id]["status"] = "pending"
                        existing[node_id]["last_updated"] = now
                synced_count += 1

        self.state["categories"] = sorted(existing.values(), key=lambda item: (item["pet_type"], item["category_path"]))
        self.state["last_synced"] = now
        self.save()
        LOGGER.info("Synced %s leaf categories into %s", synced_count, self.tracking_path)
        return synced_count

    @staticmethod
    def _find_leaf_categories(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paths = [node.get("category_path", "") for node in nodes]
        leaves = []
        for node in nodes:
            path = node.get("category_path", "")
            if not path:
                continue
            is_parent = any(other != path and other.startswith(f"{path} >") for other in paths)
            if not is_parent and node.get("bsr_url"):
                leaves.append(node)
        return leaves

    def get_next_batch(self, limit: int = 5) -> list[CategoryTask]:
        pending = [item for item in self.state["categories"] if item.get("status") == "pending"]
        pending.sort(key=lambda item: (item.get("last_updated", ""), item.get("node_id", "")))
        batch = pending[:limit]
        if not batch:
            return []

        now = utc_now()
        for item in batch:
            item["status"] = "processing"
            item["last_updated"] = now
        self.save()
        LOGGER.info("Claimed %s pending categories for processing", len(batch))
        return [
            CategoryTask(
                node_id=item["node_id"],
                category_path=item["category_path"],
                category_name=item["category_name"],
                bsr_url=item["bsr_url"],
                pet_type=item["pet_type"],
            )
            for item in batch
        ]

    def mark_completed(self, node_id: str, article_path: Path, article_url: str, title: str) -> None:
        self._set_status(
            node_id,
            "completed",
            {
                "article_path": str(article_path.relative_to(ROOT)),
                "article_url": article_url,
                "article_title": title,
            },
        )

    def mark_failed(self, node_id: str) -> None:
        self._set_status(node_id, "failed")

    def reset_failed_to_pending(self) -> int:
        reset_count = 0
        now = utc_now()
        for item in self.state["categories"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
                item["last_updated"] = now
                reset_count += 1
        if reset_count:
            self.save()
        return reset_count

    def reset_processing_to_pending(self) -> int:
        reset_count = 0
        now = utc_now()
        for item in self.state["categories"]:
            if item.get("status") == "processing":
                item["status"] = "pending"
                item["last_updated"] = now
                reset_count += 1
        if reset_count:
            self.save()
        return reset_count

    def get_related_articles(self, task: CategoryTask, limit: int = 5) -> list[PublishedArticle]:
        completed = [
            item
            for item in self.state["categories"]
            if item.get("status") == "completed"
            and item.get("article_title")
            and item.get("article_url")
            and item.get("node_id") != task.node_id
        ]
        same_pet = [item for item in completed if item.get("pet_type") == task.pet_type]
        task_terms = self._path_terms(task.category_path)
        same_pet.sort(
            key=lambda item: (
                len(task_terms & self._path_terms(item.get("category_path", ""))),
                item.get("last_updated", ""),
            ),
            reverse=True,
        )
        return [
            PublishedArticle(
                title=item["article_title"],
                url=item["article_url"],
                pet_type=item["pet_type"],
                category_name=item["category_name"],
                category_path=item.get("category_path", ""),
            )
            for item in same_pet[:limit]
        ]

    @staticmethod
    def _path_terms(category_path: str) -> set[str]:
        stopwords = {"pet", "supplies", "dogs", "dog", "cats", "cat", "and", "the", "for"}
        terms = re.findall(r"[a-z0-9]+", category_path.lower())
        return {term for term in terms if term not in stopwords and len(term) > 2}

    def _set_status(self, node_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        for item in self.state["categories"]:
            if item.get("node_id") == node_id:
                item["status"] = status
                item["last_updated"] = utc_now()
                if extra:
                    item.update(extra)
                self.save()
                return
        raise KeyError(f"Unknown node_id: {node_id}")


class ScraperEngine:
    def __init__(
        self,
        bestseller_cache: Path = BESTSELLER_CACHE,
        product_cache: Path = PRODUCT_CACHE,
        timeout_seconds: int = 180,
        bestsellers_command_template: str | None = None,
        product_command_template: str | None = None,
    ) -> None:
        self.bestseller_cache = bestseller_cache
        self.product_cache = product_cache
        self.timeout_seconds = timeout_seconds
        self.bestsellers_command_template = (
            bestsellers_command_template
            or os.environ.get("AUTOCLI_BESTSELLERS_COMMAND")
            or "autocli amazon bestsellers {url} -f json"
        )
        self.product_command_template = (
            product_command_template
            or os.environ.get("AUTOCLI_PRODUCT_COMMAND")
            or "autocli amazon product {asin} -f json"
        )
        self.bestseller_cache.mkdir(parents=True, exist_ok=True)
        self.product_cache.mkdir(parents=True, exist_ok=True)

    def scrape_category(self, task: CategoryTask, top_n: int = 20, min_success: int = 10) -> list[dict[str, Any]]:
        LOGGER.info("Scraping %s (%s)", task.category_name, task.node_id)
        bestseller_payload = self._cached_autocli_json(
            self._format_command(self.bestsellers_command_template, url=task.bsr_url, node_id=task.node_id),
            self.bestseller_cache / f"{safe_json_filename(task.node_id)}.json",
        )
        asins = self._extract_top_asins(bestseller_payload, limit=top_n)
        LOGGER.info("Found %s ASIN candidates for %s", len(asins), task.node_id)

        products: list[dict[str, Any]] = []
        for asin in asins:
            try:
                payload = self._cached_autocli_json(
                    self._format_command(self.product_command_template, asin=asin),
                    self.product_cache / f"{asin}.json",
                )
            except Exception as exc:
                LOGGER.warning("Skipping ASIN %s after product fetch failure: %s", asin, exc)
                continue

            compact = self._compact_product_payload(payload, asin)
            if compact:
                products.append(compact)

        if len(products) < min_success:
            raise RuntimeError(f"Only fetched {len(products)} usable products; need at least {min_success}")

        LOGGER.info("Fetched %s usable products for %s", len(products), task.node_id)
        return products

    def _cached_autocli_json(self, command: list[str], cache_path: Path) -> Any:
        cached = self._read_cache(cache_path)
        if cached is not None:
            LOGGER.info("Cache hit: %s", cache_path)
            return cached

        LOGGER.info("Cache miss; running: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            LOGGER.error("AutoCLI command failed with exit code %s", exc.returncode)
            if stdout:
                LOGGER.error("AutoCLI stdout:\n%s", stdout[-4000:])
            if stderr:
                LOGGER.error("AutoCLI stderr:\n%s", stderr[-4000:])
            raise

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            LOGGER.error("AutoCLI returned non-JSON stdout:\n%s", completed.stdout[-4000:])
            if completed.stderr.strip():
                LOGGER.error("AutoCLI stderr:\n%s", completed.stderr[-4000:])
            raise
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def _format_command(template: str, **values: str) -> list[str]:
        return shlex.split(template.format(**{key: shlex.quote(value) for key, value in values.items()}))

    @staticmethod
    def _read_cache(cache_path: Path) -> Any | None:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid JSON cache file: %s", cache_path)
            return None

    def _extract_top_asins(self, payload: Any, limit: int) -> list[str]:
        asins: list[str] = []
        for item in self._iter_product_like_items(payload):
            asin = self._extract_asin(item)
            if asin and asin not in asins:
                asins.append(asin)
            if len(asins) >= limit:
                break
        return asins

    @staticmethod
    def _iter_product_like_items(payload: Any) -> Iterable[Any]:
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict):
            return
        for key in ("products", "items", "results", "bestsellers", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield payload

    def _extract_asin(self, item: Any) -> str | None:
        if isinstance(item, dict):
            for key in ("asin", "ASIN", "product_asin", "productAsin"):
                value = item.get(key)
                if isinstance(value, str) and re.fullmatch(r"[A-Z0-9]{10}", value):
                    return value
            for key in ("url", "link", "product_url", "productUrl", "href"):
                value = item.get(key)
                if isinstance(value, str):
                    asin = self._extract_asin_from_text(value)
                    if asin:
                        return asin
            return self._extract_asin_from_text(json.dumps(item, ensure_ascii=False))
        if isinstance(item, str):
            return self._extract_asin_from_text(item)
        return None

    @staticmethod
    def _extract_asin_from_text(text: str) -> str | None:
        match = ASIN_RE.search(text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    @staticmethod
    def _compact_product_payload(payload: Any, asin: str) -> dict[str, Any]:
        source = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else payload
        if not isinstance(source, dict):
            return {"asin": asin, "raw": source}

        def first(*keys: str) -> Any:
            for key in keys:
                if key in source and source[key] not in (None, "", []):
                    return source[key]
            return None

        image = first("image", "image_url", "imageUrl", "main_image", "mainImage", "thumbnail")
        if isinstance(image, dict):
            image = first("url", "src") or image.get("url") or image.get("src")

        return {
            "asin": asin,
            "title": first("title", "name", "product_title", "productTitle"),
            "price": first("price", "current_price", "currentPrice", "display_price", "displayPrice"),
            "rating": first("rating", "stars", "average_rating", "averageRating"),
            "review_count": first("review_count", "reviewCount", "ratings_count", "ratingsCount"),
            "image_url": image,
            "customers_say": first(
                "customers_say",
                "customersSay",
                "consumers_say",
                "consumersSay",
                "customer_summary",
                "customerSummary",
            ),
            "star_distribution": first("star_distribution", "starDistribution")
            or {
                "5_star": first("rating_5star", "rating5star"),
                "4_star": first("rating_4star", "rating4star"),
                "3_star": first("rating_3star", "rating3star"),
                "2_star": first("rating_2star", "rating2star"),
                "1_star": first("rating_1star", "rating1star"),
            },
            "reviews": first("reviews", "review_snippets", "reviewSnippets"),
            "features": first("features", "bullets", "bullet_points", "bulletPoints"),
            "ai_vision_report": first("ai_vision_report", "aiVisionReport", "vision_report", "visionReport"),
        }


class ContentGenerator:
    SYSTEM_PROMPT = """
You are a senior pet behavior-informed product reviewer, conversion-focused SEO editor, and skeptical buyer advocate.
Write in English for US pet owners. Your job is not to sound like a catalog. Your job is to help a real dog or cat owner avoid regret.

Banned Phrases & Tone Rules (CRITICAL):
- NEVER use the following AI clichés: "delve into", "a testament to", "crucial", "in conclusion", "vital", "elevate", "realm", "bustling", "moreover", "furthermore", "tapestry", "game-changer", "unleash", "furry friend", "picture this", "navigate", "symphony", "undeniable", "paramount".
- DO NOT use robotic transitional phrases or summary paragraphs that add no value.
- Write in short, punchy paragraphs (maximum 2-3 sentences).
- Use a conversational, slightly cynical, yet highly experienced first-hand tone (e.g., "What we noticed", "The biggest flaw here is", "I'd skip this if").
- Use bold text heavily to make the article highly scannable for mobile readers.

SEO and helpful-content strategy:
- Follow Google's E-E-A-T guidelines strictly. Emphasize original decision value: regret analysis, owner-fit matching, tradeoffs, red flags, and usage tips.
- You must use LSI (Latent Semantic Indexing) keywords naturally throughout the text related to the specific pet and product category.
- Use question-based H3 headers for long-tail SEO where natural (e.g., instead of just "Durability", use "Is it safe for power chewers?").
- Do not repeat the same praise for every product. Every product section needs a distinct reason to exist.

Evidence rules:
- Use only the facts provided in the Product JSON. Do not invent specs, testing, photos, studies, counts, or claims.
- The most valuable section is not the spec list. Focus on what buyers might regret after purchase and how to use the product smarter.
- Do not overstate medical, nutrition, behavior, or safety claims. 

Link and compliance rules:
- Do not use affiliate, tracking, shortened, or redirected links.
- Every purchase link must be exactly: [Check Price on Amazon](https://www.amazon.com/dp/{ASIN})
- Never include exact prices. Use broad tiers like "Budget-friendly", "Mid-range", and "Premium price".
- Use descriptive anchor text for internal links; avoid anchors like "click here".

Image rules:
- For every individual product section, place the product image immediately under that product heading using Markdown:
  ![{SEO alt text featuring specific long-tail keywords}]({image_url})
- If a product has no image_url, omit only the image line for that product.

Required Markdown structure:
1. Introduction: 2-4 tight paragraphs, pain-first, no generic filler.
2. How We Read This List: short, transparent evidence note based on marketplace signals.
3. Quick Picks: a compact bullet list naming the best product for 4-6 specific buyer needs.
4. Buying Guide: practical criteria, red flags, safety notes, and 1-2 authority outbound links where relevant.
5. Comparison Table: include product, best for, standout upside, buyer caution, skip-if. (Do NOT include price).
6. Deep Reviews: Provide detailed reviews for exactly 10 products. For each, include:
   - image under heading
   - short verdict
   - best for
   - skip it if
   - what buyers may regret
   - pros / cons
   - Expert Tip (actionable, non-obvious)
   - clean Amazon link
7. Final Summary: brief, scenario-based wrap-up.

Output Markdown body only. Do not output YAML frontmatter.
""".strip()

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen-plus"
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "8000"))
        self.draft_model = os.environ.get("GEMINI_MODEL")
        self.draft_base_url = os.environ.get("GEMINI_BASE_URL")
        self.draft_api_key = os.environ.get("GEMINI_API_KEY")

    def generate(
        self,
        task: CategoryTask,
        products: list[dict[str, Any]],
        related_articles: list[PublishedArticle] | None = None,
    ) -> str:
        prompt_products = self._prepare_products_for_prompt(products)[:10]
        compact_json = json.dumps(prompt_products, ensure_ascii=False, separators=(",", ":"))
        user_prompt = (
            f"Category path: {task.category_path}\n"
            f"Category name: {task.category_name}\n"
            f"Pet type: {task.pet_type}\n"
            "Article goal: create a helpful long-tail search page that helps owners decide what to buy, what to skip, "
            "and what tradeoffs to expect. Prioritize regret prevention, owner-fit matching, and practical use tips.\n"
            f"Related internal articles already published:\n{self._related_articles_json(related_articles or [])}\n"
            f"Product JSON:\n{compact_json}\n"
        )

        LOGGER.info("Generating Markdown with model %s for %s", self.model, task.node_id)

        # Only use the native Anthropic Claude streaming path when hitting the
        # official Anthropic endpoint.  Third-party proxies (gptsapi, openrouter,
        # etc.) expose an OpenAI-compatible interface and will return an empty
        # stream if we send Anthropic's SSE format, so route them through the
        # OpenAI SDK path instead.
        has_draft_model = bool(self.draft_model and self.draft_api_key)
        is_proxy_claude = "claude" in self.model.lower() and self.base_url
        is_native_anthropic = "claude" in self.model.lower() and not self.base_url

        if has_draft_model:
            LOGGER.info("Using %s to generate the initial article draft.", self.draft_model)
            draft_prompt = user_prompt + "\n\nCRITICAL DIRECTIVE: Generate the entire Markdown article (all 7 sections) based on the JSON above. Ensure facts are absolutely accurate."
            
            old_model, old_base_url, old_api_key = self.model, self.base_url, self.api_key
            self.model, self.base_url, self.api_key = self.draft_model, self.draft_base_url, self.draft_api_key
            try:
                draft_markdown = self._generate_openai(self.SYSTEM_PROMPT, draft_prompt)
            finally:
                self.model, self.base_url, self.api_key = old_model, old_base_url, old_api_key
                
            skip_refinement = os.environ.get("SKIP_CLAUDE_REFINEMENT", "").lower() == "true"
            if skip_refinement:
                LOGGER.info("SKIP_CLAUDE_REFINEMENT is true. Bypassing Claude refinement and using Gemini draft directly.")
                body = draft_markdown
            else:
                LOGGER.info("Draft generated successfully. Now using %s to refine the draft.", self.model)
                refinement_prompt = (
                    "Here is a drafted article. Your task is to rewrite and polish it to match the required "
                    "editorial voice, professional tone, and SEO logic. Do not change the underlying facts, product names, "
                    "or prices. Maintain the exact same 7 sections.\n\n"
                    f"=== DRAFT ARTICLE ===\n{draft_markdown}\n=== END DRAFT ==="
                )
                
                if is_proxy_claude:
                    LOGGER.info("Using chunked generation for proxy Claude to bypass timeout limit.")
                    p1 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY refine Sections 1, 2, 3, 4, and 5 (Introduction, How We Read This List, Quick Picks, Buying Guide, Comparison Table). You MUST STOP after Section 5. Do NOT output Section 6 (Deep Reviews) or Section 7. Start your response directly with Section 1."
                    LOGGER.info("Generating Chunk 1/3 (Intro -> Comparison Table)")
                    body1 = self._generate_openai(self.SYSTEM_PROMPT, p1)
                    
                    p2 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY refine Section 6 (Deep Reviews) for the FIRST 5 products in the draft (or all products if there are fewer than 5). Do NOT generate Sections 1-5. Do NOT generate reviews for products 6-10. Do NOT generate Section 7. Start your response directly with the '## 6. Deep Reviews' header, then write the reviews."
                    LOGGER.info("Generating Chunk 2/3 (Deep Reviews 1-5)")
                    body2 = self._generate_openai(self.SYSTEM_PROMPT, p2)
                    
                    p3 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY refine Section 6 (Deep Reviews) for the REMAINING products in the draft (products 6+, if any), followed by Section 7 (Final Summary). Do NOT generate Sections 1-5. Do NOT generate reviews for the first 5 products. Do NOT output the '## 6. Deep Reviews' header again. Start your response directly with the review for the next product, or Section 7 if there are no remaining products."
                    LOGGER.info("Generating Chunk 3/3 (Deep Reviews 6-10 + Final Summary)")
                    body3 = self._generate_openai(self.SYSTEM_PROMPT, p3)
                    body = f"{body1}\n\n{body2}\n\n{body3}"
                else:
                    body = self._generate_openai(self.SYSTEM_PROMPT, refinement_prompt)

        elif is_proxy_claude:
            LOGGER.info("Using chunked generation for proxy Claude to bypass timeout limit.")
            
            p1 = user_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY generate Sections 1, 2, 3, 4, and 5 (Introduction, How We Read This List, Quick Picks, Buying Guide, Comparison Table). You MUST STOP after Section 5. Do NOT output Section 6 (Deep Reviews) or Section 7. Start your response directly with Section 1."
            LOGGER.info("Generating Chunk 1/3 (Intro -> Comparison Table)")
            body1 = self._generate_openai(self.SYSTEM_PROMPT, p1)
            
            p2 = user_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY generate Section 6 (Deep Reviews) for the FIRST 5 products in the JSON list (or all products if there are fewer than 5). Do NOT generate Sections 1-5. Do NOT generate reviews for products 6-10. Do NOT generate Section 7. Start your response directly with the '## 6. Deep Reviews' header, then write the reviews."
            LOGGER.info("Generating Chunk 2/3 (Deep Reviews 1-5)")
            body2 = self._generate_openai(self.SYSTEM_PROMPT, p2)
            
            p3 = user_prompt + "\n\nCRITICAL DIRECTIVE: You must ONLY generate Section 6 (Deep Reviews) for the REMAINING products in the JSON list (products 6+, if any), followed by Section 7 (Final Summary). Do NOT generate Sections 1-5. Do NOT generate reviews for the first 5 products. Do NOT output the '## 6. Deep Reviews' header again. Start your response directly with the review for the next product, or Section 7 if there are no remaining products."
            LOGGER.info("Generating Chunk 3/3 (Deep Reviews 6-10 + Final Summary)")
            body3 = self._generate_openai(self.SYSTEM_PROMPT, p3)
            
            body = f"{body1}\n\n{body2}\n\n{body3}"

        elif is_native_anthropic:
            try:
                body = self._generate_claude(self.SYSTEM_PROMPT, user_prompt)
            except Exception as exc:
                LOGGER.warning("Claude messages streaming failed: %s; falling back to chat/completions", exc)
                body = self._generate_openai(self.SYSTEM_PROMPT, user_prompt)
        else:
            body = self._generate_openai(self.SYSTEM_PROMPT, user_prompt)

        body = self._enforce_clean_amazon_links(body)
        body = self._sanitize_external_links(body)
        body = self._sanitize_exact_prices(body)
        return self._sanitize_unsupported_vision_claims(body)

    def _generate_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call /v1/messages for Claude models using requests with streaming.

        Streaming keeps the connection alive so the proxy won't 504 timeout
        while Claude is generating a long article.
        """
        import requests as _requests

        base = (self.base_url or "https://api.anthropic.com/v1").rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        url = f"{base}/messages"

        LOGGER.info("Calling %s via requests (streaming)", url)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        resp = self._post_with_retries(
            _requests,
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
        )
        if resp.status_code != 200:
            LOGGER.error("API error %s: %s", resp.status_code, resp.text[:2000])
            resp.raise_for_status()

        # Parse SSE stream to assemble full text.
        chunks: list[str] = []
        event_types: list[str] = []
        sample_events: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            if len(sample_events) < 8:
                sample_events.append(data_str[:1000])
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if event.get("type"):
                event_types.append(event["type"])
            # Anthropic stream: content_block_delta events carry text
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
            elif event.get("type") == "error":
                raise RuntimeError(f"Claude stream error event: {event}")
            # OpenAI-compatible stream format fallback
            elif "choices" in event:
                for choice in event["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        chunks.append(delta["content"])
            elif isinstance(event.get("content"), list):
                for part in event["content"]:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        chunks.append(part["text"])
            elif isinstance(event.get("content"), str) and event["content"]:
                chunks.append(event["content"])

        if not chunks:
            LOGGER.error("Claude streaming returned no content. Event types: %s", event_types[:50])
            LOGGER.error("Claude streaming sample events: %s", sample_events)
            raise RuntimeError("Claude streaming returned no content")

        LOGGER.info("Received %d stream chunks, total ~%d chars", len(chunks), sum(len(c) for c in chunks))
        return "".join(chunks).strip()

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call /v1/chat/completions using the OpenAI SDK (for non-Claude models)."""
        if self.base_url:
            return self._generate_openai_streaming(system_prompt, user_prompt)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai") from exc

        client_kwargs: dict[str, str] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if not self.base_url and hasattr(client, "responses"):
            response = client.responses.create(model=self.model, input=messages)
            return response.output_text.strip()
        else:
            response = client.chat.completions.create(model=self.model, messages=messages)
            return response.choices[0].message.content.strip()

    def _generate_openai_streaming(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible /v1/chat/completions with stream=True.

        Third-party gateways often time out long non-streaming article requests
        at 60-120 seconds. Streaming keeps the connection active while the model
        writes the article.
        """
        import requests as _requests

        base = (self.base_url or "").rstrip("/")
        if not base.endswith("/v1") and "openai" not in base.lower():
            base += "/v1"
        url = f"{base}/chat/completions"

        LOGGER.info("Calling %s via requests (streaming chat/completions)", url)
        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = self._post_with_retries(
            _requests,
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            payload=payload,
        )
        if resp.status_code != 200:
            LOGGER.error("API error %s: %s", resp.status_code, resp.text[:2000])
            resp.raise_for_status()

        resp.encoding = "utf-8"

        chunks: list[str] = []
        tool_call_chunks: list[str] = []
        sample_events: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):].strip()
            if data_str == "[DONE]":
                break
            if len(sample_events) < 8:
                sample_events.append(data_str[:1000])
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "choices" in event:
                for choice in event["choices"]:
                    delta = choice.get("delta", {})
                    message = choice.get("message", {})
                    if delta.get("tool_calls"):
                        tool_call_chunks.append(json.dumps(delta["tool_calls"], ensure_ascii=False)[:1000])
                    if delta.get("content"):
                        chunks.append(delta["content"])
                    elif message.get("content"):
                        chunks.append(message["content"])
            elif isinstance(event.get("content"), str) and event["content"]:
                chunks.append(event["content"])

        if not chunks:
            if tool_call_chunks:
                LOGGER.error("Model returned tool_calls instead of article text. Tool-call chunks: %s", tool_call_chunks[:8])
                raise RuntimeError("Model returned tool_calls instead of article text; tool_choice=none may not be honored by this provider")
            LOGGER.error("Streaming chat/completions returned no content. Sample events: %s", sample_events)
            raise RuntimeError("Streaming chat/completions returned no content")

        LOGGER.info("Received %d chat stream chunks, total ~%d chars", len(chunks), sum(len(c) for c in chunks))
        return "".join(chunks).strip()

    @staticmethod
    def _post_with_retries(_requests, url: str, headers: dict[str, str], payload: dict[str, Any]):
        last_resp = None
        for attempt in range(1, 4):
            resp = _requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=600,
                stream=True,
            )
            last_resp = resp
            if resp.status_code < 500:
                return resp
            LOGGER.warning("API returned %s on attempt %s/3; retrying", resp.status_code, attempt)
            try:
                resp.close()
            except Exception:
                pass
            import time as _time
            _time.sleep(10 * attempt)
        return last_resp

    @staticmethod
    def _related_articles_json(related_articles: list[PublishedArticle]) -> str:
        payload = [
            {
                "title": article.title,
                "url": article.url,
                "pet_type": article.pet_type,
                "category_name": article.category_name,
                "category_path": article.category_path,
            }
            for article in related_articles
        ]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _prepare_products_for_prompt(cls, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt_products: list[dict[str, Any]] = []
        for product in products:
            item = dict(product)
            raw_price = item.pop("price", None)
            item["price_tier"] = cls._price_to_tier(raw_price)
            prompt_products.append(cls._compact_prompt_product(item))
        return prompt_products

    @classmethod
    def _compact_prompt_product(cls, item: dict[str, Any]) -> dict[str, Any]:
        limits = {
            "title": 220,
            "customers_say": 700,
            "reviews": 1000,
            "features": 900,
            "star_distribution": 500,
            "ai_vision_report": 900,
        }
        keep_keys = (
            "asin",
            "title",
            "rating",
            "review_count",
            "image_url",
            "customers_say",
            "star_distribution",
            "reviews",
            "features",
            "ai_vision_report",
            "price_tier",
        )
        compact = {}
        for key in keep_keys:
            value = item.get(key)
            if value in (None, "", [], {}):
                continue
            compact[key] = cls._clip_jsonish(value, limits.get(key, 600))
        return compact

    @classmethod
    def _clip_jsonish(cls, value: Any, limit: int) -> Any:
        if isinstance(value, str):
            return cls._clip_text(value, limit)
        if isinstance(value, list):
            return [cls._clip_jsonish(item, max(160, limit // 4)) for item in value[:6]]
        if isinstance(value, dict):
            clipped = {}
            for key, subvalue in list(value.items())[:12]:
                clipped[key] = cls._clip_jsonish(subvalue, max(160, limit // 4))
            return clipped
        return value

    @staticmethod
    def _clip_text(value: str, limit: int) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        if len(value) <= limit:
            return value
        return value[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _price_to_tier(raw_price: Any) -> str:
        if raw_price in (None, "", []):
            return "Price varies"
        price_text = json.dumps(raw_price, ensure_ascii=False) if not isinstance(raw_price, str) else raw_price
        numbers = [float(value.replace(",", "")) for value in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", price_text)]
        if not numbers:
            return "Price varies"
        price = min(numbers)
        if price < 20:
            return "$ / Budget-friendly"
        if price < 50:
            return "$$ / Mid-range"
        return "$$$ / Premium price"

    @staticmethod
    def _enforce_clean_amazon_links(markdown: str) -> str:
        def clean_url(match: re.Match[str]) -> str:
            label, url = match.group(1), match.group(2)
            parsed = urlparse(url)
            if "amazon." not in parsed.netloc:
                return match.group(0)
            asin_match = re.search(r"/dp/([A-Z0-9]{10})", parsed.path)
            if not asin_match:
                return match.group(0)
            return f"[{label}](https://www.amazon.com/dp/{asin_match.group(1)})"

        return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", clean_url, markdown)

    @staticmethod
    def _sanitize_external_links(markdown: str) -> str:
        allowed_domains = (
            "akc.org",
            "aspca.org",
            "vcahospitals.com",
            "merckvetmanual.com",
            "avma.org",
            "vet.cornell.edu",
            "pubmed.ncbi.nlm.nih.gov",
        )

        def sanitize(match: re.Match[str]) -> str:
            label, url = match.group(1), match.group(2).strip()
            if url.startswith(("/", "#")):
                return match.group(0)

            parsed = urlparse(url)
            host = parsed.netloc.lower().removeprefix("www.")
            if "amazon." in host:
                asin_match = re.search(r"/dp/([A-Z0-9]{10})", parsed.path)
                if asin_match:
                    return f"[{label}](https://www.amazon.com/dp/{asin_match.group(1)})"
                return label

            if any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains):
                return match.group(0)

            return label

        return re.sub(r"(?<!!)\[([^\]]+)\]\((https?://[^)]+|/[^)]+|#[^)]+)\)", sanitize, markdown)

    @staticmethod
    def _sanitize_exact_prices(markdown: str) -> str:
        markdown = re.sub(
            r"(?i)(?:US\$|\$)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?",
            "budget tier",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\bUSD\s*\d+(?:,\d{3})*(?:\.\d{1,2})?",
            "budget tier",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:sale|discount|deal|coupon|was|now)\b[^.\n]*(?:US\$|\$|USD)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?",
            "price may vary",
            markdown,
        )
        return markdown

    @staticmethod
    def _sanitize_unsupported_vision_claims(markdown: str) -> str:
        markdown = re.sub(
            r"(?i)\b(?:we|our team|our ai(?: visual)? scanner)\s+(?:scanned|analyzed|reviewed)\s+"
            r"(?:over\s+|more than\s+)?\d[\d,]*\s+(?:customer\s+)?(?:photos|images|pictures|reviews)\b",
            "we reviewed the available customer-summary signals",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:over\s+|more than\s+)?\d[\d,]*\s+(?:real\s+)?(?:customer\s+)?(?:photos|images|pictures)\b",
            "customer image signals",
            markdown,
        )
        markdown = re.sub(
            r"(?i)\b(?:scanned|analyzed|reviewed)\s+(?:over\s+|more than\s+)?\d[\d,]*\s+"
            r"(?:raw\s+|customer\s+|original\s+)?reviews\b",
            "reviewed the available customer-summary signals",
            markdown,
        )
        return markdown


class MarkdownExporter:
    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir

    def export(self, task: CategoryTask, markdown_body: str) -> tuple[Path, str, str]:
        target_dir = self.output_dir / task.pet_type
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"best-{slugify(task.category_name)}.md"
        target_path = target_dir / filename
        title = self.title_for(task)
        frontmatter = self._frontmatter(task, title)
        target_path.write_text(f"{frontmatter}\n\n{markdown_body.strip()}\n", encoding="utf-8")
        LOGGER.info("Wrote Markdown: %s", target_path)
        return target_path, self.url_for(task), title

    @staticmethod
    def title_for(task: CategoryTask) -> str:
        return f"Best {task.category_name} for {task.pet_type.title()}"

    @staticmethod
    def url_for(task: CategoryTask) -> str:
        return f"/{task.pet_type}/best-{slugify(task.category_name)}/"

    @staticmethod
    def _frontmatter(task: CategoryTask, title: str) -> str:
        description = (
            f"Compare popular {task.category_name.lower()} for {task.pet_type}, with buyer cautions, "
            "best-fit scenarios, and practical tips before you buy."
        )
        path_tags = [
            part.strip().lower()
            for part in task.category_path.split(">")
            if part.strip() and part.strip().lower() not in {"pet supplies", task.pet_type}
        ]
        tags = list(dict.fromkeys([task.pet_type, task.category_name.lower(), *path_tags, "pet supplies"]))
        yaml_tags = "\n".join(f"  - {MarkdownExporter._yaml_quote(tag)}" for tag in tags)
        return (
            "---\n"
            f"title: {MarkdownExporter._yaml_quote(title)}\n"
            f"description: {MarkdownExporter._yaml_quote(description)}\n"
            f"slug: {MarkdownExporter._yaml_quote(f'best-{slugify(task.category_name)}')}\n"
            f'date: "{utc_now()}"\n'
            f'lastmod: "{utc_now()}"\n'
            "draft: false\n"
            "categories:\n"
            f"  - {MarkdownExporter._yaml_quote(task.pet_type)}\n"
            "tags:\n"
            f"{yaml_tags}\n"
            f"pet_type: {MarkdownExporter._yaml_quote(task.pet_type)}\n"
            f"amazon_node_id: {MarkdownExporter._yaml_quote(task.node_id)}\n"
            f"category_path: {MarkdownExporter._yaml_quote(task.category_path)}\n"
            "---"
        )

    @staticmethod
    def _yaml_quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)


def run_pipeline(args: argparse.Namespace) -> int:
    load_dotenv()
    task_manager = TaskManager(args.tracking_json)
    scraper = ScraperEngine(
        timeout_seconds=args.timeout,
        bestsellers_command_template=args.autocli_bestsellers_command,
        product_command_template=args.autocli_product_command,
    )
    generator = ContentGenerator(model=args.model, base_url=args.base_url, api_key=args.api_key)
    exporter = MarkdownExporter()

    try:
        if args.reset_processing:
            reset_count = task_manager.reset_processing_to_pending()
            LOGGER.info("Reset %s processing categories to pending", reset_count)
        if args.retry_failed:
            reset_count = task_manager.reset_failed_to_pending()
            LOGGER.info("Reset %s failed categories to pending", reset_count)

        task_manager.sync_category_tree(args.category_json)
        batch = task_manager.get_next_batch(limit=args.batch_size)
        if not batch:
            LOGGER.info("No pending categories found. Nothing to do.")
            return 0

        for task in batch:
            try:
                related_articles = task_manager.get_related_articles(task, limit=args.related_limit)
                products = scraper.scrape_category(task, top_n=args.top_n, min_success=args.min_products)
                markdown = generator.generate(task, products, related_articles=related_articles)
                article_path, article_url, title = exporter.export(task, markdown)
                task_manager.mark_completed(task.node_id, article_path=article_path, article_url=article_url, title=title)
                LOGGER.info("Completed category %s", task.node_id)
            except Exception as exc:
                LOGGER.exception("Failed category %s: %s", task.node_id, exc)
                task_manager.mark_failed(task.node_id)
        return 0
    finally:
        task_manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drip-feed Amazon pet content pipeline")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of pending leaf categories to process")
    parser.add_argument("--category-json", type=Path, default=default_category_json(), help="Path to my_output.json")
    parser.add_argument("--tracking-json", type=Path, default=TRACKING_JSON, help="Git-trackable status file path")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL"), help="LLM model name")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"), help="LLM API key; defaults also check DASHSCOPE_API_KEY and OPENAI_API_KEY")
    parser.add_argument("--top-n", type=int, default=20, help="Top ASINs to fetch from each bestseller page")
    parser.add_argument("--min-products", type=int, default=10, help="Minimum successful product details per category")
    parser.add_argument("--related-limit", type=int, default=5, help="Completed articles to offer as internal-link candidates")
    parser.add_argument(
        "--autocli-bestsellers-command",
        default=os.environ.get("AUTOCLI_BESTSELLERS_COMMAND"),
        help='Command template for bestseller JSON, e.g. "autocli amazon bestsellers {url} -f json"',
    )
    parser.add_argument(
        "--autocli-product-command",
        default=os.environ.get("AUTOCLI_PRODUCT_COMMAND"),
        help='Command template for product JSON, e.g. "autocli amazon product {asin} -f json"',
    )
    parser.add_argument("--timeout", type=int, default=180, help="autocli timeout in seconds per request")
    parser.add_argument(
        "--reset-processing",
        action="store_true",
        help="Reset stuck processing rows back to pending before claiming a new batch",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset failed rows back to pending before claiming a new batch",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

