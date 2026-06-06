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
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import tempfile
import time
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
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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
    source_path: Path | None = None


@dataclass(frozen=True)
class AuthorityResource:
    title: str
    url: str
    note: str
    pet_type: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SEOTopic:
    title: str
    description: str
    keywords: tuple[str, ...]
    faqs: tuple[tuple[str, str], ...]


class AutoCLITimeoutError(TimeoutError):
    def __init__(self, message: str, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def configure_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
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


class SEOTopicCatalog:
    TOPICS: dict[tuple[str, str], SEOTopic] = {
        ("cats", "apparel"): SEOTopic(
            title="Best Cat Recovery Suits and Costumes for Safe Short-Term Wear",
            description="Compare cat recovery suits, sweaters, costumes, and photo props by fit, escape risk, fabric comfort, and safety for supervised wear.",
            keywords=("cat recovery suit", "cat costume safety", "cat sweater", "post spay cat suit", "cat apparel sizing"),
            faqs=(
                ("What cat apparel is safest after spay surgery?", "A soft recovery suit with stretch fabric, secure closures, and enough room for normal movement is usually safer than a decorative costume. Measure chest girth and body length before buying."),
                ("Can cats wear costumes all day?", "No. Costumes, hats, wings, and birthday props should be supervised, short-session items because strings, elastic, sequins, and tight necklines can become safety hazards."),
                ("Why do so many cat clothes fit badly?", "Many pet outfits are patterned around small dogs, while cats have longer spines and more flexible shoulders. Size charts based only on weight are a red flag."),
            ),
        ),
        ("cats", "bed-blankets"): SEOTopic(
            title="Best Cat Blankets for Waterproof Couch Protection and Cozy Beds",
            description="Find washable cat blankets for urine protection, hairballs, kneading, carriers, and senior cats, with fabric and drying cautions.",
            keywords=("waterproof cat blanket", "cat blanket for couch", "washable pet blanket", "cat urine blanket", "blanket for senior cats"),
            faqs=(
                ("What is the best cat blanket for urine accidents?", "Look for a true waterproof inner layer such as TPU, not just fleece marketed as water resistant. For repeat accidents, buy two so one can dry while the other protects the sofa or bed."),
                ("Are sherpa cat blankets safe for kneading?", "Short-pile fleece and tightly woven sherpa are safer than loose loops that can catch claws. If your cat kneads aggressively, inspect seams and backing after each wash."),
                ("Why do waterproof cat blankets take so long to dry?", "The waterproof membrane traps water during the spin cycle. Use cold wash, run an extra spin cycle, and air dry when the care label warns against dryer heat."),
            ),
        ),
        ("cats", "disposable-litter-boxes"): SEOTopic(
            title="Best Disposable Litter Boxes for Travel, Kittens, and Easy Cleanup",
            description="Compare disposable cat litter boxes for travel, foster kittens, odor control, jumbo sizing, and short-term backup use.",
            keywords=("disposable litter box", "travel litter box for cats", "cardboard litter box", "jumbo disposable litter box", "kitten litter box"),
            faqs=(
                ("Are disposable litter boxes good for travel?", "Yes, for short trips, hotels, foster setups, and emergency backups. Check the folded size, wall height, and leak resistance before relying on one for multi-day travel."),
                ("Can a large cat use a disposable litter box?", "Only if the listed dimensions are genuinely large enough. Many disposable boxes labeled large are too short for Maine Coons, Ragdolls, or big domestic shorthairs."),
                ("How long does a disposable litter box last?", "It depends on urine volume, litter type, and coating quality. For healthy adult cats, many are short-term tools, not month-long replacements for a sturdy plastic or steel box."),
            ),
        ),
        ("cats", "enclosures"): SEOTopic(
            title="Best Cat Enclosures for Indoor Safety, Catios, and Apartment Balconies",
            description="Compare indoor cat enclosures, outdoor catios, balcony-safe pens, and portable habitats by escape risk, weathering, and space.",
            keywords=("cat enclosure", "outdoor catio", "cat balcony enclosure", "indoor cat pen", "escape proof cat enclosure"),
            faqs=(
                ("What type of cat enclosure is best for an apartment balcony?", "A balcony setup needs secure mesh, no climb-out gaps, weather-safe anchoring, and enough clearance from railings. Never rely on decorative netting without testing tension and attachment points."),
                ("Can a cat enclosure replace supervised outdoor time?", "It can reduce risk, but it still needs shade, water, escape checks, and temperature monitoring. Outdoor time is safest when the enclosure is inspected before each use."),
                ("What should I avoid in a cheap cat enclosure?", "Avoid weak zippers, flexible frames that collapse when leaned on, untreated wood for outdoor use, and mesh panels large enough for paws or heads to push through."),
            ),
        ),
        ("cats", "litter-waste-receptacle-refills"): SEOTopic(
            title="Best Cat Litter Waste Receptacle Refills for Odor Control",
            description="Compare cat litter disposal refill bags and cartridges by odor sealing, fit compatibility, scent, thickness, and cost per change.",
            keywords=("litter waste receptacle refills", "cat litter disposal bags", "litter genie refill alternative", "odor control litter bags", "cat waste refill bags"),
            faqs=(
                ("Do generic litter receptacle refills fit brand-name pails?", "Some do, but fit depends on ring shape, bag width, and cartridge style. Match the exact model name before buying a value pack."),
                ("Are scented litter waste refills better for odor?", "Not always. Scent can mask odor briefly, but thick film, tight seals, and frequent emptying matter more. Sensitive cats may dislike strong fragrance near the litter area."),
                ("How do I reduce litter pail refill costs?", "Compare bag length, number of changes per roll, and whether the refill wastes extra plastic at each knot. The cheapest pack is not always cheapest per use."),
            ),
        ),
        ("cats", "litter-waste-receptacles"): SEOTopic(
            title="Best Cat Litter Waste Receptacles for Odor Control and Small Spaces",
            description="Find cat litter waste pails for odor control, multi-cat homes, apartments, and easy scooping, with refill and sealing cautions.",
            keywords=("cat litter waste receptacle", "litter disposal system", "cat litter pail odor control", "litter genie alternative", "small apartment litter pail"),
            faqs=(
                ("What makes a litter waste receptacle control odor better?", "A tight lid, sealed bag path, durable refill film, and frequent emptying matter more than marketing claims. Thin bags and loose trap doors leak smell quickly."),
                ("Is a litter pail worth it for one cat?", "It can be worth it in small apartments, bathrooms, or bedrooms where daily trash trips are annoying. If the box is near an outdoor trash can, a pail may be less necessary."),
                ("What should multi-cat homes look for?", "Prioritize capacity, refill cost, and one-handed scooping. A tiny pail can fill too fast and become smellier than a regular covered trash can."),
            ),
        ),
        ("cats", "playpens"): SEOTopic(
            title="Best Cat Playpens for Kittens, Travel, and Recovery",
            description="Compare portable cat playpens for kittens, travel, recovery, introductions, and temporary containment by size and escape risk.",
            keywords=("cat playpen", "portable cat playpen", "kitten playpen", "cat recovery pen", "travel playpen for cats"),
            faqs=(
                ("What size cat playpen do I need for a kitten?", "Choose enough floor space for food, water, a small litter box, and a resting spot. A tiny tent works for short supervision but not all-day confinement."),
                ("Can a playpen help after cat surgery?", "Yes, a stable playpen can limit jumping during recovery, but ask your vet about activity limits and make sure the cat cannot climb the mesh or snag stitches."),
                ("Why are pop-up cat playpens hard to fold?", "Most use a twisted wire-frame design. If daily setup matters, choose a model with clearer folding instructions, a larger storage bag, or a framed design instead."),
            ),
        ),
        ("cats", "replacement-filters"): SEOTopic(
            title="Best Cat Litter Box Replacement Filters for Odor Control",
            description="Compare cat litter box carbon filters by compatibility, odor control, cut-to-fit design, thickness, and refill value.",
            keywords=("cat litter box replacement filters", "activated carbon litter filter", "litter box odor filter", "hooded litter box filter", "cat litter filter refill"),
            faqs=(
                ("How often should cat litter box filters be replaced?", "Most carbon filters need replacement when odor returns, often every few weeks to a month depending on humidity, box type, and the number of cats."),
                ("Are cut-to-fit carbon filters worth it?", "They can be good value if the sheet is thick enough and easy to trim. Measure the filter slot first so the cut piece does not sag or block ventilation."),
                ("Do litter filters fix a dirty litter box smell?", "No. Filters help with airborne odor, but they cannot replace scooping, washing the box, or changing litter when ammonia builds up."),
            ),
        ),
        ("cats", "scratching-posts"): SEOTopic(
            title="Best Cat Scratching Posts That Do Not Tip Over",
            description="Compare tall cat scratching posts for large cats, kittens, couch protection, sisal durability, and stable bases.",
            keywords=("cat scratching post that does not tip", "tall scratching post", "sisal cat scratching post", "scratching post for large cats", "save couch from cat scratching"),
            faqs=(
                ("How tall should a cat scratching post be?", "Most adult cats need a post around 30 inches or taller so they can fully stretch. Short posts are usually better for kittens or horizontal scratching preferences."),
                ("Is sisal rope or sisal fabric better?", "Woven sisal fabric often lasts longer and gives steady resistance, while sisal rope is common and cheaper but can unravel under heavy daily scratching."),
                ("Why does my cat ignore the scratching post?", "The post may be too short, too wobbly, or placed away from the furniture your cat already targets. Move it near the problem spot and stabilize the base."),
            ),
        ),
        ("cats", "self-cleaning-litter-boxes"): SEOTopic(
            title="Best Self-Cleaning Litter Boxes for Large Cats and Multi-Cat Homes",
            description="Compare automatic litter boxes for large cats, multi-cat homes, odor control, safety sensors, app features, and maintenance.",
            keywords=("self cleaning litter box", "automatic litter box for large cats", "multi cat automatic litter box", "safe self cleaning litter box", "litter robot alternative"),
            faqs=(
                ("Are self-cleaning litter boxes safe for kittens?", "Use caution. Many automatic boxes have minimum weight limits or kitten modes because tiny cats may not trigger sensors reliably."),
                ("What matters most for large cats?", "Interior space, entry height, drum opening, and waste drawer capacity matter more than the outside dimensions. A big shell can still have a cramped usable area."),
                ("Do automatic litter boxes eliminate odor?", "They reduce waste exposure time, but odor still depends on drawer sealing, litter type, filter design, and how often the waste drawer is emptied."),
            ),
        ),
        ("cats", "standard-litter-boxes"): SEOTopic(
            title="Best Standard Litter Boxes for Large Cats, Seniors, and High Sprayers",
            description="Compare open, hooded, high-sided, and stainless steel litter boxes for large cats, senior access, odor, and scatter control.",
            keywords=("standard litter box", "litter box for large cats", "high sided litter box", "stainless steel cat litter box", "senior cat litter box"),
            faqs=(
                ("What size litter box is best for a large cat?", "A practical rule is at least 1.5 times the cat's body length from nose to tail base. Marketing labels like XL are less useful than actual dimensions."),
                ("Are stainless steel litter boxes better than plastic?", "Stainless steel resists odor absorption and scratching better than plastic, but it costs more and may still include plastic lids or shields that can crack."),
                ("Should senior cats use high-sided boxes?", "Senior cats often need a low front entry, even if the back and sides are high. Arthritis or mobility changes can make tall entry walls a problem."),
            ),
        ),
        ("cats", "trees"): SEOTopic(
            title="Best Cat Trees for Large Cats, Maine Coons, and Small Apartments",
            description="Compare sturdy cat trees for large cats, Maine Coons, multi-cat homes, small spaces, sisal durability, and stable bases.",
            keywords=("cat tree for large cats", "Maine Coon cat tree", "sturdy cat tree", "cat tree for small apartment", "multi cat tree"),
            faqs=(
                ("What cat tree is best for a Maine Coon?", "Look for a wide base, large platforms, thick posts, and weight capacity that makes sense for jumping force, not just resting weight."),
                ("How do I know if a cat tree will wobble?", "Check the base footprint, post thickness, total height, and whether reviews mention tipping during jumps. Tall towers with narrow bases are risky."),
                ("Are small cat trees worth buying?", "Small trees can work for kittens, seniors, or compact rooms, but active adult cats usually need taller vertical territory and a full-stretch scratching surface."),
            ),
        ),
        ("cats", "window-perches"): SEOTopic(
            title="Best Cat Window Perches for Heavy Cats and Narrow Windows",
            description="Compare cat window perches for heavy cats, suction-free setups, narrow sills, washable covers, winter glass, and bird watching.",
            keywords=("cat window perch", "window perch for heavy cats", "suction free cat window perch", "cat hammock for window", "cat perch for narrow sill"),
            faqs=(
                ("Are suction cup cat window perches safe?", "They can be safe for average cats if installed perfectly, but cold glass, dirty windows, aging cups, and heavy jumpers increase failure risk."),
                ("What is better than suction cups for heavy cats?", "A sill-mounted perch with metal hooks or a freestanding cat tree near the window is usually safer for large cats and aggressive jumpers."),
                ("Will a window perch fit modern windows?", "Not always. Measure sill depth, track slot depth, blind clearance, and whether the window opens vertically or horizontally before buying."),
            ),
        ),
        ("dogs", "air-dried"): SEOTopic(
            title="Best Air-Dried Dog Food for Picky Eaters and Sensitive Stomachs",
            description="Compare air-dried dog foods by protein, texture, transition risk, topper value, smell, and fit for picky or sensitive dogs.",
            keywords=("air dried dog food", "air dried dog food for sensitive stomach", "air dried dog food topper", "dog food for picky eaters", "limited ingredient dog food"),
            faqs=(
                ("Is air-dried dog food good for sensitive stomachs?", "It can be, but the high protein density means slow transition matters. Start with a small topper amount and watch stool quality before increasing."),
                ("Can air-dried food replace kibble?", "Some products are complete diets, while others work better as toppers or treats. Check feeding guidelines and cost per day before switching fully."),
                ("Why do some dogs refuse air-dried food?", "Texture, smell, protein source, and fat level vary widely. Picky dogs may love one recipe and reject another from the same category."),
            ),
        ),
        ("dogs", "ball-launchers"): SEOTopic(
            title="Best Dog Ball Launchers for Large Dogs, Small Dogs, and Tired Arms",
            description="Compare manual and automatic dog ball launchers by ball size, throw distance, training needs, durability, and dog size.",
            keywords=("dog ball launcher", "automatic dog ball launcher", "ball launcher for large dogs", "small dog ball launcher", "Chuckit launcher"),
            faqs=(
                ("Are automatic ball launchers worth it?", "They help when a dog can learn to reload and play safely, but they are not a substitute for supervision, training, or matching ball size to the dog."),
                ("What ball size is safe for large dogs?", "Large dogs need balls too large to swallow. Do not use small-dog or mini launcher balls with big breeds, even if the machine accepts them."),
                ("Is a manual launcher better than an automatic one?", "Manual launchers are cheaper, quieter, and more reliable outdoors. Automatic launchers are more about owner convenience and indoor repetition."),
            ),
        ),
        ("dogs", "bully-sticks"): SEOTopic(
            title="Best Bully Sticks for Aggressive Chewers, Puppies, and Odor Control",
            description="Compare bully sticks by thickness, odor, digestibility, supervision needs, puppy fit, calories, and aggressive-chewer value.",
            keywords=("bully sticks for dogs", "odor free bully sticks", "bully sticks for aggressive chewers", "puppy bully sticks", "digestible dog chews"),
            faqs=(
                ("Are bully sticks safe for aggressive chewers?", "They can be, but only with supervision and the right thickness. Use a holder if your dog tries to swallow short end pieces."),
                ("Do odor-free bully sticks really have no smell?", "They usually smell less, not zero. Low-odor processing can make them more household-friendly, but dogs may prefer stronger-smelling sticks."),
                ("How often can dogs have bully sticks?", "Treat them as calorie-dense chews, not free snacks. Dogs with pancreatitis risk, weight issues, or sensitive stomachs need extra caution."),
            ),
        ),
        ("dogs", "cameras-monitors"): SEOTopic(
            title="Best Dog Cameras and Monitors for Separation Anxiety and Treat Tossing",
            description="Compare dog cameras by video quality, app reliability, barking alerts, treat tossing, subscriptions, and anxiety-monitoring fit.",
            keywords=("dog camera", "dog camera for separation anxiety", "pet camera with treat dispenser", "dog monitor no subscription", "barking alert camera"),
            faqs=(
                ("Can a dog camera help separation anxiety?", "It can help you observe patterns, but it does not treat anxiety by itself. Treat tossing can even become a problem for dogs with diet restrictions."),
                ("Do dog cameras need subscriptions?", "Some useful features, like cloud recording or smart alerts, may require paid plans. Check what works without a subscription before buying."),
                ("What matters more than video resolution?", "App stability, night vision, alert accuracy, privacy controls, and whether the treat mechanism jams matter more in daily use than headline resolution."),
            ),
        ),
        ("dogs", "crate-covers"): SEOTopic(
            title="Best Dog Crate Covers for Anxiety, Airflow, and Blackout Sleep",
            description="Compare dog crate covers by breathable fabric, blackout level, chew risk, zipper quality, crate fit, and overheating concerns.",
            keywords=("dog crate cover", "breathable dog crate cover", "blackout crate cover", "crate cover for anxious dogs", "chew proof crate cover"),
            faqs=(
                ("Do crate covers help anxious dogs?", "They can reduce visual triggers for crate-trained dogs, but they will not fix severe separation anxiety or destructive chewing by themselves."),
                ("Can a crate cover cause overheating?", "Yes. Thick blackout fabric and PVC coatings can trap heat. Leave airflow panels open and avoid fully sealing the crate in warm rooms."),
                ("How do I choose the right crate cover size?", "Measure your exact crate, including brand-specific door placement. Universal sizing often causes tight zippers or loose fabric that dogs can pull inside."),
            ),
        ),
        ("dogs", "dna-tests"): SEOTopic(
            title="Best Dog DNA Tests for Breed ID, Health Screening, and Rescue Dogs",
            description="Compare dog DNA tests for breed mix, health screening, age estimates, rescue dogs, trait reports, and vet-useful results.",
            keywords=("dog DNA test", "best dog DNA test for mixed breed", "dog breed identification test", "dog health DNA test", "rescue dog DNA test"),
            faqs=(
                ("Are dog DNA tests accurate for mixed breeds?", "Accuracy depends on the reference database and how mixed the dog is. Results are often most useful as probability ranges, not absolute certainty."),
                ("Should I choose breed ID or health screening?", "Breed ID answers curiosity and behavior context. Health screening is more useful if you want mutation, drug sensitivity, or inherited disease information to discuss with a vet."),
                ("Are dog age tests precise?", "Age estimates can be helpful for rescue dogs, but they often come with a broad range. Do not expect a precise birthday."),
            ),
        ),
        ("dogs", "enclosure-covers"): SEOTopic(
            title="Best Dog Enclosure Covers for Outdoor Kennels, Shade, and Rain",
            description="Compare dog kennel and enclosure covers by shade, airflow, rain resistance, wind security, grommet strength, and fit.",
            keywords=("dog enclosure cover", "outdoor dog kennel cover", "dog kennel shade cover", "waterproof dog pen cover", "dog run cover"),
            faqs=(
                ("What is the best cover for an outdoor dog kennel?", "Look for UV shade, secure tie-downs, sloped rain runoff, and airflow. Flat covers that collect water can sag or tear."),
                ("Can a kennel cover keep dogs warm in winter?", "It can reduce wind exposure, but it is not insulation or a substitute for safe indoor shelter during unsafe temperatures."),
                ("How do I stop an enclosure cover from tearing?", "Match the cover to frame size, keep tension even, use all tie points, and avoid letting water pool on top during storms."),
            ),
        ),
        ("dogs", "location-trackers"): SEOTopic(
            title="Best Dog GPS Trackers for Escape Artists, Hiking, and No-Subscription Needs",
            description="Compare dog GPS trackers by subscription cost, battery life, escape alerts, health metrics, rural coverage, and collar fit.",
            keywords=("dog GPS tracker", "GPS tracker for dogs no subscription", "dog location tracker", "escape alert dog tracker", "dog tracker for hiking"),
            faqs=(
                ("Do dog GPS trackers work without a subscription?", "Some use Bluetooth or radio-style tracking, but most real-time GPS and cellular alerts require a paid plan. Check coverage in your area."),
                ("What tracker is best for escape artists?", "Fast escape alerts, reliable geofencing, strong collar attachment, and battery life matter more than extra wellness metrics."),
                ("Can a dog tracker replace a microchip?", "No. A tracker helps locate a dog while it is attached and charged. A microchip is still important for identification if the collar comes off."),
            ),
        ),
        ("dogs", "prescription-medications"): SEOTopic(
            title="Best Dog Prescription Medication Services and Vet-Approved Pet Med Options",
            description="Compare dog prescription medication options by vet approval needs, pharmacy reliability, safety cautions, refills, and use cases.",
            keywords=("dog prescription medication", "pet medication online", "vet approved dog meds", "dog prescription refill", "pet pharmacy for dogs"),
            faqs=(
                ("Can I buy dog prescription medication without a vet?", "Legitimate prescription medications require veterinary authorization. Avoid sellers that bypass prescriptions or make unsupported medical claims."),
                ("What should I check before ordering pet meds online?", "Confirm pharmacy legitimacy, exact medication name and strength, expiration handling, shipping temperature needs, and whether your vet must approve the refill."),
                ("Should I switch dog medications based on online reviews?", "No. Medication changes should go through your veterinarian, especially for heart, seizure, pain, allergy, or anxiety drugs."),
            ),
        ),
        ("dogs", "styptic-gels-powders"): SEOTopic(
            title="Best Styptic Powders and Gels for Dog Nail Bleeding and Grooming Kits",
            description="Compare styptic powder, gel, and sticks for dog nail bleeding, quick-stop grooming kits, pain relief, and emergency use.",
            keywords=("styptic powder for dogs", "dog nail bleeding powder", "quick stop dog nail bleeding", "styptic gel for dogs", "dog grooming first aid"),
            faqs=(
                ("What stops dog nail bleeding fastest?", "Styptic powder usually works quickly because it coats the nail quick. Have it open before trimming if your dog has dark nails or moves suddenly."),
                ("Is styptic powder painful for dogs?", "It can sting, especially without a pain-relief ingredient. Apply firm pressure calmly and avoid rubbing the powder into surrounding skin."),
                ("Can styptic products replace a vet visit?", "No. They are for minor nail quick bleeding. Deep cuts, torn nails, repeated bleeding, or signs of infection need veterinary care."),
            ),
        ),
    }

    @classmethod
    def for_task(cls, task: CategoryTask) -> SEOTopic:
        return cls.for_values(task.pet_type, task.category_name)

    @classmethod
    def for_article(cls, article: PublishedArticle) -> SEOTopic:
        return cls.for_values(article.pet_type, article.category_name)

    @classmethod
    def for_values(cls, pet_type: str, category_name: str) -> SEOTopic:
        key = (pet_type, slugify(category_name))
        if key in cls.TOPICS:
            return cls.TOPICS[key]
        pet_label = "Dogs" if pet_type == "dogs" else "Cats"
        category_label = category_name.strip().title()
        lower_category = category_name.strip().lower()
        return SEOTopic(
            title=f"Best {category_label} for {pet_label}: Buyer Fit, Safety, and Regret Checks",
            description=(
                f"Compare {lower_category} for {pet_type} by buyer fit, safety cautions, durability, "
                "common complaints, and practical use cases."
            ),
            keywords=(
                f"best {lower_category} for {pet_type}",
                f"{lower_category} buying guide",
                f"{lower_category} reviews",
                f"{pet_type} {lower_category}",
            ),
            faqs=(
                (f"What should I check before buying {lower_category}?", "Start with fit, safety, durability, cleaning, and the most common complaint pattern instead of choosing only by rating."),
                (f"Who should skip budget {lower_category}?", "Skip the cheapest option if the product needs to handle daily use, large pets, destructive behavior, or a medical or safety-sensitive situation."),
                (f"How do I compare {lower_category} without exact prices?", "Compare the use case, failure risk, replacement cost, and whether the product solves the specific problem you are buying it for."),
            ),
        )


class SEOResourceLinker:
    """Adds crawlable, contextual article links after LLM generation."""

    INTERNAL_LINK_LIMIT = 4
    RELATED_SECTION_RE = re.compile(
        r"\n*#{2,3}[ \t]+Related Resources[ \t]*\n.*?(?=\n#{2,3}[ \t]+(?:Comparison Table|Deep Reviews|Final Summary)\b|\Z)",
        re.DOTALL,
    )
    INSERT_TARGETS = (
        re.compile(r"(?m)^#{2,3}\s+Comparison Table\b"),
        re.compile(r"(?m)^#{2,3}\s+Deep Reviews\b"),
        re.compile(r"(?m)^#{2,3}\s+Final Summary\b"),
    )
    MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\((https?://[^)]+|/[^)]+)\)")
    FRONTMATTER_RE = re.compile(r"\A(---\s*\n.*?\n---\s*\n?)(.*)\Z", re.DOTALL)

    AUTHORITY_RESOURCES = (
        AuthorityResource(
            title="ASPCA dog nutrition tips",
            url="https://www.aspca.org/pet-care/dog-care/dog-nutrition-tips",
            note="Feeding and transition guidance for diet changes.",
            pet_type="dogs",
            keywords=("food", "dry", "air-dried", "nutrition", "diet", "treat"),
        ),
        AuthorityResource(
            title="Merck Veterinary Manual on dog and cat foods",
            url="https://www.merckvetmanual.com/management-and-nutrition/nutrition-small-animals/dog-and-cat-foods",
            note="Veterinary detail on pet food labels, feeding guidelines, and diet types.",
            pet_type="dogs",
            keywords=("food", "dry", "air-dried", "nutrition", "diet", "feeding"),
        ),
        AuthorityResource(
            title="VCA Hospitals guide to chew toys and bones",
            url="https://vcahospitals.com/know-your-pet/bones-and-chew-toys",
            note="Veterinary context for chew safety and supervision.",
            pet_type="dogs",
            keywords=("bully", "stick", "chew", "toy", "ball", "launcher"),
        ),
        AuthorityResource(
            title="AKC crate training guide",
            url="https://www.akc.org/expert-advice/training/why-crate-training-is-great-for-your-dog/",
            note="Training context for crate and kennel setups.",
            pet_type="dogs",
            keywords=("crate", "kennel", "enclosure", "cover", "pen", "house"),
        ),
        AuthorityResource(
            title="VCA Hospitals pet first aid basics",
            url="https://vcahospitals.com/know-your-pet/first-aid-general-information",
            note="Veterinary first-aid context for home grooming and minor bleeding prep.",
            pet_type="dogs",
            keywords=("styptic", "grooming", "health", "medication", "prescription", "dna"),
        ),
        AuthorityResource(
            title="AKC dog health guide",
            url="https://www.akc.org/expert-advice/health/",
            note="General health background for monitoring and care decisions.",
            pet_type="dogs",
            keywords=("camera", "monitor", "tracker", "location", "health", "default"),
        ),
        AuthorityResource(
            title="ASPCA dog care basics",
            url="https://www.aspca.org/pet-care/dog-care",
            note="Practical care and behavior guidance for dog owners.",
            pet_type="dogs",
            keywords=("default",),
        ),
        AuthorityResource(
            title="VCA Hospitals litter box problems in cats",
            url="https://vcahospitals.com/know-your-pet/litter-box-problems-in-cats",
            note="Veterinary context for litter box placement, avoidance, and health flags.",
            pet_type="cats",
            keywords=("litter", "box", "waste", "receptacle", "refill", "filter"),
        ),
        AuthorityResource(
            title="Merck Veterinary Manual on cat nutrition",
            url="https://www.merckvetmanual.com/cat-owners/selecting-and-providing-a-home-for-a-cat/proper-nutrition-for-cats",
            note="Veterinary guidance on feeding standards, life stages, and balanced diets.",
            pet_type="cats",
            keywords=("food", "nutrition", "diet", "feeding", "treat"),
        ),
        AuthorityResource(
            title="Cornell Feline Health Center on destructive behavior",
            url="https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/feline-behavior-problems-destructive-behavior",
            note="Feline behavior context for scratching, kneading, bedding, and enrichment.",
            pet_type="cats",
            keywords=("bed", "blanket", "furniture", "scratch", "tree", "sofa", "perch"),
        ),
        AuthorityResource(
            title="Cornell Feline Health Center on feline dental disease",
            url="https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/feline-dental-disease",
            note="Veterinary context for dental pain, prevention, and when home care is not enough.",
            pet_type="cats",
            keywords=("dental", "teeth", "tooth", "oral", "gum"),
        ),
        AuthorityResource(
            title="Cornell Feline Health Center on fleas in cats",
            url="https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/fleas-source-torment-your-cat",
            note="Veterinary background on flea risks, treatment expectations, and home control.",
            pet_type="cats",
            keywords=("flea", "tick", "parasite"),
        ),
        AuthorityResource(
            title="VCA Hospitals on Elizabethan collars in cats",
            url="https://vcahospitals.com/chancellor/know-your-pet/elizabethan-collars-in-cats",
            note="Veterinary guidance on cone fit, recovery use, and collar alternatives.",
            pet_type="cats",
            keywords=("recovery", "cone", "elizabethan", "e-collar"),
        ),
        AuthorityResource(
            title="ASPCA general cat care",
            url="https://www.aspca.org/pet-care/cat-care/general-cat-care",
            note="General cat-care guidance covering identification, safety collars, grooming, and litter basics.",
            pet_type="cats",
            keywords=("tag", "tags", "id", "identification", "leash", "harness", "collar", "grooming", "shedding", "dander"),
        ),
        AuthorityResource(
            title="ASPCA Halloween safety tips for pets",
            url="https://www.aspca.org/pet-care/general-pet-care/halloween-safety-tips",
            note="Safety context for costumes, props, and supervised wear.",
            pet_type="cats",
            keywords=("apparel", "costume", "clothing", "bandana"),
        ),
        AuthorityResource(
            title="ASPCA cat care basics",
            url="https://www.aspca.org/pet-care/cat-care",
            note="General health and behavior guidance for cat owners.",
            pet_type="cats",
            keywords=("enclosure", "playpen", "carrier", "stroller", "door", "default"),
        ),
        AuthorityResource(
            title="Cornell Feline Health Center",
            url="https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center",
            note="Authoritative feline health information from a veterinary college.",
            pet_type="cats",
            keywords=("default",),
        ),
    )
    RELATED_TOPIC_GROUPS = (
        ("dogs", "dog feeding", frozenset(("air", "dried", "dry", "food", "nutrition", "diet", "bully", "stick"))),
        ("dogs", "dog activity", frozenset(("toy", "ball", "launcher", "chew", "bully", "stick"))),
        ("dogs", "dog crates and covers", frozenset(("crate", "kennel", "enclosure", "cover", "pen", "house"))),
        ("dogs", "dog health", frozenset(("health", "dna", "prescription", "medication", "styptic", "grooming", "tracker", "location", "camera", "monitor"))),
        ("cats", "cat litter setup", frozenset(("litter", "box", "waste", "receptacle", "refill", "filter"))),
        ("cats", "cat containment", frozenset(("enclosure", "playpen", "door", "net", "pen", "carrier", "stroller"))),
        ("cats", "cat comfort", frozenset(("bed", "blanket", "furniture", "bedding", "apparel", "clothing", "costume"))),
        ("cats", "cat scratching and enrichment", frozenset(("scratch", "scratcher", "tree", "perch", "hammock", "sofa"))),
        ("cats", "cat collars and identification", frozenset(("collar", "harness", "leash", "tag", "tags", "identification", "breakaway", "flea"))),
        ("cats", "cat health supplies", frozenset(("health", "dental", "digestive", "ear", "mites", "recovery", "cone", "relaxant", "calming"))),
        ("cats", "cat grooming and coat care", frozenset(("grooming", "shedding", "dander", "spray", "brush", "coat", "flea"))),
        ("cats", "cat feeding and hydration", frozenset(("food", "milk", "replacer", "nursing", "feeding", "water", "fountain", "bottle", "syringe"))),
    )

    @classmethod
    def enrich(
        cls,
        markdown_body: str,
        task: CategoryTask,
        related_articles: list[PublishedArticle] | None = None,
        current_url: str | None = None,
    ) -> str:
        body = cls._remove_related_resources(markdown_body)
        resource_section = cls._build_resource_section(
            task=task,
            related_articles=related_articles or [],
            body=body,
            current_url=current_url or cls._url_for(task),
        )
        return cls._insert_resource_section(body, resource_section)

    @classmethod
    def refresh_existing_content(cls, content_dir: Path = OUTPUT_DIR) -> int:
        articles = cls.collect_published_articles(content_dir)
        changed_count = 0
        for article in articles:
            if not article.source_path:
                continue
            original = article.source_path.read_text(encoding="utf-8")
            updated = cls.enrich_document(original, article, articles)
            if updated != original:
                article.source_path.write_text(updated, encoding="utf-8")
                changed_count += 1
        return changed_count

    @classmethod
    def collect_published_articles(cls, content_dir: Path = OUTPUT_DIR) -> list[PublishedArticle]:
        articles: list[PublishedArticle] = []
        for path in sorted(content_dir.rglob("*.md")):
            if path.name == "_index.md":
                continue
            document = path.read_text(encoding="utf-8")
            fields = cls._parse_frontmatter(document)
            if str(fields.get("draft", "")).lower() == "true":
                continue
            pet_type = str(fields.get("pet_type") or path.parent.name).strip().lower()
            if pet_type not in {"dogs", "cats"}:
                continue
            title = str(fields.get("title") or cls._title_from_filename(path.stem, pet_type))
            category_path = str(fields.get("category_path") or "")
            category_name = cls._category_name(title, category_path)
            slug = str(fields.get("slug") or path.stem).strip("/")
            articles.append(
                PublishedArticle(
                    title=title,
                    url=f"/{pet_type}/{slug}/",
                    pet_type=pet_type,
                    category_name=category_name,
                    category_path=category_path,
                    source_path=path,
                )
            )
        return articles

    @classmethod
    def enrich_document(cls, document: str, article: PublishedArticle, all_articles: list[PublishedArticle]) -> str:
        frontmatter, body = cls._split_frontmatter(document)
        task = CategoryTask(
            node_id="",
            category_path=article.category_path,
            category_name=article.category_name,
            bsr_url="",
            pet_type=article.pet_type,
        )
        body_without_resource_section = cls._remove_related_resources(body)
        related_articles = cls._rank_related_articles(task, all_articles, article.url, body_without_resource_section)
        updated_body = cls.enrich(body, task, related_articles=related_articles, current_url=article.url).strip()
        if not frontmatter:
            return updated_body + "\n"
        if updated_body == body.strip():
            return document
        return cls._touch_lastmod(frontmatter) + "\n\n" + updated_body + "\n"

    @classmethod
    def _build_resource_section(
        cls,
        task: CategoryTask,
        related_articles: list[PublishedArticle],
        body: str,
        current_url: str,
    ) -> str:
        internal_links = cls._rank_related_articles(task, related_articles, current_url, body)
        lines = []
        for article in internal_links[: cls.INTERNAL_LINK_LIMIT]:
            lines.append(
                f"- **Related Review:** [{article.title}]({article.url}) - "
                f"{cls._internal_link_note(task, article)}"
            )

        authority = cls._select_authority_resource(task, body)
        lines.append(
            f"- **Authority Reference:** [{authority.title}]({authority.url}) - {authority.note}"
        )
        return "## Related Resources\n\n" + "\n".join(lines)

    @classmethod
    def _rank_related_articles(
        cls,
        task: CategoryTask,
        related_articles: list[PublishedArticle],
        current_url: str,
        body: str,
    ) -> list[PublishedArticle]:
        existing_urls = cls._extract_link_urls(body)
        task_terms = cls._link_terms(f"{task.category_name} {task.category_path}")
        task_groups = cls._topic_groups(task.pet_type, task_terms)
        scored: list[tuple[int, str, PublishedArticle]] = []
        for article in related_articles:
            if article.url == current_url or article.pet_type != task.pet_type:
                continue
            article_terms = cls._link_terms(f"{article.title} {article.category_name} {article.category_path}")
            article_groups = cls._topic_groups(article.pet_type, article_terms)
            overlap = len(task_terms & article_terms)
            common_path_depth = cls._common_path_depth(task.category_path, article.category_path)
            topic_overlap = len(task_groups & article_groups)
            score = (overlap * 10) + (topic_overlap * 8) + (common_path_depth * 4)
            if article.url in existing_urls:
                score -= 3
            scored.append((score, article.title, article))

        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [article for _, _, article in scored[: cls.INTERNAL_LINK_LIMIT]]

    @classmethod
    def _select_authority_resource(cls, task: CategoryTask, body: str) -> AuthorityResource:
        haystack = f"{task.category_name} {task.category_path}".lower()
        existing_urls = cls._extract_link_urls(body)
        candidates = [resource for resource in cls.AUTHORITY_RESOURCES if resource.pet_type == task.pet_type]

        def score(resource: AuthorityResource) -> int:
            return sum(1 for keyword in resource.keywords if keyword != "default" and keyword in haystack)

        ordered = sorted(candidates, key=lambda resource: (-score(resource), "default" in resource.keywords))
        matched = [resource for resource in ordered if score(resource) > 0]
        for resource in matched:
            if resource.url not in existing_urls:
                return resource
        if matched:
            return matched[0]

        for resource in ordered:
            if "default" in resource.keywords and resource.url not in existing_urls:
                return resource

        defaults = [resource for resource in candidates if "default" in resource.keywords]
        return defaults[0] if defaults else candidates[0]

    @classmethod
    def _remove_related_resources(cls, markdown_body: str) -> str:
        return cls._collapse_blank_lines(cls.RELATED_SECTION_RE.sub("\n\n", markdown_body))

    @classmethod
    def _insert_resource_section(cls, markdown_body: str, resource_section: str) -> str:
        body = markdown_body.strip()
        for pattern in cls.INSERT_TARGETS:
            match = pattern.search(body)
            if match:
                prefix = body[: match.start()].rstrip()
                suffix = body[match.start() :].lstrip()
                return f"{prefix}\n\n{resource_section}\n\n{suffix}".strip()
        return f"{body}\n\n{resource_section}".strip()

    @classmethod
    def _parse_frontmatter(cls, document: str) -> dict[str, str]:
        match = cls.FRONTMATTER_RE.match(document)
        if not match:
            return {}
        frontmatter = match.group(1)
        fields: dict[str, str] = {}
        for line in frontmatter.splitlines():
            if not line or line == "---" or line.startswith(" ") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = cls._frontmatter_value(value.strip())
        return fields

    @classmethod
    def _split_frontmatter(cls, document: str) -> tuple[str, str]:
        match = cls.FRONTMATTER_RE.match(document)
        if not match:
            return "", document
        return match.group(1).strip(), match.group(2)

    @staticmethod
    def _frontmatter_value(value: str) -> str:
        if not value:
            return ""
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"').strip("'")
        return str(decoded)

    @staticmethod
    def _touch_lastmod(frontmatter: str) -> str:
        now = utc_now()
        if re.search(r"(?m)^lastmod:\s*.*$", frontmatter):
            return re.sub(r"(?m)^lastmod:\s*.*$", f'lastmod: "{now}"', frontmatter, count=1)
        return frontmatter.replace("\n---", f'\nlastmod: "{now}"\n---', 1)

    @staticmethod
    def _extract_link_urls(markdown_body: str) -> set[str]:
        return {match.group(1).strip() for match in SEOResourceLinker.MARKDOWN_LINK_RE.finditer(markdown_body)}

    @staticmethod
    def _link_terms(value: str) -> set[str]:
        stopwords = {
            "and",
            "best",
            "cat",
            "cats",
            "dog",
            "dogs",
            "for",
            "pet",
            "review",
            "reviews",
            "supplies",
            "treat",
            "the",
        }
        normalized_terms = set()
        for term in re.findall(r"[a-z0-9]+", value.lower()):
            if term in stopwords or len(term) <= 2:
                continue
            if term.endswith("ies") and len(term) > 4:
                term = term[:-3] + "y"
            elif term.endswith("s") and len(term) > 3:
                term = term[:-1]
            normalized_terms.add(term)
        return normalized_terms

    @classmethod
    def _topic_groups(cls, pet_type: str, terms: set[str]) -> set[str]:
        return {
            group_name
            for group_pet_type, group_name, group_terms in cls.RELATED_TOPIC_GROUPS
            if group_pet_type == pet_type and terms & group_terms
        }

    @staticmethod
    def _common_path_depth(left: str, right: str) -> int:
        ignored = {"pet supplies", "dogs", "cats"}
        left_parts = [part.strip().lower() for part in left.split(">") if part.strip() and part.strip().lower() not in ignored]
        right_parts = [part.strip().lower() for part in right.split(">") if part.strip() and part.strip().lower() not in ignored]
        depth = 0
        for left_part, right_part in zip(left_parts, right_parts):
            if left_part != right_part:
                break
            depth += 1
        return depth

    @staticmethod
    def _category_name(title: str, category_path: str) -> str:
        if category_path:
            return category_path.split(">")[-1].strip()
        match = re.match(r"Best\s+(.+?)\s+for\s+(?:Dogs|Cats)$", title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return title.removeprefix("Best ").strip()

    @staticmethod
    def _title_from_filename(stem: str, pet_type: str) -> str:
        words = stem.removeprefix("best-").replace("-", " ").title()
        return f"Best {words} for {pet_type.title()}"

    @staticmethod
    def _url_for(task: CategoryTask) -> str:
        return f"/{task.pet_type}/best-{slugify(task.category_name)}/"

    @classmethod
    def _internal_link_note(cls, task: CategoryTask, article: PublishedArticle) -> str:
        shared_context = cls._shared_context(task.category_path, article.category_path)
        if shared_context == "pet gear":
            task_groups = cls._topic_groups(task.pet_type, cls._link_terms(f"{task.category_name} {task.category_path}"))
            article_groups = cls._topic_groups(
                article.pet_type,
                cls._link_terms(f"{article.title} {article.category_name} {article.category_path}"),
            )
            shared_groups = sorted(task_groups & article_groups)
            if shared_groups:
                shared_context = shared_groups[0]
        return f"Useful when you are comparing {shared_context} fit, upkeep, safety, and long-term cost."

    @staticmethod
    def _shared_context(left: str, right: str) -> str:
        ignored = {"pet supplies", "dogs", "cats"}
        left_parts = [part.strip().lower() for part in left.split(">") if part.strip()]
        right_parts = [part.strip().lower() for part in right.split(">") if part.strip()]
        shared = [part for part in left_parts if part in right_parts and part not in ignored]
        return shared[-1] if shared else "pet gear"

    @staticmethod
    def _collapse_blank_lines(value: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", value).strip()


class SEOArticleOptimizer:
    LEADING_H1_RE = re.compile(r"\A\s*#\s+.+?\n+")
    FAQ_SECTION_RE = re.compile(
        r"\n*## Common Questions Before Buying\n.*?(?=\n#{2,3}\s+(?:Related Resources|Comparison Table|Deep Reviews|Final Summary)\b|\Z)",
        re.DOTALL,
    )
    INSERT_TARGETS = (
        re.compile(r"(?m)^#{2,3}\s+Related Resources\b"),
        re.compile(r"(?m)^#{2,3}\s+Comparison Table\b"),
        re.compile(r"(?m)^#{2,3}\s+Deep Reviews\b"),
        re.compile(r"(?m)^#{2,3}\s+Final Summary\b"),
    )

    @classmethod
    def enrich_body(cls, markdown_body: str, task: CategoryTask) -> str:
        topic = SEOTopicCatalog.for_task(task)
        return cls._insert_faq_section(markdown_body, topic)

    @classmethod
    def refresh_existing_content(cls, content_dir: Path = OUTPUT_DIR) -> int:
        articles = SEOResourceLinker.collect_published_articles(content_dir)
        changed_count = 0
        for article in articles:
            if not article.source_path:
                continue
            original = article.source_path.read_text(encoding="utf-8")
            updated = cls.enrich_document(original, article)
            if updated != original:
                article.source_path.write_text(updated, encoding="utf-8")
                changed_count += 1
        return changed_count

    @classmethod
    def enrich_document(cls, document: str, article: PublishedArticle) -> str:
        frontmatter, body = SEOResourceLinker._split_frontmatter(document)
        if not frontmatter:
            return document

        topic = SEOTopicCatalog.for_article(article)
        updated_frontmatter = cls._update_frontmatter(frontmatter, topic)
        updated_body = cls._insert_faq_section(body, topic).strip()
        if updated_frontmatter == frontmatter and updated_body == body.strip():
            return document
        updated_frontmatter = SEOResourceLinker._touch_lastmod(updated_frontmatter)
        return f"{updated_frontmatter}\n\n{updated_body}\n"

    @classmethod
    def _insert_faq_section(cls, markdown_body: str, topic: SEOTopic) -> str:
        body = cls._remove_leading_h1(markdown_body)
        body = cls.FAQ_SECTION_RE.sub("\n\n", body).strip()
        faq_section = cls._faq_section(topic)
        for pattern in cls.INSERT_TARGETS:
            match = pattern.search(body)
            if match:
                prefix = body[: match.start()].rstrip()
                suffix = body[match.start() :].lstrip()
                return f"{prefix}\n\n{faq_section}\n\n{suffix}".strip()
        return f"{body}\n\n{faq_section}".strip()

    @classmethod
    def _remove_leading_h1(cls, markdown_body: str) -> str:
        return cls.LEADING_H1_RE.sub("", markdown_body, count=1).strip()

    @staticmethod
    def _faq_section(topic: SEOTopic) -> str:
        blocks = ["## Common Questions Before Buying"]
        for question, answer in topic.faqs:
            blocks.append(f"### {question}\n\n{answer}")
        return "\n\n".join(blocks)

    @classmethod
    def _update_frontmatter(cls, frontmatter: str, topic: SEOTopic) -> str:
        lines = frontmatter.splitlines()
        lines = cls._remove_yaml_block(lines, "keywords")
        lines = cls._set_scalar(lines, "title", topic.title)
        lines = cls._set_scalar(lines, "description", topic.description)
        insert_at = cls._line_index(lines, "description")
        keyword_lines = ["keywords:", *[f"  - {MarkdownExporter._yaml_quote(keyword)}" for keyword in topic.keywords]]
        if insert_at is None:
            insert_at = 1
        lines[insert_at + 1 : insert_at + 1] = keyword_lines
        return "\n".join(lines)

    @staticmethod
    def _line_index(lines: list[str], key: str) -> int | None:
        for index, line in enumerate(lines):
            if re.match(rf"^{re.escape(key)}\s*:", line):
                return index
        return None

    @classmethod
    def _set_scalar(cls, lines: list[str], key: str, value: str) -> list[str]:
        quoted_value = MarkdownExporter._yaml_quote(value)
        index = cls._line_index(lines, key)
        if index is None:
            lines.insert(1, f"{key}: {quoted_value}")
        else:
            lines[index] = f"{key}: {quoted_value}"
        return lines

    @staticmethod
    def _remove_yaml_block(lines: list[str], key: str) -> list[str]:
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if re.match(rf"^{re.escape(key)}\s*:", line):
                index += 1
                while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                    index += 1
                continue
            output.append(line)
            index += 1
        return output


class ScraperEngine:
    def __init__(
        self,
        bestseller_cache: Path = BESTSELLER_CACHE,
        product_cache: Path = PRODUCT_CACHE,
        timeout_seconds: int = 180,
        bestsellers_command_template: str | None = None,
        product_command_template: str | None = None,
        autocli_path: str | None = None,
    ) -> None:
        self.bestseller_cache = bestseller_cache
        self.product_cache = product_cache
        self.timeout_seconds = timeout_seconds
        self.autocli_path = (autocli_path or os.environ.get("AUTOCLI_PATH") or "autocli").strip()
        autocli_command = self._quote_command_token(self.autocli_path)
        self.bestsellers_command_template = (
            bestsellers_command_template
            or os.environ.get("AUTOCLI_BESTSELLERS_COMMAND")
            or f"{autocli_command} amazon bestsellers {{url}} -f json"
        )
        self.product_command_template = (
            product_command_template
            or os.environ.get("AUTOCLI_PRODUCT_COMMAND")
            or f"{autocli_command} amazon product {{asin}} -f json"
        )
        self.bestseller_cache.mkdir(parents=True, exist_ok=True)
        self.product_cache.mkdir(parents=True, exist_ok=True)
        self._log_autocli_resolution()

    def scrape_category(self, task: CategoryTask, top_n: int = 20, min_success: int = 10) -> list[dict[str, Any]]:
        LOGGER.info("Scraping %s (%s)", task.category_name, task.node_id)
        bestseller_payload = self._cached_autocli_json(
            self._format_command(self.bestsellers_command_template, url=task.bsr_url, node_id=task.node_id),
            self.bestseller_cache / f"{safe_json_filename(task.node_id)}.json",
        )
        asins = self._extract_top_asins(bestseller_payload, limit=top_n)
        LOGGER.info("Found %s ASIN candidates for %s", len(asins), task.node_id)

        products: list[dict[str, Any]] = []
        for index, asin in enumerate(asins, start=1):
            LOGGER.info("Fetching product detail %s/%s for %s: %s", index, len(asins), task.node_id, asin)
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
            stdout, stderr = self._run_command_with_timeout(command)
        except OSError as exc:
            LOGGER.error("Failed to start AutoCLI command: %s", " ".join(command))
            LOGGER.error("AutoCLI start error: %s", exc)
            raise
        except AutoCLITimeoutError as exc:
            stdout, stderr = exc.stdout, exc.stderr
            payload = self._load_autocli_json(stdout, stderr, strict=False)
            if payload is not None:
                LOGGER.warning("AutoCLI timed out but returned valid JSON; saving cache anyway: %s", cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                LOGGER.info("Saved cache: %s", cache_path)
                return payload
            self._save_autocli_raw_output(command, cache_path, stdout, stderr, "timeout_without_json")
            raise
        except subprocess.CalledProcessError as exc:
            LOGGER.error("AutoCLI failed with exit code %s", exc.returncode)
            if exc.stdout:
                LOGGER.error("AutoCLI stdout:\n%s", str(exc.stdout)[-4000:])
            if exc.stderr:
                LOGGER.error("AutoCLI stderr:\n%s", str(exc.stderr)[-4000:])
            payload = self._load_autocli_json(str(exc.stdout or ""), str(exc.stderr or ""), strict=False)
            if payload is not None:
                LOGGER.warning("AutoCLI failed but output contained valid JSON; saving cache anyway: %s", cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                LOGGER.info("Saved cache: %s", cache_path)
                return payload
            self._save_autocli_raw_output(
                command,
                cache_path,
                str(exc.stdout or ""),
                str(exc.stderr or ""),
                f"exit_{exc.returncode}_without_json",
            )
            raise

        payload = self._load_autocli_json(stdout, stderr, strict=False)
        if payload is None:
            self._save_autocli_raw_output(command, cache_path, stdout, stderr, "success_without_json")
            LOGGER.error("AutoCLI exited successfully but no JSON payload could be parsed.")
            raise json.JSONDecodeError("AutoCLI did not return valid JSON", stdout or stderr or "", 0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved cache: %s", cache_path)
        return payload

    @staticmethod
    def _load_autocli_json(stdout: str, stderr: str = "", strict: bool = True) -> Any | None:
        sources = [stdout or "", stderr or "", f"{stdout or ''}\n{stderr or ''}"]
        payloads: list[Any] = []
        for raw_text in sources:
            text = ScraperEngine._clean_autocli_text(raw_text)
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            for line in reversed(text.splitlines()):
                candidate = line.strip().rstrip(",")
                if candidate.startswith(("{", "[")):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

            payloads.extend(ScraperEngine._scan_json_payloads(text))
        if payloads:
            return max(payloads, key=ScraperEngine._json_payload_size)

        if strict:
            LOGGER.error("AutoCLI returned non-JSON stdout:\n%s", (stdout or "")[-4000:])
            if stderr.strip():
                LOGGER.error("AutoCLI stderr:\n%s", stderr[-4000:])
            raise json.JSONDecodeError("AutoCLI did not return valid JSON", stdout or "", 0)
        return None

    @staticmethod
    def _clean_autocli_text(text: str) -> str:
        text = ANSI_ESCAPE_RE.sub("", text or "")
        return text.replace("\x00", "").lstrip("\ufeff").strip()

    @staticmethod
    def _scan_json_payloads(text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        payloads: list[Any] = []
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            payloads.append(payload)
        return payloads

    @staticmethod
    def _json_payload_size(payload: Any) -> int:
        try:
            return len(json.dumps(payload, ensure_ascii=False))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _save_autocli_raw_output(command: list[str], cache_path: Path, stdout: str, stderr: str, reason: str) -> None:
        debug_dir = ROOT / "logs" / "autocli_raw"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_reason = safe_json_filename(reason) or "autocli"
        base = f"{stamp}-{safe_json_filename(cache_path.stem)}-{safe_reason}"
        metadata = {
            "reason": reason,
            "cache_path": str(cache_path),
            "command": command,
            "stdout_bytes": len((stdout or "").encode("utf-8", errors="replace")),
            "stderr_bytes": len((stderr or "").encode("utf-8", errors="replace")),
        }
        (debug_dir / f"{base}.meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_dir / f"{base}.stdout.txt").write_text(stdout or "", encoding="utf-8", errors="replace")
        (debug_dir / f"{base}.stderr.txt").write_text(stderr or "", encoding="utf-8", errors="replace")
        LOGGER.error("Saved raw AutoCLI output for inspection: %s", debug_dir / f"{base}.*")

    def _run_command_with_timeout(self, command: list[str]) -> tuple[str, str]:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        use_shell = os.name == "nt"
        popen_command: list[str] | str = self._windows_shell_join(command) if use_shell else command
        start_time = time.monotonic()
        stdout_fd, stdout_name = tempfile.mkstemp(prefix="autocli-stdout-", suffix=".log")
        stderr_fd, stderr_name = tempfile.mkstemp(prefix="autocli-stderr-", suffix=".log")
        os.close(stdout_fd)
        os.close(stderr_fd)
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)
        timed_out = False
        try:
            with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as stderr_file:
                process = subprocess.Popen(
                    popen_command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=creationflags,
                    shell=use_shell,
                )
                LOGGER.info("AutoCLI started with PID %s", process.pid)
                deadline = start_time + self.timeout_seconds
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        self._kill_process_tree(process)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    try:
                        process.wait(timeout=min(15, remaining))
                    except subprocess.TimeoutExpired:
                        LOGGER.info("AutoCLI still running after %.0fs (PID %s)", time.monotonic() - start_time, process.pid)

            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            elapsed = time.monotonic() - start_time
            if timed_out:
                LOGGER.error("AutoCLI timed out after %s seconds: %s", self.timeout_seconds, " ".join(command))
                if stdout:
                    LOGGER.error("AutoCLI stdout before timeout:\n%s", stdout[-4000:])
                if stderr:
                    LOGGER.error("AutoCLI stderr before timeout:\n%s", stderr[-4000:])
                raise AutoCLITimeoutError(
                    f"AutoCLI timed out after {self.timeout_seconds} seconds",
                    stdout=stdout,
                    stderr=stderr,
                )

            LOGGER.info("AutoCLI exited with code %s in %.1fs", process.returncode, elapsed)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
            return stdout, stderr
        finally:
            for temp_path in (stdout_path, stderr_path):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    LOGGER.debug("Could not remove temp AutoCLI log file: %s", temp_path)

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if process.poll() is None:
                process.kill()
            return
        process.kill()

    @staticmethod
    def _quote_command_token(token: str) -> str:
        token = token.strip()
        if not token:
            return "autocli"
        if len(token) >= 2 and token[0] in {"'", '"'} and token[-1] == token[0]:
            return token
        if os.name == "nt":
            return subprocess.list2cmdline([token])
        return shlex.quote(token)

    @staticmethod
    def _windows_shell_join(command: list[str]) -> str:
        parts: list[str] = []
        for arg in command:
            quoted = subprocess.list2cmdline([arg])
            if not quoted.startswith('"') and re.search(r'[&|<>^()]', arg):
                quoted = f'"{arg}"'
            parts.append(quoted)
        return " ".join(parts)

    def _log_autocli_resolution(self) -> None:
        executable = self.autocli_path.strip().strip('"').strip("'")
        if Path(executable).is_file():
            LOGGER.info("Using AutoCLI executable: %s", executable)
            return
        resolved = shutil.which(executable)
        if resolved:
            LOGGER.info("Using AutoCLI executable from PATH: %s", resolved)
            return
        LOGGER.warning(
            "AutoCLI executable was not found on PATH: %s. Set AUTOCLI_PATH to the full autocli executable path.",
            executable,
        )

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
You are a senior pet behavior-informed product reviewer, people-first SEO editor, and skeptical buyer advocate.
Write in English for US pet owners. Your job is not to sound like a catalog. Your job is to help a real dog or cat owner avoid regret.

Banned Phrases & Tone Rules (CRITICAL):
- NEVER use the following AI cliches: "delve into", "a testament to", "crucial", "in conclusion", "vital", "elevate", "realm", "bustling", "moreover", "furthermore", "tapestry", "game-changer", "unleash", "furry friend", "picture this", "navigate", "symphony", "undeniable", "paramount".
- DO NOT use robotic transitional phrases or summary paragraphs that add no value.
- Write in short, punchy paragraphs (maximum 2-3 sentences).
- Use a calm, field-guide tone: specific, skeptical, and practical without sounding theatrical.
- Avoid repeating phrases like "customer-summary signals", "marketplace signals", "no hands-on testing", or "AI vision report" throughout the article. If evidence is limited, say it once in plain editorial language.
- Use bold text heavily to make the article highly scannable for mobile readers.

SEO and helpful-content strategy:
- Match the provided primary SEO title, search intent, and target keywords naturally. Do not keyword-stuff.
- Emphasize original decision value: regret analysis, owner-fit matching, tradeoffs, red flags, safety cautions, and usage tips.
- Use question-based H3 headers for long-tail SEO where natural (e.g., instead of just "Durability", use "Is it safe for power chewers?").
- Treat "best" as scenario-based, not absolute. Recommend by pet size, behavior, household setup, safety risk, and owner tolerance for maintenance.
- Do not write primarily to manipulate search rankings; write to satisfy the real owner's task.
- Do not repeat the same praise for every product. Every product section needs a distinct reason to exist.
- Do not create a standalone FAQ section or Related Resources section; the publishing pipeline adds those consistently after generation.

Evidence rules:
- Use only the facts provided in the Product JSON. Do not invent specs, testing, photos, studies, counts, or claims.
- The most valuable section is not the spec list. Focus on what buyers might regret after purchase and how to use the product smarter.
- The How We Read This List section must be concise and professional. Do not lead with a legalistic disclaimer. If Product JSON does not include hands-on testing, say the guide is based on product specifications, buyer feedback patterns, and listing evidence.
- Never mention AI, AI vision, or internal pipeline details in published article copy.
- Do not present Amazon bestseller status as proof of quality; treat it only as a marketplace popularity signal.
- Do not overstate medical, nutrition, behavior, training, or safety claims.
- Do not diagnose, prescribe, promise health outcomes, or imply that products replace veterinary care.

Link and compliance rules:
- Do not use affiliate, tracking, shortened, or redirected links.
- Every purchase link must be exactly: [Check Price on Amazon](https://www.amazon.com/dp/{ASIN})
- Never include exact prices. Use broad tiers like "Budget-friendly", "Mid-range", and "Premium price".
- Use descriptive anchor text for internal links; avoid anchors like "click here".
- Use only the provided internal article URLs; do not invent internal links.
- Do not create a standalone authority-links block. If an authority link is truly useful inside the Buying Guide, use only akc.org, aspca.org, vcahospitals.com, merckvetmanual.com, avma.org, vet.cornell.edu, or pubmed.ncbi.nlm.nih.gov.

Image rules:
- For every individual product section, place the product image immediately under that product heading using Markdown:
  ![{SEO alt text featuring specific long-tail keywords}]({image_url})
- If a product has no image_url, omit only the image line for that product.

Required Markdown structure:
- Do not output a Markdown H1. Hugo frontmatter supplies the page H1.
- Use these exact H2 headings without numeric prefixes:
  ## How We Read This List
  ## Quick Picks
  ## Buying Guide
  ## Comparison Table
  ## Deep Reviews
  ## Final Summary
- Before "## How We Read This List", write 2-4 tight introduction paragraphs, pain-first, no "Introduction" heading.
- Quick Picks: a compact bullet list naming the best product for 4-6 specific buyer needs.
- Buying Guide: practical criteria, red flags, fit/safety notes, maintenance cautions, and buying mistakes.
- Comparison Table: include product, best for, standout upside, buyer caution, skip-if. Do not include price.
- Deep Reviews: exactly 10 products. Use H3 product headings. For each, include image if available, short verdict, best for, skip it if, what buyers may regret, complaint/watch-out pattern, pros, cons, Expert Tip, and clean Amazon link.
- Final Summary: brief, scenario-based wrap-up.

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
        topic = SEOTopicCatalog.for_task(task)
        prompt_products = self._prepare_products_for_prompt(products)[:10]
        compact_json = json.dumps(prompt_products, ensure_ascii=False, separators=(",", ":"))
        seo_brief = (
            f"Primary SEO title: {topic.title}\n"
            f"Meta description: {topic.description}\n"
            f"Canonical URL path: {MarkdownExporter.url_for(task)}\n"
            f"Target keywords: {', '.join(topic.keywords)}\n"
            f"FAQ questions reserved for pipeline: {json.dumps([q for q, _ in topic.faqs], ensure_ascii=False)}\n"
            f"Category path: {task.category_path}\n"
            f"Category name: {task.category_name}\n"
            f"Pet type: {task.pet_type}\n"
            "Search intent: help pet owners decide what to buy, what to skip, and what safety or fit tradeoffs to expect before purchase.\n"
        )
        user_prompt = (
            f"{seo_brief}"
            "Important: do not answer the reserved FAQ questions in a standalone FAQ section; the pipeline adds that section later.\n"
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
            draft_prompt = user_prompt + "\n\nCRITICAL DIRECTIVE: Generate the entire Markdown article with the required intro and 6 exact H2 sections. Ensure facts are evidence-bound."

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
                    "or prices. Maintain the required intro and 6 exact H2 headings. "
                    "Do not add a Markdown H1, FAQ section, or Related Resources section.\n\n"
                    f"=== ORIGINAL SEO BRIEF ===\n{seo_brief}=== END SEO BRIEF ===\n\n"
                    f"=== DRAFT ARTICLE ===\n{draft_markdown}\n=== END DRAFT ==="
                )

                if is_proxy_claude:
                    LOGGER.info("Using chunked generation for proxy Claude to bypass timeout limit.")
                    p1 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: Only refine the intro plus these H2 sections: ## How We Read This List, ## Quick Picks, ## Buying Guide, and ## Comparison Table. Stop after the Comparison Table. Do not use numbered headings."
                    LOGGER.info("Generating Chunk 1/3 (Intro -> Comparison Table)")
                    body1 = self._generate_openai(self.SYSTEM_PROMPT, p1)

                    p2 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: Only refine ## Deep Reviews for the first 5 products. Start with exactly '## Deep Reviews'. Use H3 product headings. Do not generate intro, Quick Picks, Buying Guide, Comparison Table, FAQ, Related Resources, or Final Summary."
                    LOGGER.info("Generating Chunk 2/3 (Deep Reviews 1-5)")
                    body2 = self._generate_openai(self.SYSTEM_PROMPT, p2)

                    p3 = refinement_prompt + "\n\nCRITICAL DIRECTIVE: Only refine the remaining product H3 reviews, then ## Final Summary. Do not output ## Deep Reviews again. Do not repeat intro, Quick Picks, Buying Guide, Comparison Table, FAQ, or Related Resources."
                    LOGGER.info("Generating Chunk 3/3 (Deep Reviews 6-10 + Final Summary)")
                    body3 = self._generate_openai(self.SYSTEM_PROMPT, p3)
                    body = f"{body1}\n\n{body2}\n\n{body3}"
                else:
                    body = self._generate_openai(self.SYSTEM_PROMPT, refinement_prompt)

        elif is_proxy_claude:
            LOGGER.info("Using chunked generation for proxy Claude to bypass timeout limit.")

            p1 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate the intro plus these H2 sections: ## How We Read This List, ## Quick Picks, ## Buying Guide, and ## Comparison Table. Stop after the Comparison Table. Do not use numbered headings."
            LOGGER.info("Generating Chunk 1/3 (Intro -> Comparison Table)")
            body1 = self._generate_openai(self.SYSTEM_PROMPT, p1)

            p2 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate ## Deep Reviews for the first 5 products in the JSON list. Start with exactly '## Deep Reviews'. Use H3 product headings. Do not generate intro, Quick Picks, Buying Guide, Comparison Table, FAQ, Related Resources, or Final Summary."
            LOGGER.info("Generating Chunk 2/3 (Deep Reviews 1-5)")
            body2 = self._generate_openai(self.SYSTEM_PROMPT, p2)

            p3 = user_prompt + "\n\nCRITICAL DIRECTIVE: Only generate the remaining product H3 reviews, then ## Final Summary. Do not output ## Deep Reviews again. Do not repeat intro, Quick Picks, Buying Guide, Comparison Table, FAQ, or Related Resources."
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
        body = self._sanitize_unsupported_vision_claims(body)
        body = SEOArticleOptimizer.enrich_body(body, task)
        body = SEOResourceLinker.enrich(body, task, related_articles=related_articles)
        return self._sanitize_external_links(body)

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
            "we reviewed the available buyer feedback patterns",
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
            "reviewed the available buyer feedback patterns",
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
        return SEOTopicCatalog.for_task(task).title

    @staticmethod
    def url_for(task: CategoryTask) -> str:
        return f"/{task.pet_type}/best-{slugify(task.category_name)}/"

    @staticmethod
    def _frontmatter(task: CategoryTask, title: str) -> str:
        topic = SEOTopicCatalog.for_task(task)
        description = topic.description
        path_tags = [
            part.strip().lower()
            for part in task.category_path.split(">")
            if part.strip() and part.strip().lower() not in {"pet supplies", task.pet_type}
        ]
        tags = list(dict.fromkeys([task.pet_type, task.category_name.lower(), *path_tags, "pet supplies"]))
        yaml_tags = "\n".join(f"  - {MarkdownExporter._yaml_quote(tag)}" for tag in tags)
        yaml_keywords = "\n".join(f"  - {MarkdownExporter._yaml_quote(keyword)}" for keyword in topic.keywords)
        return (
            "---\n"
            f"title: {MarkdownExporter._yaml_quote(title)}\n"
            f"description: {MarkdownExporter._yaml_quote(description)}\n"
            "keywords:\n"
            f"{yaml_keywords}\n"
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
    if args.refresh_links_only:
        changed_count = SEOResourceLinker.refresh_existing_content(OUTPUT_DIR)
        LOGGER.info("Refreshed SEO resource links in %s article files", changed_count)
        return 0
    if args.refresh_seo_only:
        seo_changed_count = SEOArticleOptimizer.refresh_existing_content(OUTPUT_DIR)
        link_changed_count = SEOResourceLinker.refresh_existing_content(OUTPUT_DIR)
        LOGGER.info(
            "Refreshed SEO metadata/FAQ in %s article files and resource links in %s article files",
            seo_changed_count,
            link_changed_count,
        )
        return 0

    task_manager = TaskManager(args.tracking_json)
    scraper = ScraperEngine(
        timeout_seconds=args.timeout,
        bestsellers_command_template=args.autocli_bestsellers_command,
        product_command_template=args.autocli_product_command,
        autocli_path=args.autocli_path,
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

        success_count = 0
        failed_count = 0
        for task in batch:
            try:
                related_articles = task_manager.get_related_articles(task, limit=args.related_limit)
                products = scraper.scrape_category(task, top_n=args.top_n, min_success=args.min_products)
                markdown = generator.generate(task, products, related_articles=related_articles)
                article_path, article_url, title = exporter.export(task, markdown)
                task_manager.mark_completed(task.node_id, article_path=article_path, article_url=article_url, title=title)
                LOGGER.info("Completed category %s", task.node_id)
                success_count += 1
            except Exception as exc:
                LOGGER.exception("Failed category %s: %s", task.node_id, exc)
                task_manager.mark_failed(task.node_id)
                failed_count += 1
        if failed_count and not success_count:
            LOGGER.error("All %s claimed categories failed; stopping with exit code 1", failed_count)
            return 1
        if failed_count:
            LOGGER.warning("Completed %s categories with %s failures", success_count, failed_count)
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
    parser.add_argument("--autocli-path", default=os.environ.get("AUTOCLI_PATH"), help="Full path to autocli if it is not on PATH")
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
    parser.add_argument(
        "--refresh-links-only",
        action="store_true",
        help="Rebuild SEO Related Resources sections for existing content and exit",
    )
    parser.add_argument(
        "--refresh-seo-only",
        action="store_true",
        help="Refresh article titles, descriptions, keywords, FAQ sections, and SEO resource links, then exit",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
