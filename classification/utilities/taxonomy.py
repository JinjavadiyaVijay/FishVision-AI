"""
taxonomy.py — Species common-name lookup and master species catalog builder.

The master catalog is the single source of truth for the entire pipeline.
Each entry contains:
  scientific_name  : "Genus species"  (canonical identifier)
  common_name      : English common name (display only; None if unknown)
  family           : Taxonomic family
  genus            : Taxonomic genus
  class_id         : Integer index (assigned at training time, not here)
  image_count      : Number of crops in OzFish (populated at build time)
  dataset_source   : "OzFish"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common-name lookup table for OzFish species (sourced from FishBase / Fisheries)
# Key: "Genus species" (lowercase species)
# ---------------------------------------------------------------------------
COMMON_NAMES: dict[str, str] = {
    # Lethrinidae — Emperors
    "Lethrinus punctulatus":     "Bluespotted emperor",
    "Lethrinus digramma":        "Longnose emperor",
    "Lethrinus atkinsoni":       "Pacific yellowtail emperor",
    "Lethrinus nebulosus":       "Spangled emperor",
    "Lethrinus olivaceus":       "Longface emperor",
    "Lethrinus erythropterus":   "Longfin emperor",
    "Lethrinus laticaudis":      "Grass emperor",
    "Lethrinus lentjan":         "Pink ear emperor",
    "Lethrinus variegatus":      "Slender emperor",
    "Lethrinus miniatus":        "Trumpet emperor",
    "Lethrinus rubrioperculatus":"Spotcheek emperor",
    # Lutjanidae — Snappers
    "Lutjanus sebae":            "Red emperor",
    "Lutjanus rubrioperculatus": "Slater's snapper",
    "Lutjanus vitta":            "Brownstripe red snapper",
    "Lutjanus caudimacula":      "Spot-tail snapper",
    "Lutjanus fulvus":           "Blacktail snapper",
    "Lutjanus bohar":            "Two-spot red snapper",
    "Lutjanus argentimaculatus": "Mangrove red snapper",
    "Lutjanus kasmira":          "Common bluestripe snapper",
    "Lutjanus fulviflamma":      "Dory snapper",
    "Lutjanus ehrenbergii":      "Blackspot snapper",
    "Lutjanus monostigma":       "One-spot snapper",
    "Lutjanus rivulatus":        "Blubberlip snapper",
    "Lutjanus gibbus":           "Humpback red snapper",
    "Lutjanus decussatus":       "Checkered snapper",
    "Lutjanus quinquelineatus":  "Five-lined snapper",
    "Lutjanus multilineatus":    "Multiline snapper",
    "Lutjanus semicinctus":      "Black-banded snapper",
    "Lutjanus biguttatus":       "Two-spot snapper",
    "Lutjanus bengalensis":      "Bengal snapper",
    "Lutjanus lemniscatus":      "Yellowstreaked snapper",
    "Lutjanus malabaricus":      "Malabar blood snapper",
    # Acanthuridae — Surgeonfishes
    "Acanthurus lutescens":      "Yellowfin surgeonfish",
    "Acanthurus triostegus":     "Convict surgeonfish",
    "Acanthurus olivaceus":      "Orangespot surgeonfish",
    "Acanthurus nigricauda":     "Epaulette surgeonfish",
    "Acanthurus thompsoni":      "Thompson's surgeonfish",
    "Naso gymnostethus":         "Bignose unicornfish",
    "Naso unicornis":            "Bluespine unicornfish",
    "Naso lituratus":            "Orangespine unicornfish",
    "Naso vlamingii":            "Bignose unicornfish",
    # Labridae — Wrasses
    "Thalassoma lunare":         "Moon wrasse",
    "Thalassoma hardwicke":      "Sixbar wrasse",
    "Thalassoma lutescens":      "Sunset wrasse",
    "Halichoeres dimidiatus":    "Twotone wrasse",
    "Halichoeres hortulanus":    "Checkerboard wrasse",
    "Cheilinus multinotatus":    "Pastel-green wrasse",
    "Oxycheilinus rivulatus":    "Dotted wrasse",
    "Stethojulis striatus":      "Kākou wrasse",
    "Hemigymnus fasciatus":      "Barred thicklip",
    "Hemigymnus melapterus":     "Blackeye thicklip",
    "Bodianus loxozonus":        "Blackfin hogfish",
    "Gomphosus varius":          "Birdwrasse",
    "Coris gaimard":             "African coris",
    # Pomacentridae — Damselfishes
    "Pomacentrus coelestis":     "Neon damselfish",
    "Chromis viridis":           "Blue-green chromis",
    "Dascyllus reticulatus":     "Reticulate dascyllus",
    "Pomacentrus moluccensis":   "Lemon damsel",
    "Chrysiptera parasema":      "Yellowtail blue damsel",
    # Carangidae — Jacks / Trevallies
    "Caranx argenteus":          "Silver trevally",
    "Caranx sexfasciatus":       "Bigeye trevally",
    "Caranx melampygus":         "Bluefin trevally",
    "Caranx ignobilis":          "Giant trevally",
    "Caranx papuensis":          "Brassy trevally",
    # Caesionidae — Fusiliers
    "Caesio tile":               "Dark-banded fusilier",
    "Pterocaesio tile":          "Dark-banded fusilier",
    "Caesio cuning":             "Redbelly yellowtail fusilier",
    "Pterocaesio chrysozona":    "Goldband fusilier",
    # Serranidae — Groupers
    "Epinephelus fulvoguttatus": "Tiger grouper",
    "Epinephelus areolatus":     "Areolate grouper",
    "Epinephelus malabaricus":   "Malabar grouper",
    "Epinephelus coioides":      "Orange-spotted grouper",
    "Epinephelus fasciatus":     "Blacktip grouper",
    # Scaridae — Parrotfishes
    "Scarus sordidus":           "Bullethead parrotfish",
    "Scarus dimidiatus":         "Yellowbarred parrotfish",
    "Scarus niger":              "Swarthy parrotfish",
    "Scarus rivulatus":          "Surf parrotfish",
    "Chlorurus sordidus":        "Daisy parrotfish",
    # Balistidae — Triggerfishes
    "Odonus niger":              "Redfang triggerfish",
    "Balistoides viridescens":   "Titan triggerfish",
    "Sufflamen annulatus":       "Scythe triggerfish",
    "Sufflamen chrysopterum":    "Halfmoon triggerfish",
    # Mullidae — Goatfishes
    "Parupeneus temminckii":     "Doublebar goatfish",
    "Mulloidichthys vanicolensis":"Yellowfin goatfish",
    "Parupeneus indicus":        "Indian goatfish",
    # Siganidae — Rabbitfishes
    "Siganus fuscescens":        "Mottled spinefoot",
    "Siganus argenteus":         "Streamlined spinefoot",
    "Siganus puellus":           "Decorated rabbitfish",
    # Nemipteridae — Threadfin breams
    "Nemipterus grammoptilus":   "Threadfin bream",
    "Nemipterus nemurus":        "Notchedfin threadfin bream",
    # Haemulidae
    "Diagramma porosus":         "Australian sweetlips",
    "Plectorhinchus diagrammus": "Striped sweetlips",
    # Chaetodontidae — Butterflyfishes
    "Chaetodon auriga":          "Threadfin butterflyfish",
    "Chaetodon vagabundus":      "Vagabond butterflyfish",
    "Chaetodon lunulatus":       "Oval butterflyfish",
    # Carcharhinidae — Requiem sharks
    "Carcharhinus melanopterus": "Blacktip reef shark",
    "Carcharhinus amblyrhynchos":"Grey reef shark",
}


class TaxonomyCatalogBuilder:
    """
    Build the master species catalog from a parsed OzFish DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of OzFishParser.parse().
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def build(self) -> pd.DataFrame:
        """
        Build the master species catalog.

        Returns
        -------
        pd.DataFrame with columns:
            scientific_name, common_name, family, genus,
            image_count, dataset_source,
            is_viable, is_recommended,
            class_id (assigned in rank order by image_count)
        """
        clean = self.df[~self.df["is_ambiguous"] & self.df["file_exists"]].copy()

        catalog = (
            clean.groupby(["full_species", "family", "genus"])
            .size()
            .reset_index(name="image_count")
            .sort_values("image_count", ascending=False)
            .reset_index(drop=True)
        )

        catalog.rename(columns={"full_species": "scientific_name"}, inplace=True)
        catalog["common_name"]    = catalog["scientific_name"].map(COMMON_NAMES)
        catalog["dataset_source"] = "OzFish"
        catalog["is_viable"]      = catalog["image_count"] >= 100
        catalog["is_recommended"] = catalog["image_count"] >= 200
        # class_id assigned later when training set is finalised — placeholder
        catalog["class_id"]       = pd.array(
            range(len(catalog)), dtype="Int64"
        )

        logger.info(
            "Master catalog: %d total species, %d viable (≥100), %d recommended (≥200), "
            "%d with known common name",
            len(catalog),
            catalog["is_viable"].sum(),
            catalog["is_recommended"].sum(),
            catalog["common_name"].notna().sum(),
        )
        return catalog

    def save(self, catalog: pd.DataFrame, output_dir: Path) -> None:
        """Save catalog as JSON and CSV to output_dir/metadata/."""
        meta_dir = Path(output_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # JSON — full catalog
        records = catalog.to_dict(orient="records")
        json_path = meta_dir / "master_species_catalog.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved catalog JSON: %s", json_path)

        # CSV — easy to open in Excel
        csv_path = meta_dir / "master_species_catalog.csv"
        catalog.to_csv(csv_path, index=False)
        logger.info("Saved catalog CSV:  %s", csv_path)

    @staticmethod
    def get_common_name(scientific_name: str) -> Optional[str]:
        """Look up common name for a scientific name."""
        return COMMON_NAMES.get(scientific_name)
