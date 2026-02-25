#!/usr/bin/env python3
"""Batch generation of RPG item icons + environment assets using SDXL Lightning.
V2: Generate at native 1024x1024 then downscale to 512. Improved prompts.
"""

import os
import sys
import time
import json
import random

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import torch
from PIL import Image
from datetime import datetime

GAME_ASSETS = "/home/x/pythonpg/diabloremix/diabloremix_godot/assets"
ITEM_DIR = os.path.join(GAME_ASSETS, "items")
TEXTURE_DIR = os.path.join(GAME_ASSETS, "textures")

RENDER_SIZE = 1024
FINAL_ITEM_SIZE = 512
NUM_STEPS = 4

RARITY_NAMES = ["Common", "Fine", "Superior", "Legendary", "Godly"]
RARITY_STYLE = {
    0: "simple iron, worn, basic",
    1: "polished steel, well-crafted, slightly glowing edges",
    2: "enchanted, blue magical runes, ethereal glow, masterwork",
    3: "legendary artifact, brilliant golden glow, ornate engravings, radiating power, epic",
    4: "godly divine weapon, blinding holy light, celestial golden aura, transcendent power, impossible perfection, heavenly radiance",
}

STYLE_CORE = ", single item on black background, RPG icon, digital painting, sharp details, studio lighting"
QUALITY_BOOST = ", masterpiece, artstation, 4k"

NEGATIVE_PROMPT = (
    "blurry, out of focus, soft, deformed, disfigured, low quality, ugly, bad proportions, "
    "watermark, text, letters, words, signature, frame, border, multiple items, collage, grid, "
    "person, hand, fingers, human, body parts, face, "
    "pixelated, jpeg artifacts, noise, grainy, cropped, poorly drawn, amateur, "
    "white background, gradient background, colorful background, busy background, "
    "photorealistic, photograph, 3d render"
)

WEAPON_PROMPTS = [
    "a gleaming short sword with leather-wrapped handle",
    "a long sword with ornate crossguard and polished blade",
    "a broad sword with wide blade and golden pommel",
    "a massive great sword with rune-etched blade",
    "a curved scimitar with jeweled hilt",
    "a heavy falchion with single-edged curved blade",
    "a scottish claymore with distinctive cross-hilt",
    "an elegant katana with wrapped handle and curved blade",
    "a thin rapier with elaborate basket hilt",
    "a sturdy hand axe with wooden handle",
    "a double-headed battle axe with long shaft",
    "a war axe with crescent blade and spike",
    "a massive great axe with enormous blade",
    "a small throwing hatchet with leather grip",
    "a flanged mace with iron head",
    "a spiked morning star on chain",
    "a war flail with spiked iron ball on chain",
    "a heavy war hammer with square head",
    "an enormous maul with stone head",
    "a slim dagger with double-edged blade",
    "a narrow stiletto with triangular blade",
    "a wavy-bladed kris dagger",
    "a long dirk with Celtic knot pommel",
    "a throwing knife with balanced blade",
    "a tall wooden wizard staff with crystal orb",
    "a magic wand with glowing tip",
    "a royal scepter with gemstone head",
    "a iron rod with magical runes",
    "a crystal orb pulsing with energy",
    "a long spear with leaf-shaped blade",
    "a military pike with steel point",
    "a halberd with axe blade and spike",
    "a three-pronged trident dripping water",
    "a sleek javelin with feathered shaft",
    "a wooden recurve bow with arrows",
    "a mechanical crossbow with iron bolts",
    "an elven long bow with silver inlay",
    "a composite bow with horn and sinew",
    "a bone recurve bow with dark string",
    "an ancient tome crackling with lightning",
]

ARMOR_PROMPTS = [
    "rugged leather chest armor with buckles",
    "studded leather armor with metal rivets",
    "interlocking chain mail shirt",
    "overlapping scale mail armor",
    "full plate mail armor with embossed chest",
    "splint mail armor with vertical metal strips",
    "ring mail armor with linked rings",
    "brigandine armor with steel plates in cloth",
    "gothic full plate armor with pointed design",
    "complete set of shining full plate armor",
    "quilted padded armor vest",
    "rough hide armor from beast skin",
    "steel breastplate with shoulder guards",
    "ornate field plate armor",
    "ancient ceremonial armor with symbols",
    "lightweight plate armor with mobility",
    "bone armor made from dragon bones",
    "serpent-scale armor with green sheen",
    "demonic black armor with red glow",
    "dragon scale armor with iridescent scales",
    "dark shadow vest that seems to absorb light",
    "flowing mage robe with arcane symbols",
    "battle-worn war coat with steel plates",
    "crusader plate armor with holy cross",
    "templar knight armor with red cross",
    "flowing silk robe with gold trim",
    "mystic purple robe with star patterns",
    "arcane vestment crackling with energy",
    "leather battle harness with pouches",
    "iron cuirass with hammered details",
    "elven mithril chain mail",
    "dwarven master-forged plate armor",
    "crystal armor that refracts light",
    "astral plate armor with galaxy patterns",
    "chaos armor with shifting dark energy",
    "runic vest with glowing inscriptions",
    "storm coat with lightning patterns",
    "frost-covered ice armor",
    "fire-forged plate with ember glow",
    "void shroud with dark purple energy",
]

BOOTS_PROMPTS = [
    "sturdy leather boots with thick soles",
    "heavy iron-shod boots",
    "chain mail boots with steel toes",
    "plate greaves with knee guards",
    "tall war boots with steel plates",
    "light leather greaves",
    "battle-worn combat boots",
    "fine mesh boots with silver wire",
    "demonhide boots with red stitching",
    "sharkskin boots with grey texture",
    "myrmidon heavy greaves",
    "bone-plated greaves",
    "scarab-engraved boots",
    "boneweave boots with skeletal design",
    "mirrored chrome boots",
    "simple leather sandals with straps",
    "soft cloth slippers",
    "wrapped fur boots",
    "clasped metal boots",
    "laced-up riding boots",
    "steel sabatons with articulated toes",
    "iron-treaded heavy boots",
    "bronze greaves with lion motif",
    "silver-plated boots with moonstone",
    "gold-trimmed ceremonial boots",
    "shadow-black assassin boots",
    "wind-touched feathered boots",
    "storm-treaded boots with lightning",
    "frost-covered ice boots",
    "fire-walker boots with ember soles",
    "crystal greaves with prismatic glow",
    "titan-sized heavy boots",
    "dragon-scale boots",
    "ethereal translucent boots",
    "void-walker boots with dark energy",
    "elven leaf-pattern boots",
    "dwarven iron-shod boots",
    "nomad wrapped desert boots",
    "raider spiked boots",
    "pilgrim simple walking boots",
]

RING_PROMPTS = [
    "simple iron band ring",
    "bronze ring with small gem",
    "silver ring with intricate filigree",
    "gold ring with large gemstone",
    "platinum band with diamond",
    "bone ring carved from skull",
    "coral ring with ocean colors",
    "crystal ring refracting light",
    "diamond ring with brilliant cut stone",
    "emerald ring with green glow",
    "ruby ring with deep red stone",
    "sapphire ring with blue sparkle",
    "topaz ring with amber glow",
    "amethyst ring with purple crystal",
    "onyx ring with dark stone",
    "jade ring with green carved stone",
    "moonstone ring with pearlescent glow",
    "opal ring with rainbow fire",
    "garnet ring with dark red stone",
    "tourmaline ring with multicolor stone",
    "shadow ring that absorbs light",
    "storm ring with lightning crackling",
    "frost ring with ice crystals",
    "fire ring with ember glow",
    "nature ring with vine patterns",
    "war band with spiked edges",
    "mage ring with arcane symbols",
    "thief ring with hidden blade",
    "signet ring with noble crest",
    "serpent ring shaped like coiling snake",
    "dragon ring with dragon head",
    "phoenix ring with fiery feathers",
    "titan ring of massive size",
    "angel ring with white wings",
    "demon ring with dark horns",
    "ancient ring with worn runes",
    "runic band with glowing symbols",
    "mystic loop with swirling energy",
    "void ring with dark portal",
    "celestial ring with star pattern",
]

AMULET_PROMPTS = [
    "bone amulet on leather cord",
    "coral amulet with ocean charm",
    "crystal pendant on silver chain",
    "gold talisman with ancient symbols",
    "silver charm with protective runes",
    "jade amulet with dragon carving",
    "onyx pendant with dark energy",
    "ruby talisman with blood-red glow",
    "sapphire charm with blue radiance",
    "emerald amulet with forest green glow",
    "dragon tooth pendant on chain",
    "phoenix feather charm with fire",
    "star-shaped pendant with cosmic glow",
    "crescent moon charm with silver",
    "sun talisman with golden rays",
    "shadow pendant absorbing light",
    "storm amulet with lightning bolt",
    "frost charm with ice crystal",
    "fire talisman with flame within",
    "holy symbol with divine light",
    "dark amulet with cursed energy",
    "arcane pendant with magic circles",
    "runic charm with ancient letters",
    "ancient talisman with hieroglyphs",
    "void pendant with swirling darkness",
]

CROWN_PROMPTS = [
    "simple leather cap with stitching",
    "iron helm with nose guard",
    "steel helm with visor",
    "bone helm made from skull",
    "full helm with face plate",
    "great helm with plume",
    "war crown with iron spikes",
    "winged helm with golden wings",
    "horned helm with curved horns",
    "spired helm with tall point",
    "golden crown with gemstones",
    "silver tiara with diamonds",
    "elegant circlet with moonstone",
    "coronet with pearls and rubies",
    "royal diadem with celestial gems",
    "dark hood with shadow magic",
    "mysterious mask with eye slots",
    "monk cowl with runes",
    "chain coif with mail links",
    "armet helm with articulated visor",
    "dragon crown with dragon motif",
    "phoenix helm with fire plume",
    "titan crown of enormous size",
    "arcane hood with magical stars",
    "void helm with dark energy swirl",
]

PANTS_BASES = [
    "leather leggings with knee patches",
    "studded leather pants with rivets",
    "chain mail leggings",
    "plate leg armor with knee guards",
    "war pants with metal plates",
    "light cloth leggings",
    "battle-worn leggings with patches",
    "mesh pants with wire weave",
    "demonhide leggings with red glow",
    "sharkskin pants with grey scales",
    "heavy myrmidon leg armor",
    "bone-plated leg guards",
    "scarab-engraved leggings",
    "boneweave pants with skeletal design",
    "mirrored chrome leg armor",
    "silk pants with gold embroidery",
    "mystic leggings with magic runes",
    "arcane pants with energy lines",
    "leather battle leggings",
    "iron-plated leg armor",
    "elven leaf-pattern leggings",
    "dwarven forged leg plates",
    "crystal leg armor with glow",
    "astral pants with star pattern",
    "chaos leggings with dark energy",
]

ITEM_TYPES = {
    "weapon": {"count": 200, "prompts": WEAPON_PROMPTS},
    "armor":  {"count": 200, "prompts": ARMOR_PROMPTS},
    "boots":  {"count": 200, "prompts": BOOTS_PROMPTS},
    "ring":   {"count": 200, "prompts": RING_PROMPTS},
    "amulet": {"count": 100, "prompts": AMULET_PROMPTS},
    "crown":  {"count": 100, "prompts": CROWN_PROMPTS},
    "pants":  {"count": 100, "prompts": PANTS_BASES},
}

ENV_PROMPTS = {
    "town_bg": {
        "prompt": (
            "a beautiful medieval fantasy town square at golden hour sunset, "
            "cobblestone streets leading to a central ornate fountain, "
            "half-timber tudor style buildings with colorful flower boxes, "
            "market stalls with vibrant cloth awnings in red and gold, "
            "warm orange lantern light glowing, lush green ivy on walls, "
            "peaceful welcoming atmosphere, wide panoramic view, "
            "professional concept art, highly detailed, digital painting, "
            "fantasy game environment, artstation quality, masterpiece"
        ),
        "size": (1024, 1024),
    },
    "floor_tex": {
        "prompt": (
            "seamless tileable dark dungeon stone floor texture, top-down orthographic view, "
            "aged worn cobblestone pavement with visible cracks and gaps, "
            "dark grey and brown earth tones, traces of green moss in crevices, "
            "RPG game dungeon tileset texture, photorealistic material, "
            "even lighting no shadows, highly detailed surface, 4k texture, masterpiece"
        ),
        "size": (1024, 1024),
    },
    "wall_tex": {
        "prompt": (
            "seamless tileable dark dungeon stone brick wall texture, front-facing flat view, "
            "aged weathered stone bricks with visible mortar lines between blocks, "
            "dark atmospheric grey-brown tones, patches of green moss and wear, "
            "RPG game dungeon tileset wall texture, photorealistic material, "
            "even lighting no perspective, highly detailed surface, 4k texture, masterpiece"
        ),
        "size": (1024, 1024),
    },
    "menu_bg": {
        "prompt": (
            "a dark gothic castle entrance hall interior, massive stone archway entrance, "
            "burning torches on walls casting dramatic orange volumetric light beams, "
            "rows of tall candles flickering, ancient stone pillars with carved details, "
            "mystical dark fantasy atmosphere, Diablo game visual style, "
            "professional concept art, cinematic dramatic composition, "
            "highly detailed digital painting, dark moody color palette, masterpiece"
        ),
        "size": (1024, 1024),
    },
}


def setup_pipeline():
    """Load SDXL Lightning pipeline with CPU offload for 8GB VRAM GPUs."""
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    print(f"CUDA: {has_cuda}, dtype: {dtype}", flush=True)
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB", flush=True)

    print("Loading SDXL base model...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if has_cuda else None,
    )
    print("Base model loaded.", flush=True)

    print("Loading Lightning UNet (4-step variant)...", flush=True)
    unet_path = hf_hub_download(
        repo_id="ByteDance/SDXL-Lightning",
        filename="sdxl_lightning_4step_unet.safetensors",
    )
    pipe.unet.load_state_dict(load_file(unet_path))
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    print("Lightning UNet loaded.", flush=True)

    if has_cuda:
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        print("CPU offload + VAE tiling enabled (saves VRAM).", flush=True)
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()

    print("Pipeline ready!", flush=True)
    return pipe


def pick_rarity(index, total):
    r = random.random() * 100
    if r < 1.0: return 4
    if r < 5.0: return 3
    if r < 15.0: return 2
    if r < 40.0: return 1
    return 0


def generate_items(pipe, resume_from=False):
    progress_file = os.path.join(GAME_ASSETS, "gen_progress.json")
    done_set = set()
    if resume_from and os.path.exists(progress_file):
        with open(progress_file) as f:
            done_set = set(json.load(f).get("done", []))
        print(f"Resuming: {len(done_set)} items already done.")

    global_id = 0
    total_generated = 0
    start_time = time.time()

    for item_type, cfg in ITEM_TYPES.items():
        type_dir = os.path.join(ITEM_DIR, item_type)
        os.makedirs(type_dir, exist_ok=True)
        count = cfg["count"]
        prompts = cfg["prompts"]

        print(f"\n--- Generating {count} {item_type} icons ---")
        for i in range(count):
            file_key = f"{item_type}/item_{global_id:04d}"
            out_path = os.path.join(type_dir, f"item_{global_id:04d}.png")

            if file_key in done_set:
                global_id += 1
                continue

            rarity = pick_rarity(i, count)
            base_prompt = prompts[i % len(prompts)]
            rarity_desc = RARITY_STYLE[rarity]

            prompt = f"{base_prompt}, {rarity_desc}, {STYLE_CORE}{QUALITY_BOOST}"

            try:
                result = pipe(
                    prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=0.0,
                    height=RENDER_SIZE,
                    width=RENDER_SIZE,
                )
                img = result.images[0]
                if FINAL_ITEM_SIZE != RENDER_SIZE:
                    img = img.resize((FINAL_ITEM_SIZE, FINAL_ITEM_SIZE), Image.LANCZOS)
                img.save(out_path, optimize=True)
                total_generated += 1
                done_set.add(file_key)

                if total_generated % 5 == 0:
                    with open(progress_file, "w") as f:
                        json.dump({"done": list(done_set)}, f)
                    elapsed = time.time() - start_time
                    rate = elapsed / total_generated if total_generated > 0 else 0
                    eta_remaining = rate * (1100 - total_generated - len(done_set))
                    print(f"  [{total_generated}] {file_key} | {RARITY_NAMES[rarity]} | {rate:.1f}s/img | ETA: {eta_remaining/60:.0f}min")

                if torch.cuda.is_available() and total_generated % 15 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR generating {file_key}: {e}")

            global_id += 1

    with open(progress_file, "w") as f:
        json.dump({"done": list(done_set)}, f)

    elapsed = time.time() - start_time
    print(f"\nItem generation complete: {total_generated} images in {elapsed:.1f}s ({elapsed/60:.1f}min)")


def generate_environments(pipe):
    os.makedirs(TEXTURE_DIR, exist_ok=True)

    for name, cfg in ENV_PROMPTS.items():
        w, h = cfg["size"]
        prompt = cfg["prompt"]
        if name == "menu_bg":
            out_path = os.path.join(GAME_ASSETS, "menu_bg.png")
        else:
            out_path = os.path.join(TEXTURE_DIR, f"{name}.png")

        print(f"  Generating {name} ({w}x{h}) with {NUM_STEPS} steps...")
        try:
            result = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_STEPS,
                guidance_scale=0.0,
                height=h,
                width=w,
            )
            result.images[0].save(out_path, optimize=True)
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clean_old_assets():
    """Delete all previously generated assets so they get regenerated fresh."""
    count = 0
    for item_type in ITEM_TYPES:
        type_dir = os.path.join(ITEM_DIR, item_type)
        if os.path.isdir(type_dir):
            for f in os.listdir(type_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(type_dir, f))
                    count += 1
    for name in ENV_PROMPTS:
        if name == "menu_bg":
            p = os.path.join(GAME_ASSETS, "menu_bg.png")
        else:
            p = os.path.join(TEXTURE_DIR, f"{name}.png")
        if os.path.exists(p):
            os.remove(p)
            count += 1
    progress_file = os.path.join(GAME_ASSETS, "gen_progress.json")
    if os.path.exists(progress_file):
        os.remove(progress_file)
    print(f"Cleaned {count} old asset files.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate game assets with SDXL Lightning (HQ V2)")
    parser.add_argument("--items-only", action="store_true")
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress (skip cleaning)")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete old assets before generating")
    args = parser.parse_args()

    print("=" * 60)
    print("DiabloRemix Asset Generator V2 (High Quality)")
    print(f"Output: {GAME_ASSETS}")
    print(f"Render: {RENDER_SIZE}x{RENDER_SIZE} -> Final items: {FINAL_ITEM_SIZE}x{FINAL_ITEM_SIZE}")
    print(f"Steps: {NUM_STEPS}")
    print("=" * 60)

    if not args.resume and not args.no_clean:
        print("\n=== Cleaning old assets ===")
        clean_old_assets()

    pipe = setup_pipeline()

    if not args.items_only:
        print("\n=== Generating Environment Assets ===")
        generate_environments(pipe)

    if not args.env_only:
        print("\n=== Generating Item Icons ===")
        generate_items(pipe, resume_from=args.resume)

    print("\nAll done!")


if __name__ == "__main__":
    main()
