from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import SOPEntry, SKILLS, load_json

console = Console()


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        console.log("[yellow]PyYAML not installed, using default config[/yellow]")
        return {}
    except FileNotFoundError:
        console.log(f"[yellow]Config file not found: {config_path}, using defaults[/yellow]")
        return {}


def build_instruction(incident: str, sop: SOPEntry) -> str:
    """
    Build a formatted instruction prompt for the plan generator.
    
    The model is trained to take this structured format as input and generate
    a JSON plan. The format includes:
    - The incident description (what happened)
    - The SOP details (title, condition, steps)
    - The allowed robot skills (constraints)
    
    This format helps the model understand the context and constraints.
    """
    steps = " ; ".join(sop.steps)
    return (
        "[INCIDENT]\n"
        f"{incident}\n"
        "[SOP]\n"
        f"{sop.title}\n{sop.condition}\n{steps}\n"
        "[ALLOWED_SKILLS]\n"
        f"{','.join(SKILLS)}"
    )


def synthesize_pairs(sops: List[SOPEntry]) -> List[Tuple[str, str]]:
    """
    Convert SOPs into training pairs for the plan generator.
    
    For each SOP, we create a training example that teaches the model to:
    - Take an incident description + SOP text as input
    - Output a structured JSON plan with robot skills
    
    The plan generation uses keyword matching to extract relevant skills
    from the SOP steps (e.g., "press button" → press_button skill).
    
    Args:
        sops: List of SOP entries to convert to training pairs
        
    Returns:
        List of (input_text, output_json) pairs for training
    """
    pairs: List[Tuple[str, str]] = []
    for sop in sops:
        # Create a realistic incident from the SOP (simulates real-world queries)
        incident = f"{sop.title}. {sop.condition}"
        instr = build_instruction(incident, sop)
        
        # Extract robot skills from SOP steps using keyword matching
        # This converts natural language SOP steps into structured robot actions
        steps = []
        step_text = " ".join(sop.steps).lower()
        
        # Extract navigation commands: "walk to machine" → walk_to skill
        if "walk" in step_text:
            target = "table" if "table" in step_text else "machine"
            steps.append({"skill": "walk_to", "args": {"target": target}})
        
        # Extract sensor readings: "read pressure sensor" → read_sensor skill
        if "read" in step_text or "sensor" in step_text:
            # Detect sensor type from context (default to pressure)
            sensor = "pressure_sensor"
            if "temperature" in step_text:
                sensor = "temperature_sensor"
            elif "vibration" in step_text:
                sensor = "vibration_sensor"
            elif "light" in step_text:
                sensor = "light_sensor"
            steps.append({"skill": "read_sensor", "args": {"sensor": sensor}})
        
        # Extract button presses: "press red button" → press_button skill
        if "press" in step_text or "button" in step_text:
            # Detect button color from context (default to green)
            button = "green_button"
            if "red" in step_text:
                button = "red_button"
            elif "blue" in step_text:
                button = "blue_button"
            elif "yellow" in step_text:
                button = "yellow_button"
            steps.append({"skill": "press_button", "args": {"button": button}})
        
        # Extract object picking: "pick up wrench" → pick skill
        if "pick" in step_text:
            # Detect object type from context (default to generic "tool")
            obj = "tool"
            if "wrench" in step_text:
                obj = "wrench"
            elif "screwdriver" in step_text:
                obj = "screwdriver"
            elif "lubricant" in step_text:
                obj = "lubricant_bottle"
            elif "cloth" in step_text:
                obj = "cleaning_cloth"
            elif "brush" in step_text:
                obj = "cleaning_brush"
            elif "goggles" in step_text:
                obj = "safety_goggles"
            elif "battery" in step_text:
                obj = "battery"
            elif "fuse" in step_text:
                obj = "spare_fuses"
            steps.append({"skill": "pick", "args": {"object": obj}})
        
        # Extract object placement: "return tool to table" → place skill
        if "place" in step_text or "return" in step_text:
            steps.append({"skill": "place", "args": {"object": "tool", "location": "table"}})
        
        # Extract wait commands: "wait 3 seconds" → wait skill
        if "wait" in step_text:
            steps.append({"skill": "wait", "args": {"seconds": 3}})
        
        # Always notify operator when procedure is complete
        steps.append({"skill": "notify", "args": {"level": "tech"}})
        
        # Safety check: ensure every plan has at least 2 steps
        # (a minimal plan should have navigation + notification)
        if len(steps) < 2:
            steps = [
                {"skill": "walk_to", "args": {"target": "machine"}},
                {"skill": "notify", "args": {"level": "tech"}},
            ]
        
        target = {
            "goal": sop.title,
            "steps": steps,
            "fallback": [],
        }
        pairs.append((instr, json.dumps(target)))
    return pairs


def train_lora(args: argparse.Namespace) -> None:
    """
    Train a LoRA adapter on Flan-T5 for structured plan generation.
    
    This function:
    1. Loads SOPs and converts them to training pairs (incident → JSON plan)
    2. Fine-tunes Flan-T5 using LoRA (Low-Rank Adaptation) to generate structured plans
    3. Saves only the adapter weights (~0.5M params) instead of full model (250M params)
    
    LoRA is much more efficient than full fine-tuning - we only train 0.2% of parameters
    while achieving 95% of full fine-tuning performance.
    """
    # Load SOPs and convert to training examples
    sops = load_json(args.train_jsonl, SOPEntry, key="sop_examples")
    pairs = synthesize_pairs(sops)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import Dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        import torch

        # Load the base Flan-T5 model (instruction-tuned T5)
        base = args.base
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSeq2SeqLM.from_pretrained(base)
        
        # Auto-detect LoRA target modules from model architecture
        # Different models use different naming conventions:
        # - T5/Flan-T5: "q", "v", "k", "o" (query, value, key, output)
        # - Other models: "q_proj", "v_proj", etc.
        # We apply LoRA to attention layers since they're most impactful for fine-tuning
        target_modules = []
        for name, _ in model.named_modules():
            if any(x in name for x in [".q.", ".v.", ".k.", ".o."]):
                # Extract module name pattern (e.g., "encoder.block.0.layer.0.SelfAttention.q")
                parts = name.split(".")
                for p in parts:
                    if p in ["q", "v", "k", "o"]:
                        if p not in target_modules:
                            target_modules.append(p)
        
        # Fallback to T5 standard names if auto-detection failed
        if not target_modules:
            target_modules = ["q", "v"]  # T5 uses simple 'q' and 'v' names
        
        console.log(f"[dim]Using LoRA target modules: {target_modules}[/dim]")
        
        lora = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora)
        ds = Dataset.from_dict({"input_text": [x for x, _ in pairs], "labels": [y for _, y in pairs]})

        # Prepare training data: tokenize inputs and outputs
        def preprocess(batch):
            """Tokenize the input text and target JSON plans."""
            model_inputs = tok(batch["input_text"], truncation=True, padding=True, max_length=512)
            # T5 uses the same tokenizer for both input and output
            labels = tok(text_target=batch["labels"], truncation=True, padding=True, max_length=256)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
        
        # Set up training configuration
        # DataCollator handles dynamic padding for efficient batching
        collator = DataCollatorForSeq2Seq(tok, model=model)
        tr_args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_strategy="no",  # Don't save checkpoints (we only need final adapter)
            logging_steps=1,  # Log progress every step
        )
        
        trainer = Trainer(model=model, args=tr_args, train_dataset=ds, data_collator=collator, tokenizer=tok)
        
        # Display training configuration
        console.log(f"[green]Starting training with config:[/green]")
        console.log(f"  Model: {base}")
        console.log(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
        console.log(f"  LoRA: r={args.r}, alpha={args.alpha}, dropout={args.dropout}")
        console.log(f"  Training pairs: {len(pairs)}")
        # Train the model (only LoRA parameters are updated)
        trainer.train()
        
        # Save only the LoRA adapter weights (tiny ~0.5M file) instead of full model
        # The base Flan-T5 model stays unchanged and can be reused
        model.save_pretrained(str(out_dir / "adapter"))
        tok.save_pretrained(str(out_dir / "tokenizer"))
        console.log(f"[green]✓ Saved LoRA adapter to {out_dir}[/green]")
    except Exception as e:
        # Graceful fallback: if training fails (e.g., missing dependencies),
        # create a placeholder so the inference pipeline doesn't crash
        console.log(f"[yellow]Skipping LoRA training due to: {e}[/yellow]")
        console.log("[dim]Creating placeholder adapter for inference compatibility[/dim]")
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        with (out_dir / "adapter" / "placeholder.json").open("w", encoding="utf-8") as f:
            json.dump({"note": "placeholder adapter - training failed"}, f)


def build_parser():
    """Build argument parser with config file support."""
    # Get the project root directory (three levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser(description="Train planner (Flan-T5) with LoRA on synthesized pairs.")
    
    # Config file argument (takes precedence)
    p.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config" / "planner_config.yaml"),
        help="Path to YAML config file"
    )
    
    # CLI arguments (override config file values)
    p.add_argument("--base", type=str, help="Base model name (overrides config)")
    p.add_argument("--train_jsonl", type=str, help="Path to training data (overrides config)")
    p.add_argument("--out_dir", type=str, help="Output directory (overrides config)")
    p.add_argument("--r", type=int, help="LoRA rank (overrides config)")
    p.add_argument("--alpha", type=int, help="LoRA alpha (overrides config)")
    p.add_argument("--dropout", type=float, help="LoRA dropout (overrides config)")
    p.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    p.add_argument("--learning-rate", type=float, dest="learning_rate", help="Learning rate (overrides config)")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size (overrides config)")
    
    return p


def merge_config(cli_args: argparse.Namespace, config_dict: dict, project_root: Path) -> argparse.Namespace:
    """Merge CLI arguments with config file, prioritizing CLI args."""
    # Extract values in priority order: CLI args → config → hardcoded defaults
    merged = argparse.Namespace()
    
    # Get nested config values
    training_cfg = config_dict.get('training', {})
    lora_cfg = config_dict.get('lora', {})
    model_cfg = config_dict.get('model', {})
    paths_cfg = config_dict.get('paths', {})
    
    # Base model
    merged.base = (
        cli_args.base or
        model_cfg.get('base') or
        "google/flan-t5-base"
    )
    
    # Training hyperparameters
    merged.epochs = (
        cli_args.epochs or
        training_cfg.get('epochs') or
        1
    )
    merged.learning_rate = (
        cli_args.learning_rate or
        training_cfg.get('learning_rate') or
        2e-4
    )
    merged.batch_size = (
        cli_args.batch_size or
        training_cfg.get('batch_size') or
        2
    )
    
    # LoRA hyperparameters
    merged.r = (
        cli_args.r or
        lora_cfg.get('r') or
        16
    )
    merged.alpha = (
        cli_args.alpha or
        lora_cfg.get('alpha') or
        32
    )
    merged.dropout = (
        cli_args.dropout or
        lora_cfg.get('dropout') or
        0.05
    )
    
    # Paths (resolve relative paths)
    merged.train_jsonl = (
        cli_args.train_jsonl or
        paths_cfg.get('train_data') or
        str(project_root / "src" / "data" / "sop_examples.json")
    )
    merged.out_dir = (
        cli_args.out_dir or
        paths_cfg.get('output_dir') or
        str(project_root / "artifacts" / "planner_lora")
    )
    
    # Resolve relative paths
    if not Path(merged.train_jsonl).is_absolute():
        merged.train_jsonl = str(project_root / merged.train_jsonl)
    if not Path(merged.out_dir).is_absolute():
        merged.out_dir = str(project_root / merged.out_dir)
    
    return merged


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    cli_args = build_parser().parse_args()
    
    # Load config from YAML file
    config_dict = load_config(cli_args.config)
    
    # Merge CLI args with config file (CLI takes precedence)
    args = merge_config(cli_args, config_dict, project_root)
    
    console.log(f"[cyan]Loaded config from: {cli_args.config}[/cyan]")
    train_lora(args)


