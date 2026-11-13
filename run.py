"""
Inference, BLEU evaluation & attention visualiser for the Transformer.
Chitransh [03529802024]  |  Parth Kapoor [01129802024]

Usage:
    # Interactive translation loop
    python run.py

    # Translate a single sentence
    python run.py --text "i love coding in python ."

    # Evaluate BLEU on the test split & save attention heatmap
    python run.py --bleu --plot
"""

import os, re, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sacrebleu

from transformer import Transformer, pad_mask, causal_mask
from train import Vocab, build_corpus, load_tatoeba_corpus, PAD, SOS, EOS

CHECKPOINT = "checkpoints/best_model.pt"
VOCAB_DIR  = "checkpoints"


# ── Load model ────────────────────────────────────────────────────────────────

def load(device):
    ckpt      = torch.load(CHECKPOINT, map_location=device)
    cfg       = ckpt["cfg"]
    src_vocab = Vocab.load(f"{VOCAB_DIR}/src_vocab.json")
    tgt_vocab = Vocab.load(f"{VOCAB_DIR}/tgt_vocab.json")

    model = Transformer(
        src_vocab=len(src_vocab), tgt_vocab=len(tgt_vocab),
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab


# ── Greedy decoding ───────────────────────────────────────────────────────────

@torch.no_grad()
def translate(model, src_vocab, tgt_vocab, text, device, max_len=50):
    tokens   = torch.tensor([src_vocab.encode(text)], dtype=torch.long, device=device)
    src_mask = pad_mask(tokens, PAD)
    enc_out  = model.encode(tokens, src_mask)

    generated   = [SOS]
    cross_attns = []

    for _ in range(max_len):
        tgt      = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = causal_mask(tgt.size(1), device)
        dec_out  = model.decode(tgt, enc_out, src_mask, tgt_mask)
        logits   = model.output_proj(dec_out[:, -1, :])
        next_tok = logits.argmax(-1).item()
        generated.append(next_tok)

        # Collect cross-attention weights from the last decoder layer
        cross_attns.append(model.decoder[-1].cross_attn.attn_weights.cpu())

        if next_tok == EOS:
            break

        # Loop guard: stop if the last 6 tokens contain a 2-token repeating pattern
        if len(generated) >= 6:
            tail = generated[-6:]
            if tail[:3] == tail[3:] or len(set(tail)) <= 2:
                break

    return tgt_vocab.decode(generated), tokens[0].tolist(), generated, cross_attns


# ── Attention heatmap ─────────────────────────────────────────────────────────

def plot_attention(src_tokens, tgt_tokens, cross_attns, path="attention.png"):
    # cross_attns: list of [1, h, 1, src_len]  (one per decoding step)
    # Stack along the target dimension → [tgt_len, h, src_len]
    weights = torch.cat(cross_attns, dim=2)  # [1, h, tgt_len, src_len]
    weights = weights.squeeze(0).numpy()     # [h, tgt_len, src_len]

    n_heads = weights.shape[0]
    cols    = 4
    rows    = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).ravel()

    src_labels = [src_vocab_ref.i2w.get(i, "?") for i in src_tokens]
    tgt_labels = [tgt_vocab_ref.i2w.get(i, "?") for i in tgt_tokens[1:] if i not in (PAD, SOS)]

    for h in range(n_heads):
        ax = axes[h]
        # Clip to actual tgt/src lengths
        w = weights[h, :len(tgt_labels), :len(src_labels)]
        ax.imshow(w, aspect="auto", cmap="Blues")
        ax.set_title(f"Head {h+1}", fontsize=10)
        ax.set_xticks(range(len(src_labels))); ax.set_xticklabels(src_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tgt_labels))); ax.set_yticklabels(tgt_labels, fontsize=8)

    for h in range(n_heads, len(axes)):
        fig.delaxes(axes[h])

    plt.suptitle("Cross-Attention Weights (last decoder layer)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention heatmap → {path}")

# Global ref for labels inside plot_attention (set in main)
src_vocab_ref = None
tgt_vocab_ref = None


# ── BLEU evaluation ───────────────────────────────────────────────────────────

def evaluate_bleu(model, src_vocab, tgt_vocab, device, n=100, dataset_type="demo"):
    if dataset_type == "tatoeba" or os.path.exists("data/fra.txt"):
        pairs = load_tatoeba_corpus(num_pairs=15000)
    else:
        pairs = build_corpus(n_synthetic=800)
    test   = pairs[-n:]
    hyps, refs = [], []

    for src_text, ref_text in test:
        hyp, *_ = translate(model, src_vocab, tgt_vocab, src_text, device)
        hyps.append(hyp)
        refs.append(ref_text)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    print("\nSample translations:")
    print("─" * 70)
    for i in range(min(6, len(test))):
        ok = "✓" if hyps[i].strip() == refs[i].strip() else "≈"
        print(f"{ok} SRC: {test[i][0]}")
        print(f"  REF: {refs[i]}")
        print(f"  HYP: {hyps[i]}")
    print("─" * 70)
    print(f"BLEU: {bleu.score:.2f}  (1-4 gram precisions: {[f'{p:.1f}' for p in bleu.precisions]})")
    return bleu.score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global src_vocab_ref, tgt_vocab_ref

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--bleu", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.exists(CHECKPOINT):
        print("No checkpoint found. Run:  python train.py")
        return

    device = torch.device(args.device)
    model, src_vocab, tgt_vocab = load(device)
    src_vocab_ref = src_vocab
    tgt_vocab_ref = tgt_vocab
    print(f"Model loaded on {device}\n")

    if args.bleu:
        evaluate_bleu(model, src_vocab, tgt_vocab, device)

    if args.text:
        pred, src_ids, tgt_ids, cross_attns = translate(model, src_vocab, tgt_vocab, args.text, device)
        print(f"Input  : {args.text}")
        print(f"Output : {pred}")
        if args.plot:
            plot_attention(src_ids, tgt_ids, cross_attns)
        return

    # Interactive loop
    print("Interactive translation  (type 'quit' to exit)")
    print("─" * 50)
    samples = [
        "hello",
        "i study computer science .",
        "attention is all you need .",
        "the encoder encodes the input .",
        "we are working on deep learning .",
    ]
    for s in samples:
        print(f"  Try: {s}")
    print()

    while True:
        try:
            text = input("EN> ").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            pred, src_ids, tgt_ids, cross_attns = translate(model, src_vocab, tgt_vocab, text, device)
            print(f"FR> {pred}\n")
            if args.plot:
                plot_attention(src_ids, tgt_ids, cross_attns)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

