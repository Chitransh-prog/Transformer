"""
Training pipeline for the Transformer (Vaswani et al., 2017)
Chitransh [03529802024]  |  Parth Kapoor [01129802024]

Covers:
  - Word-level tokenizer & vocabulary builder
  - English → French bilingual dataset (offline, no downloads needed)
  - Adam optimiser  β₁=0.9, β₂=0.98, ε=1e-9  (Section 5.3)
  - Label smoothing  ε=0.1  (Section 5.4)
  - Warm-up LR schedule  (Section 5.3)
  - Loss / perplexity logging & loss curve plot

Usage:
    python train.py              # quick run with defaults
    python train.py --epochs 30  # deeper training
"""

import os, re, random, json, argparse
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformer import Transformer, make_masks, WarmupScheduler

# ── Constants ─────────────────────────────────────────────────────────────────
PAD, SOS, EOS, UNK = 0, 1, 2, 3

# ── Vocabulary / Tokenizer ────────────────────────────────────────────────────

class Vocab:
    """Minimal word-level vocab with four special tokens."""
    SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __init__(self):
        self.w2i = {t: i for i, t in enumerate(self.SPECIAL)}
        self.i2w = {i: t for i, t in enumerate(self.SPECIAL)}

    def build(self, sentences):
        counts = {}
        for s in sentences:
            for w in self._tok(s):
                counts[w] = counts.get(w, 0) + 1
        for w in sorted(counts, key=lambda x: -counts[x]):
            if w not in self.w2i:
                i = len(self.w2i)
                self.w2i[w] = i
                self.i2w[i] = w

    def encode(self, text):
        return [SOS] + [self.w2i.get(w, UNK) for w in self._tok(text)] + [EOS]

    def decode(self, ids):
        skip = {PAD, SOS, EOS}
        return " ".join(self.i2w.get(i, "<unk>") for i in ids if i not in skip)

    @staticmethod
    def _tok(text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def save(self, path):
        json.dump(self.w2i, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        v = cls()
        v.w2i = json.load(open(path, encoding="utf-8"))
        v.i2w = {int(i): w for w, i in v.w2i.items()}
        return v

    def __len__(self):
        return len(self.w2i)


# ── Bilingual Corpus (English → French, offline) ──────────────────────────────

BASE_PAIRS = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good evening", "bonsoir"),
    ("how are you ?", "comment allez-vous ?"),
    ("i am fine , thank you .", "je vais bien , merci ."),
    ("what is your name ?", "comment vous appelez-vous ?"),
    ("my name is chitransh .", "mon nom est chitransh ."),
    ("nice to meet you .", "enchanté de faire votre connaissance ."),
    ("where is the library ?", "où est la bibliothèque ?"),
    ("the library is near the university .", "la bibliothèque est près de l' université ."),
    ("i study computer science .", "j' étudie l' informatique ."),
    ("we are working on deep learning .", "nous travaillons sur l' apprentissage profond ."),
    ("attention is all you need .", "l' attention est tout ce dont vous avez besoin ."),
    ("you only need attention .", "vous avez seulement besoin d' attention ."),
    ("we need attention to build transformers .", "nous avons besoin d' attention pour construire des transformateurs ."),
    ("transformers replace recurrent neural networks .", "les transformateurs remplacent les réseaux de neurones récurrents ."),
    ("multi head attention is powerful .", "l' attention multi-têtes est puissante ."),
    ("attention helps the model focus .", "l' attention aide le modèle à se concentrer ."),
    ("i like to read books .", "j' aime lire des livres ."),
    ("she is writing a paper .", "elle écrit un article ."),
    ("he is learning machine learning .", "il apprend l' apprentissage automatique ."),
    ("the weather is very nice today .", "le temps est très beau aujourd' hui ."),
    ("can you help me please ?", "pouvez-vous m' aider s' il vous plaît ?"),
    ("yes , i can help you .", "oui , je peux vous aider ."),
    ("i am going to college .", "je vais à l' université ."),
    ("we have an exam tomorrow .", "nous avons un examen demain ."),
    ("the model has six layers .", "le modèle a six couches ."),
    ("the encoder encodes the input .", "l' encodeur encode l' entrée ."),
    ("the decoder generates the output .", "le décodeur génère la sortie ."),
    ("this is a college project .", "c' est un projet universitaire ."),
    ("they are studying mathematics .", "ils étudient les mathématiques ."),
    ("knowledge is power .", "le savoir est le pouvoir ."),
    ("practice makes perfect .", "la pratique rend parfait ."),
    ("i love coding in python .", "j' adore coder en python ."),
    ("pytorch is very easy to use .", "pytorch est très facile à utiliser ."),
    ("neural networks learn representations .", "les réseaux de neurones apprennent des représentations ."),
    ("see you later .", "à plus tard ."),
    ("goodbye and have a nice day .", "au revoir et bonne journée ."),
]

def build_corpus(n_synthetic=800):
    """Expand the base pairs with combinatorial subject-verb-object sentences."""
    subjects = [("i","je"),("you","vous"),("he","il"),("she","elle"),("we","nous"),("they","ils")]
    verbs    = [("love","aime","aimons","aiment"),
                ("read","lis","lisons","lisent"),
                ("write","écris","écrivons","écrivent"),
                ("understand","comprends","comprenons","comprennent")]
    objects  = [("the book","le livre"),("the paper","l' article"),
                ("the code","le code"),("the model","le modèle"),
                ("the neural network","le réseau de neurones"),
                ("the project","le projet"),("the result","le résultat"),
                ("the language","la langue"),("the architecture","l' architecture")]

    pairs = list(BASE_PAIRS)
    rng = random.Random(42)
    for _ in range(n_synthetic):
        se, sf = rng.choice(subjects)
        v      = rng.choice(verbs)
        oe, of = rng.choice(objects)
        vf = v[2] if se == "we" else (v[3] if se == "they" else v[1])
        pairs.append((f"{se} {v[0]} {oe} .", f"{sf} {vf} {of} ."))
    return pairs


def load_tatoeba_corpus(num_pairs=15000, max_len=12):
    """
    Downloads and caches the official Tatoeba English-French corpus (Google / Stanford mirror).
    Contains ~167,000 real human-translated bilingual sentences.
    Filters to clean sentence pairs of length <= max_len for 30-60 min college-scale training.
    """
    import urllib.request, zipfile
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "fra-eng.zip")
    txt_path = os.path.join(data_dir, "fra.txt")

    if not os.path.exists(txt_path):
        print("[*] Downloading standard Tatoeba English-French benchmark dataset (~3.4 MB)...")
        url = "https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("[✓] Tatoeba dataset downloaded and extracted successfully.")

    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en = parts[0].strip().lower()
                fr = parts[1].strip().lower()
                if len(en.split()) <= max_len and len(fr.split()) <= max_len:
                    pairs.append((en, fr))
                    if len(pairs) >= num_pairs:
                        break
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = [
            (torch.tensor(src_vocab.encode(s), dtype=torch.long),
             torch.tensor(tgt_vocab.encode(t), dtype=torch.long))
            for s, t in pairs
        ]

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate(batch):
    srcs, tgts = zip(*batch)
    return (nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD),
            nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD))


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, opt, device, train=True):
    model.train(train)
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask, tgt_mask = make_masks(src, tgt_in, PAD)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total += loss.item()
    return total / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    type=str,   default="demo", choices=["demo", "tatoeba"],
                    help="'demo' (838 pairs, ~60s CPU) or 'tatoeba' (15k-30k real pairs, 30-60m CPU)")
    ap.add_argument("--num_pairs",  type=int,   default=15000, help="Number of pairs when using tatoeba")
    ap.add_argument("--max_len",    type=int,   default=12,    help="Maximum sentence length (words)")
    ap.add_argument("--epochs",     type=int,   default=12)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--d_model",    type=int,   default=128)
    ap.add_argument("--n_heads",    type=int,   default=4)
    ap.add_argument("--n_layers",   type=int,   default=2)
    ap.add_argument("--d_ff",       type=int,   default=512)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--warmup",     type=int,   default=400)
    ap.add_argument("--save_dir",   type=str,   default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset == "tatoeba":
        print(f"[*] Loading Tatoeba benchmark corpus ({args.num_pairs:,} sentence pairs)...")
        pairs = load_tatoeba_corpus(num_pairs=args.num_pairs, max_len=args.max_len)
    else:
        print("[*] Loading offline demo corpus (838 pairs)...")
        pairs = build_corpus(n_synthetic=800)
    src_vocab, tgt_vocab = Vocab(), Vocab()
    src_vocab.build([p[0] for p in pairs])
    tgt_vocab.build([p[1] for p in pairs])
    src_vocab.save(f"{args.save_dir}/src_vocab.json")
    tgt_vocab.save(f"{args.save_dir}/tgt_vocab.json")
    print(f"Corpus: {len(pairs)} pairs  |  Src vocab: {len(src_vocab)}  |  Tgt vocab: {len(tgt_vocab)}")

    dataset  = TranslationDataset(pairs, src_vocab, tgt_vocab)
    n_train  = int(0.85 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab=len(src_vocab), tgt_vocab=len(tgt_vocab),
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Adam β₁=0.9, β₂=0.98, ε=1e-9  (Section 5.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, args.d_model, args.warmup)
    # Label smoothing ε=0.1  (Section 5.4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, best_val = [], [], float("inf")

    print(f"\n{'Ep':>4}  {'Train':>9}  {'Val':>9}  {'PPL':>7}")
    print("─" * 36)

    for ep in range(1, args.epochs + 1):
        t_loss = run_epoch(model, train_dl, criterion, scheduler, device, train=True)
        v_loss = run_epoch(model, val_dl,   criterion, None,      device, train=False)
        ppl    = torch.exp(torch.tensor(v_loss)).item()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"{ep:>4}  {t_loss:>9.4f}  {v_loss:>9.4f}  {ppl:>7.2f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                "model_state": model.state_dict(),
                "cfg": vars(args),
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
            }, f"{args.save_dir}/best_model.pt")

    print(f"\nBest val loss: {best_val:.4f}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, "o-", label="Train")
    plt.plot(epochs, val_losses,   "s-", label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss (label-smoothed CE)")
    plt.title("Transformer Training Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/loss_curve.png", dpi=150)
    print(f"Loss curve → {args.save_dir}/loss_curve.png")


if __name__ == "__main__":
    main()
