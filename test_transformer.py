import torch
import torch.nn as nn
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from train import greedy_decode, run_validation

class DummyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id_map = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token_map = {idx: tok for idx, tok in enumerate(vocab)}

    def token_to_id(self, token: str) -> int:
        return self.token_to_id_map.get(token, self.token_to_id_map.get("[UNK]", 0))

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str):
        class Encoding:
            def __init__(self, ids):
                self.ids = ids
        tokens = text.lower().split()
        ids = [self.token_to_id(tok) for tok in tokens]
        return Encoding(ids)

    def decode(self, ids) -> str:
        tokens = [self.id_to_token_map.get(i, "[UNK]") for i in ids]
        return " ".join(tokens)


def run_tests():
    print("=== TEST 1: Causal Mask ===")
    mask = causal_mask(4)
    expected_mask = torch.tensor([[[1, 0, 0, 0],
                                   [1, 1, 0, 0],
                                   [1, 1, 1, 0],
                                   [1, 1, 1, 1]]], dtype=torch.bool)
    assert (mask == expected_mask).all(), f"Causal mask failed!\nGot:\n{mask}\nExpected:\n{expected_mask}"
    print("✓ Causal mask test passed.")

    print("\n=== TEST 2: BilingualDataset & DataLoader ===")
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]", "hello", "world", "ciao", "mondo"]
    tok_src = DummyTokenizer(special_tokens)
    tok_tgt = DummyTokenizer(special_tokens)

    raw_data = [
        {"translation": {"en": "hello world", "it": "ciao mondo"}},
        {"translation": {"en": "world", "it": "mondo"}}
    ]

    seq_len = 10
    dataset = BilingualDataset(raw_data, tok_src, tok_tgt, "en", "it", seq_len)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0['encoder_input'].shape == (seq_len,)
    assert item0['decoder_input'].shape == (seq_len,)
    assert item0['label'].shape == (seq_len,)
    assert item0['encoder_mask'].shape == (1, 1, seq_len)
    assert item0['decoder_mask'].shape == (1, seq_len, seq_len)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch['encoder_input'].shape == (2, seq_len)
    assert batch['decoder_input'].shape == (2, seq_len)
    assert batch['label'].shape == (2, seq_len)
    assert batch['encoder_mask'].shape == (2, 1, 1, seq_len)
    assert batch['decoder_mask'].shape == (2, 1, seq_len, seq_len)
    print("✓ BilingualDataset & DataLoader shapes test passed.")

    print("\n=== TEST 3: Transformer Model Forward & Backward ===")
    model = build_transformer(
        src_vocab_size=len(special_tokens),
        tgt_vocab_size=len(special_tokens),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=64,
        N=2,
        h=4,
        dropout=0.1,
        d_ff=128
    )

    encoder_out = model.encode(batch['encoder_input'], batch['encoder_mask'])
    assert encoder_out.shape == (2, seq_len, 64)

    decoder_out = model.decode(encoder_out, batch['encoder_mask'], batch['decoder_input'], batch['decoder_mask'])
    assert decoder_out.shape == (2, seq_len, 64)

    proj_out = model.project(decoder_out)
    assert proj_out.shape == (2, seq_len, len(special_tokens))

    loss_fn = nn.CrossEntropyLoss(ignore_index=tok_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
    loss = loss_fn(proj_out.view(-1, len(special_tokens)), batch['label'].view(-1))
    assert not torch.isnan(loss)
    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("✓ Model encode, decode, project, backward, and optimizer step passed.")

    print("\n=== TEST 4: Greedy Decode ===")
    device = torch.device("cpu")
    single_enc_input = batch['encoder_input'][0:1]
    single_enc_mask = batch['encoder_mask'][0:1]
    generated_tokens = greedy_decode(model, single_enc_input, single_enc_mask, tok_src, tok_tgt, max_len=seq_len, device=device)
    assert generated_tokens.dim() == 1
    assert generated_tokens.size(0) <= seq_len
    decoded_text = tok_tgt.decode(generated_tokens.tolist())
    print(f"Generated tokens: {generated_tokens.tolist()} -> text: '{decoded_text}'")
    print("✓ Greedy decode test passed.")

    print("\n=== TEST 5: Config & Checkpoint Save/Load ===")
    import tempfile, shutil
    from pathlib import Path
    temp_folder_name = "test_weights"
    weights_dir = Path(f"test_{temp_folder_name}")
    try:
        cfg = get_config()
        cfg['model_folder'] = temp_folder_name
        cfg['datasource'] = 'test'
        weights_dir.mkdir(parents=True, exist_ok=True)

        w_path = get_weights_file_path(cfg, "00")
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': 10
        }, w_path)

        latest = latest_weights_file_path(cfg)
        assert latest == w_path

        checkpoint = torch.load(latest)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        assert checkpoint['epoch'] == 0
        assert checkpoint['global_step'] == 10
        print("✓ Checkpoint save and reload test passed.")
    finally:
        if weights_dir.exists():
            shutil.rmtree(weights_dir, ignore_errors=True)

    print("\n=== TEST 6: Validation loop with run_validation ===")
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    printed = []
    run_validation(model, val_loader, tok_src, tok_tgt, max_len=seq_len, device=device, print_msg=lambda m: printed.append(m), global_step=0, writer=None, num_examples=1)
    assert len(printed) > 0
    print("✓ run_validation test passed.")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
