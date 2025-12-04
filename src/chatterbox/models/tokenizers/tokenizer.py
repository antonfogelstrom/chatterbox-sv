import logging
import json

import torch
from pathlib import Path
from unicodedata import category, normalize
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


# Model repository
REPO_ID = "ResembleAI/chatterbox"


class MTLTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def preprocess_text(
        self,
        raw_text: str,
        language_id: str = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)

        return preprocessed_text

    def text_to_tokens(
        self,
        text: str,
        language_id: str = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ):
        text_tokens = self.encode(
            text,
            language_id=language_id,
            lowercase=lowercase,
            nfkd_normalize=nfkd_normalize,
        )
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(
        self,
        txt: str,
        language_id: str = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ):
        txt = self.preprocess_text(
            txt,
            language_id=language_id,
            lowercase=lowercase,
            nfkd_normalize=nfkd_normalize,
        )

        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"

        txt = txt.replace(" ", SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(" ", "").replace(SPACE, " ").replace(EOT, "").replace(UNK, "")
        return txt
