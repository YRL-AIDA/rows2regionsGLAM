import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as EmbFontTokenizer
from rows2regionsGLAM.tokenizers.font_tokenizer import RowGLAMTokenizer as PDFFontTokenizer
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer as NoFontTokenizer

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    emb_font_tokenizer = EmbFontTokenizer()
    pdf_font_tokenizer = PDFFontTokenizer()
    no_font_tokenizer = NoFontTokenizer()

    def get_tokenizer(name, params):
        if name.startswith("font_emb"):
            return emb_font_tokenizer
        if name.startswith("pdf_font"):
            return pdf_font_tokenizer
        return no_font_tokenizer

    configs = {}
    for seed in range(3):
        configs[f"font_emb_seed_{seed}"] = {
            **default_arch(input_dim=527, epochs=EPOCHS, early_stopping_patience=3, seed=seed),
            "_cache_dir": "tmp_feature_font_emb",
        }
        configs[f"pdf_font_seed_{seed}"] = {
            **default_arch(input_dim=18, epochs=EPOCHS, early_stopping_patience=3, seed=seed),
            "_cache_dir": "tmp_feature_pdf_font",
        }
        configs[f"no_font_seed_{seed}"] = {
            **default_arch(input_dim=15, epochs=EPOCHS, early_stopping_patience=3, seed=seed),
            "_cache_dir": "tmp_feature_no_font",
        }

    runner = ExperimentRunner("result", get_tokenizer=get_tokenizer)
    runner.run(configs)
