import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as FontTokenizer
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer as NoFontTokenizer

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    font_tokenizer = FontTokenizer()
    no_font_tokenizer = NoFontTokenizer()

    def get_tokenizer(name, params):
        if "font" in name and "no_font" not in name:
            return font_tokenizer
        return no_font_tokenizer

    runner = ExperimentRunner("result", get_tokenizer=get_tokenizer)
    runner.run({
        "font": {**default_arch(input_dim=527, epochs=EPOCHS, early_stopping_patience=3), "_cache_dir": "tmp_feature_font"},
        "no_font": {**default_arch(input_dim=15, epochs=EPOCHS, early_stopping_patience=3), "_cache_dir": "tmp_feature_no_font"},
    })
