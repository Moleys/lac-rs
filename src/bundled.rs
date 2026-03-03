use crate::runtime::Mode;

pub(crate) struct BundledAssets {
    pub(crate) mode: Mode,
    pub(crate) args_ini: &'static [u8],
    pub(crate) word_dic: &'static [u8],
    pub(crate) q2b_dic: &'static [u8],
    pub(crate) tag_dic: &'static [u8],
}

impl BundledAssets {
    pub(crate) fn param_bytes(&self, name: &str) -> Option<&'static [u8]> {
        match self.mode {
            Mode::Lac => match name {
                "word_emb" => Some(include_bytes!("../models_general/lac_model/model/word_emb")),
                "fc_0.w_0" => Some(include_bytes!("../models_general/lac_model/model/fc_0.w_0")),
                "fc_0.b_0" => Some(include_bytes!("../models_general/lac_model/model/fc_0.b_0")),
                "fc_1.w_0" => Some(include_bytes!("../models_general/lac_model/model/fc_1.w_0")),
                "fc_1.b_0" => Some(include_bytes!("../models_general/lac_model/model/fc_1.b_0")),
                "fc_2.w_0" => Some(include_bytes!("../models_general/lac_model/model/fc_2.w_0")),
                "fc_2.b_0" => Some(include_bytes!("../models_general/lac_model/model/fc_2.b_0")),
                "fc_3.w_0" => Some(include_bytes!("../models_general/lac_model/model/fc_3.w_0")),
                "fc_3.b_0" => Some(include_bytes!("../models_general/lac_model/model/fc_3.b_0")),
                "fc_4.w_0" => Some(include_bytes!("../models_general/lac_model/model/fc_4.w_0")),
                "fc_4.b_0" => Some(include_bytes!("../models_general/lac_model/model/fc_4.b_0")),
                "gru_0.w_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_0.w_0"
                )),
                "gru_0.b_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_0.b_0"
                )),
                "gru_1.w_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_1.w_0"
                )),
                "gru_1.b_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_1.b_0"
                )),
                "gru_2.w_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_2.w_0"
                )),
                "gru_2.b_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_2.b_0"
                )),
                "gru_3.w_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_3.w_0"
                )),
                "gru_3.b_0" => Some(include_bytes!(
                    "../models_general/lac_model/model/gru_3.b_0"
                )),
                "crfw" => Some(include_bytes!("../models_general/lac_model/model/crfw")),
                _ => None,
            },
            Mode::Seg => match name {
                "word_emb" => Some(include_bytes!("../models_general/seg_model/model/word_emb")),
                "fc_0.w_0" => Some(include_bytes!("../models_general/seg_model/model/fc_0.w_0")),
                "fc_0.b_0" => Some(include_bytes!("../models_general/seg_model/model/fc_0.b_0")),
                "fc_1.w_0" => Some(include_bytes!("../models_general/seg_model/model/fc_1.w_0")),
                "fc_1.b_0" => Some(include_bytes!("../models_general/seg_model/model/fc_1.b_0")),
                "fc_2.w_0" => Some(include_bytes!("../models_general/seg_model/model/fc_2.w_0")),
                "fc_2.b_0" => Some(include_bytes!("../models_general/seg_model/model/fc_2.b_0")),
                "fc_3.w_0" => Some(include_bytes!("../models_general/seg_model/model/fc_3.w_0")),
                "fc_3.b_0" => Some(include_bytes!("../models_general/seg_model/model/fc_3.b_0")),
                "fc_4.w_0" => Some(include_bytes!("../models_general/seg_model/model/fc_4.w_0")),
                "fc_4.b_0" => Some(include_bytes!("../models_general/seg_model/model/fc_4.b_0")),
                "gru_0.w_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_0.w_0"
                )),
                "gru_0.b_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_0.b_0"
                )),
                "gru_1.w_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_1.w_0"
                )),
                "gru_1.b_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_1.b_0"
                )),
                "gru_2.w_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_2.w_0"
                )),
                "gru_2.b_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_2.b_0"
                )),
                "gru_3.w_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_3.w_0"
                )),
                "gru_3.b_0" => Some(include_bytes!(
                    "../models_general/seg_model/model/gru_3.b_0"
                )),
                "crfw" => Some(include_bytes!("../models_general/seg_model/model/crfw")),
                _ => None,
            },
        }
    }
}

pub(crate) fn bundled_assets(mode: Mode) -> BundledAssets {
    match mode {
        Mode::Lac => BundledAssets {
            mode,
            args_ini: include_bytes!("../models_general/lac_model/conf/args.ini"),
            word_dic: include_bytes!("../models_general/lac_model/conf/word.dic"),
            q2b_dic: include_bytes!("../models_general/lac_model/conf/q2b.dic"),
            tag_dic: include_bytes!("../models_general/lac_model/conf/tag.dic"),
        },
        Mode::Seg => BundledAssets {
            mode,
            args_ini: include_bytes!("../models_general/seg_model/conf/args.ini"),
            word_dic: include_bytes!("../models_general/seg_model/conf/word.dic"),
            q2b_dic: include_bytes!("../models_general/seg_model/conf/q2b.dic"),
            tag_dic: include_bytes!("../models_general/seg_model/conf/tag.dic"),
        },
    }
}
