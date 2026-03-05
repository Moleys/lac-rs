use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(feature = "bundled-models")]
use crate::bundled::bundled_assets;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Lac,
    Seg,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputItem {
    pub word: String,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub max_samples: Option<usize>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 3,
            learning_rate: 0.05,
            max_samples: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReport {
    pub epochs: usize,
    pub samples: usize,
    pub tokens: usize,
    pub token_accuracy: f32,
    pub average_log_likelihood: f32,
}

#[derive(Debug)]
pub enum LacError {
    Parse(String),
    InvalidShape {
        name: String,
        expected: usize,
        got: usize,
    },
    MissingParameter(String),
    MissingBundledModel,
    #[cfg(not(target_arch = "wasm32"))]
    Io {
        path: String,
        source: std::io::Error,
    },
}

impl Display for LacError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LacError::Parse(message) => write!(f, "{message}"),
            LacError::InvalidShape {
                name,
                expected,
                got,
            } => write!(
                f,
                "invalid parameter shape for `{name}`, expected {expected} elements, got {got}"
            ),
            LacError::MissingParameter(name) => write!(f, "missing parameter file `{name}`"),
            LacError::MissingBundledModel => {
                write!(
                    f,
                    "bundled model is disabled, enable `bundled-models` feature"
                )
            }
            #[cfg(not(target_arch = "wasm32"))]
            LacError::Io { path, source } => write!(f, "io error on `{path}`: {source}"),
        }
    }
}

impl std::error::Error for LacError {}

pub struct Lac {
    runtime: ModelRuntime,
}

impl Lac {
    pub fn new(mode: Mode) -> Result<Self, LacError> {
        #[cfg(feature = "bundled-models")]
        {
            return Ok(Self {
                runtime: ModelRuntime::from_bundled(mode)?,
            });
        }

        #[cfg(not(feature = "bundled-models"))]
        {
            let _ = mode;
            Err(LacError::MissingBundledModel)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, LacError> {
        Ok(Self {
            runtime: ModelRuntime::from_directory(model_dir.as_ref(), None)?,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_dir_with_mode(model_dir: impl AsRef<Path>, mode: Mode) -> Result<Self, LacError> {
        Ok(Self {
            runtime: ModelRuntime::from_directory(model_dir.as_ref(), Some(mode))?,
        })
    }

    pub fn mode(&self) -> Mode {
        self.runtime.mode
    }

    pub fn run(&self, query: &str) -> Vec<OutputItem> {
        self.runtime.run_single(query)
    }

    pub fn run_batch<S: AsRef<str>>(&self, queries: &[S]) -> Vec<Vec<OutputItem>> {
        queries
            .iter()
            .map(|query| self.run(query.as_ref()))
            .collect()
    }

    pub fn cut(&self, query: &str) -> Vec<String> {
        self.run(query).into_iter().map(|item| item.word).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn train_from_tsv(
        &mut self,
        train_data: impl AsRef<Path>,
        options: TrainOptions,
    ) -> Result<TrainReport, LacError> {
        self.runtime.train_from_tsv(train_data.as_ref(), &options)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn evaluate_tsv(&self, eval_data: impl AsRef<Path>) -> Result<TrainReport, LacError> {
        self.runtime.evaluate_tsv(eval_data.as_ref())
    }
}

#[derive(Debug)]
struct ModelConfig {
    mode: Mode,
    word_emb_dim: usize,
    grnn_hidden_dim: usize,
    bigru_num: usize,
}

#[derive(Debug)]
struct BigruLayer {
    fc_forward_w: Vec<f32>,
    fc_forward_b: Vec<f32>,
    gru_forward_w: Vec<f32>,
    gru_forward_b: Vec<f32>,
    fc_reverse_w: Vec<f32>,
    fc_reverse_b: Vec<f32>,
    gru_reverse_w: Vec<f32>,
    gru_reverse_b: Vec<f32>,
}

#[derive(Debug)]
struct ModelParams {
    word_emb: Vec<f32>,
    bigru_layers: Vec<BigruLayer>,
    emission_w: Vec<f32>,
    emission_b: Vec<f32>,
    crf_w: Vec<f32>,
}

#[derive(Debug)]
struct ModelRuntime {
    mode: Mode,
    word2id: HashMap<String, i64>,
    q2b: HashMap<String, String>,
    id2label: Vec<String>,
    #[cfg(not(target_arch = "wasm32"))]
    label2id: HashMap<String, usize>,
    oov_id: i64,
    config: ModelConfig,
    vocab_size: usize,
    num_labels: usize,
    params: ModelParams,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
struct TrainingSample {
    token_ids: Vec<usize>,
    gold_tag_ids: Vec<usize>,
}

impl ModelRuntime {
    #[cfg(not(target_arch = "wasm32"))]
    fn from_directory(model_dir: &Path, mode_hint: Option<Mode>) -> Result<Self, LacError> {
        let conf_dir = model_dir.join("conf");
        let param_dir = model_dir.join("model");
        let args_ini = read_file(conf_dir.join("args.ini"))?;
        let word_dic = read_file(conf_dir.join("word.dic"))?;
        let q2b_dic = read_file(conf_dir.join("q2b.dic"))?;
        let tag_dic = read_file(conf_dir.join("tag.dic"))?;

        Self::from_sources(
            mode_hint,
            &args_ini,
            &word_dic,
            &q2b_dic,
            &tag_dic,
            |name| read_file(param_dir.join(name)),
        )
    }

    #[cfg(feature = "bundled-models")]
    fn from_bundled(mode: Mode) -> Result<Self, LacError> {
        let assets = bundled_assets(mode);
        Self::from_sources(
            Some(mode),
            assets.args_ini,
            assets.word_dic,
            assets.q2b_dic,
            assets.tag_dic,
            |name| {
                assets
                    .param_bytes(name)
                    .map(|bytes| bytes.to_vec())
                    .ok_or_else(|| LacError::MissingParameter(name.to_string()))
            },
        )
    }

    fn from_sources<F>(
        mode_hint: Option<Mode>,
        args_ini: &[u8],
        word_dic: &[u8],
        q2b_dic: &[u8],
        tag_dic: &[u8],
        mut read_param: F,
    ) -> Result<Self, LacError>
    where
        F: FnMut(&str) -> Result<Vec<u8>, LacError>,
    {
        let parsed_args = parse_ini(args_ini)?;
        let word_emb_dim = parse_ini_usize(&parsed_args, "word_emb_dim")?;
        let grnn_hidden_dim = parse_ini_usize(&parsed_args, "grnn_hidden_dim")?;
        let bigru_num = parse_ini_usize(&parsed_args, "bigru_num")?;
        if bigru_num == 0 {
            return Err(LacError::Parse("`bigru_num` must be > 0".to_string()));
        }

        let word2id = load_word2id(word_dic)?;
        let q2b = load_q2b(q2b_dic)?;
        let label_map = load_id2label(tag_dic)?;
        #[cfg(not(target_arch = "wasm32"))]
        let label2id = load_label2id(tag_dic)?;
        if word2id.is_empty() {
            return Err(LacError::Parse("word dictionary is empty".to_string()));
        }
        if label_map.is_empty() {
            return Err(LacError::Parse("tag dictionary is empty".to_string()));
        }
        let mode = mode_hint.unwrap_or_else(|| infer_mode(&parsed_args, &label_map));

        let vocab_size = word2id.values().copied().max().unwrap_or(0) as usize + 1;
        let num_labels = label_map.keys().copied().max().unwrap_or(0) as usize + 1;
        let mut id2label = vec!["O".to_string(); num_labels];
        for (id, label) in label_map {
            if id >= 0 && (id as usize) < num_labels {
                id2label[id as usize] = label;
            }
        }

        let mut oov_id = word2id.len().saturating_sub(1) as i64;
        if let Some(id) = word2id.get("OOV") {
            oov_id = *id;
        }

        let config = ModelConfig {
            mode,
            word_emb_dim,
            grnn_hidden_dim,
            bigru_num,
        };

        let word_emb = read_checked_param(
            "word_emb",
            vocab_size * config.word_emb_dim,
            &mut read_param,
        )?;

        let mut bigru_layers = Vec::with_capacity(config.bigru_num);
        for layer_id in 0..config.bigru_num {
            let input_dim = if layer_id == 0 {
                config.word_emb_dim
            } else {
                config.grnn_hidden_dim * 2
            };
            let hidden = config.grnn_hidden_dim;
            let fc_forward_index = layer_id * 2;
            let fc_reverse_index = fc_forward_index + 1;

            let fc_forward_w = read_checked_param(
                &format!("fc_{fc_forward_index}.w_0"),
                input_dim * hidden * 3,
                &mut read_param,
            )?;
            let fc_forward_b = read_checked_param(
                &format!("fc_{fc_forward_index}.b_0"),
                hidden * 3,
                &mut read_param,
            )?;
            let gru_forward_w = read_checked_param(
                &format!("gru_{fc_forward_index}.w_0"),
                hidden * hidden * 3,
                &mut read_param,
            )?;
            let gru_forward_b = read_checked_param(
                &format!("gru_{fc_forward_index}.b_0"),
                hidden * 3,
                &mut read_param,
            )?;

            let fc_reverse_w = read_checked_param(
                &format!("fc_{fc_reverse_index}.w_0"),
                input_dim * hidden * 3,
                &mut read_param,
            )?;
            let fc_reverse_b = read_checked_param(
                &format!("fc_{fc_reverse_index}.b_0"),
                hidden * 3,
                &mut read_param,
            )?;
            let gru_reverse_w = read_checked_param(
                &format!("gru_{fc_reverse_index}.w_0"),
                hidden * hidden * 3,
                &mut read_param,
            )?;
            let gru_reverse_b = read_checked_param(
                &format!("gru_{fc_reverse_index}.b_0"),
                hidden * 3,
                &mut read_param,
            )?;

            bigru_layers.push(BigruLayer {
                fc_forward_w,
                fc_forward_b,
                gru_forward_w,
                gru_forward_b,
                fc_reverse_w,
                fc_reverse_b,
                gru_reverse_w,
                gru_reverse_b,
            });
        }

        let emission_index = config.bigru_num * 2;
        let emission_w = read_checked_param(
            &format!("fc_{emission_index}.w_0"),
            config.grnn_hidden_dim * 2 * num_labels,
            &mut read_param,
        )?;
        let emission_b = read_checked_param(
            &format!("fc_{emission_index}.b_0"),
            num_labels,
            &mut read_param,
        )?;
        let crf_w = read_checked_param("crfw", (num_labels + 2) * num_labels, &mut read_param)?;

        let params = ModelParams {
            word_emb,
            bigru_layers,
            emission_w,
            emission_b,
            crf_w,
        };

        Ok(Self {
            mode: config.mode,
            word2id,
            q2b,
            id2label,
            #[cfg(not(target_arch = "wasm32"))]
            label2id,
            oov_id,
            config,
            vocab_size,
            num_labels,
            params,
        })
    }

    fn run_single(&self, query: &str) -> Vec<OutputItem> {
        let seq_words: Vec<String> = query.chars().map(|ch| ch.to_string()).collect();
        if seq_words.is_empty() {
            return Vec::new();
        }

        let token_ids: Vec<usize> = seq_words
            .iter()
            .map(|word| self.token_id_for_word(word))
            .collect();
        let tag_ids = self.predict_tags(&token_ids);
        self.parse_targets(&seq_words, &tag_ids)
    }

    fn token_id_for_word(&self, word: &str) -> usize {
        let normalized = self.q2b.get(word).map(String::as_str).unwrap_or(word);
        self.word2id
            .get(normalized)
            .copied()
            .unwrap_or(self.oov_id)
            .clamp(0, (self.vocab_size.saturating_sub(1)) as i64) as usize
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn clamp_label_id(&self, id: usize) -> usize {
        if self.num_labels == 0 {
            0
        } else {
            id.min(self.num_labels - 1)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn fallback_label_id(&self) -> usize {
        if let Some(id) = self.label2id.get("O").copied() {
            return self.clamp_label_id(id);
        }
        if let Some(id) = self.label2id.get("-S").copied() {
            return self.clamp_label_id(id);
        }
        0
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn label_id_or_fallback(&self, label: &str, fallback: usize) -> usize {
        self.label2id
            .get(label)
            .copied()
            .map(|id| self.clamp_label_id(id))
            .unwrap_or(fallback)
    }

    fn compute_emission(&self, token_ids: &[usize]) -> Vec<f32> {
        if token_ids.is_empty() {
            return Vec::new();
        }

        let steps = token_ids.len();
        let hidden = self.config.grnn_hidden_dim;

        let mut features = vec![0f32; steps * self.config.word_emb_dim];
        for (row, token_id) in token_ids.iter().copied().enumerate() {
            let base = token_id * self.config.word_emb_dim;
            let out = row * self.config.word_emb_dim;
            features[out..out + self.config.word_emb_dim]
                .copy_from_slice(&self.params.word_emb[base..base + self.config.word_emb_dim]);
        }

        for (layer_index, layer) in self.params.bigru_layers.iter().enumerate() {
            let input_dim = if layer_index == 0 {
                self.config.word_emb_dim
            } else {
                hidden * 2
            };
            let pre_forward = matmul_add_bias(
                &features,
                steps,
                input_dim,
                hidden * 3,
                &layer.fc_forward_w,
                &layer.fc_forward_b,
            );
            let pre_reverse = matmul_add_bias(
                &features,
                steps,
                input_dim,
                hidden * 3,
                &layer.fc_reverse_w,
                &layer.fc_reverse_b,
            );

            let gru_forward = dynamic_gru(
                &pre_forward,
                steps,
                hidden,
                &layer.gru_forward_w,
                &layer.gru_forward_b,
                false,
            );
            let gru_reverse = dynamic_gru(
                &pre_reverse,
                steps,
                hidden,
                &layer.gru_reverse_w,
                &layer.gru_reverse_b,
                true,
            );

            let mut merged = vec![0f32; steps * hidden * 2];
            for row in 0..steps {
                let out = row * hidden * 2;
                merged[out..out + hidden]
                    .copy_from_slice(&gru_forward[row * hidden..(row + 1) * hidden]);
                merged[out + hidden..out + hidden * 2]
                    .copy_from_slice(&gru_reverse[row * hidden..(row + 1) * hidden]);
            }
            features = merged;
        }

        matmul_add_bias(
            &features,
            steps,
            hidden * 2,
            self.num_labels,
            &self.params.emission_w,
            &self.params.emission_b,
        )
    }

    fn predict_tags(&self, token_ids: &[usize]) -> Vec<usize> {
        if token_ids.is_empty() {
            return Vec::new();
        }
        let emission = self.compute_emission(token_ids);
        viterbi_decode(
            &emission,
            token_ids.len(),
            self.num_labels,
            &self.params.crf_w,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn train_from_tsv(
        &mut self,
        train_path: &Path,
        options: &TrainOptions,
    ) -> Result<TrainReport, LacError> {
        if options.epochs == 0 {
            return Err(LacError::Parse("`epochs` must be > 0".to_string()));
        }
        if options.learning_rate <= 0.0 {
            return Err(LacError::Parse("`learning_rate` must be > 0".to_string()));
        }

        let samples = self.load_training_samples(train_path, options.max_samples)?;
        if samples.is_empty() {
            return Err(LacError::Parse(format!(
                "no usable training samples in `{}`",
                train_path.display()
            )));
        }

        let samples_per_epoch = samples.len();
        let tokens_per_epoch: usize = samples.iter().map(|sample| sample.gold_tag_ids.len()).sum();
        let progress_interval = (samples_per_epoch / 20).max(1);
        eprintln!(
            "[train] start mode={:?} epochs={} lr={} samples/epoch={} tokens/epoch={} file={}",
            self.mode,
            options.epochs,
            options.learning_rate,
            samples_per_epoch,
            tokens_per_epoch,
            train_path.display()
        );

        let mut gradient = vec![0f32; self.params.crf_w.len()];
        let mut total_tokens = 0usize;
        let mut total_correct = 0usize;
        let mut total_log_likelihood = 0f32;
        let mut total_samples = 0usize;

        for epoch_index in 0..options.epochs {
            let mut epoch_tokens = 0usize;
            let mut epoch_correct = 0usize;
            let mut epoch_log_likelihood = 0f32;
            let mut epoch_samples = 0usize;

            for (sample_index, sample) in samples.iter().enumerate() {
                if sample.token_ids.is_empty() || sample.gold_tag_ids.is_empty() {
                    continue;
                }
                let emission = self.compute_emission(&sample.token_ids);
                let predicted = viterbi_decode(
                    &emission,
                    sample.token_ids.len(),
                    self.num_labels,
                    &self.params.crf_w,
                );
                total_correct += predicted
                    .iter()
                    .zip(sample.gold_tag_ids.iter())
                    .filter(|(predicted_tag, gold_tag)| predicted_tag == gold_tag)
                    .count();
                total_tokens += sample.gold_tag_ids.len();
                epoch_correct += predicted
                    .iter()
                    .zip(sample.gold_tag_ids.iter())
                    .filter(|(predicted_tag, gold_tag)| predicted_tag == gold_tag)
                    .count();
                epoch_tokens += sample.gold_tag_ids.len();

                let sample_log_likelihood = crf_log_likelihood_and_gradient(
                    &emission,
                    &sample.gold_tag_ids,
                    self.num_labels,
                    &self.params.crf_w,
                    &mut gradient,
                );
                total_log_likelihood += sample_log_likelihood;
                epoch_log_likelihood += sample_log_likelihood;
                epoch_samples += 1;
                for (weight, grad) in self.params.crf_w.iter_mut().zip(gradient.iter().copied()) {
                    *weight += options.learning_rate * grad;
                }
                total_samples += 1;

                let is_checkpoint = (sample_index + 1) % progress_interval == 0
                    || sample_index + 1 == samples_per_epoch;
                if is_checkpoint {
                    let progress = ((sample_index + 1) as f32 / samples_per_epoch as f32) * 100.0;
                    let accuracy = if epoch_tokens == 0 {
                        0.0
                    } else {
                        epoch_correct as f32 / epoch_tokens as f32
                    };
                    let average_ll = if epoch_samples == 0 {
                        0.0
                    } else {
                        epoch_log_likelihood / epoch_samples as f32
                    };
                    eprintln!(
                        "[train] epoch {}/{} progress {:>5.1}% ({}/{}) acc={:.4} avg_ll={:.4}",
                        epoch_index + 1,
                        options.epochs,
                        progress,
                        sample_index + 1,
                        samples_per_epoch,
                        accuracy,
                        average_ll
                    );
                }
            }
        }

        if total_samples == 0 {
            return Err(LacError::Parse(format!(
                "no usable training samples in `{}`",
                train_path.display()
            )));
        }

        Ok(TrainReport {
            epochs: options.epochs,
            samples: total_samples,
            tokens: total_tokens,
            token_accuracy: if total_tokens == 0 {
                0.0
            } else {
                total_correct as f32 / total_tokens as f32
            },
            average_log_likelihood: total_log_likelihood / total_samples as f32,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn evaluate_tsv(&self, eval_path: &Path) -> Result<TrainReport, LacError> {
        let samples = self.load_training_samples(eval_path, None)?;
        if samples.is_empty() {
            return Err(LacError::Parse(format!(
                "no usable evaluation samples in `{}`",
                eval_path.display()
            )));
        }

        eprintln!(
            "[eval] start mode={:?} samples={} file={}",
            self.mode,
            samples.len(),
            eval_path.display()
        );

        let mut total_tokens = 0usize;
        let mut total_correct = 0usize;
        let mut total_log_likelihood = 0f32;

        for sample in &samples {
            let emission = self.compute_emission(&sample.token_ids);
            let predicted = viterbi_decode(
                &emission,
                sample.token_ids.len(),
                self.num_labels,
                &self.params.crf_w,
            );
            total_correct += predicted
                .iter()
                .zip(sample.gold_tag_ids.iter())
                .filter(|(predicted_tag, gold_tag)| predicted_tag == gold_tag)
                .count();
            total_tokens += sample.gold_tag_ids.len();
            total_log_likelihood += crf_log_likelihood(
                &emission,
                &sample.gold_tag_ids,
                self.num_labels,
                &self.params.crf_w,
            );
        }

        Ok(TrainReport {
            epochs: 1,
            samples: samples.len(),
            tokens: total_tokens,
            token_accuracy: if total_tokens == 0 {
                0.0
            } else {
                total_correct as f32 / total_tokens as f32
            },
            average_log_likelihood: total_log_likelihood / samples.len() as f32,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_training_samples(
        &self,
        data_path: &Path,
        max_samples: Option<usize>,
    ) -> Result<Vec<TrainingSample>, LacError> {
        let bytes = read_file(data_path)?;
        let content = std::str::from_utf8(&bytes).map_err(|error| {
            LacError::Parse(format!("`{}` is not utf8: {error}", data_path.display()))
        })?;

        let limit = max_samples.unwrap_or(usize::MAX);
        let mut samples = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(sample) = self.parse_training_line(line) else {
                continue;
            };
            samples.push(sample);
            if samples.len() >= limit {
                break;
            }
        }

        Ok(samples)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn parse_training_line(&self, line: &str) -> Option<TrainingSample> {
        match self.mode {
            Mode::Lac => self.parse_lac_training_line(line),
            Mode::Seg => self.parse_seg_training_line(line),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn parse_lac_training_line(&self, line: &str) -> Option<TrainingSample> {
        let fallback = self.fallback_label_id();
        let mut token_ids = Vec::new();
        let mut gold_tag_ids = Vec::new();

        for item in line.split_whitespace() {
            let (word, tag) = item.rsplit_once('/')?;
            if word.is_empty() || tag.is_empty() {
                return None;
            }

            let begin_tag = format!("{tag}-B");
            let inside_tag = format!("{tag}-I");
            let begin_id = self.label_id_or_fallback(&begin_tag, fallback);
            let inside_id = self.label_id_or_fallback(&inside_tag, begin_id);

            for (index, ch) in word.chars().enumerate() {
                let ch_text = ch.to_string();
                token_ids.push(self.token_id_for_word(&ch_text));
                if index == 0 {
                    gold_tag_ids.push(begin_id);
                } else {
                    gold_tag_ids.push(inside_id);
                }
            }
        }

        if token_ids.is_empty() || token_ids.len() != gold_tag_ids.len() {
            return None;
        }

        Some(TrainingSample {
            token_ids,
            gold_tag_ids,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn parse_seg_training_line(&self, line: &str) -> Option<TrainingSample> {
        let fallback = self.fallback_label_id();
        let begin_id = self.label_id_or_fallback("-B", fallback);
        let inside_id = self.label_id_or_fallback("-I", begin_id);
        let end_id = self.label_id_or_fallback("-E", inside_id);
        let single_id = self.label_id_or_fallback("-S", end_id);

        let mut token_ids = Vec::new();
        let mut gold_tag_ids = Vec::new();

        for item in line.split_whitespace() {
            let word = item.rsplit_once('/').map(|(word, _)| word).unwrap_or(item);
            if word.is_empty() {
                continue;
            }

            let chars: Vec<char> = word.chars().collect();
            if chars.is_empty() {
                continue;
            }

            match chars.len() {
                1 => {
                    let ch_text = chars[0].to_string();
                    token_ids.push(self.token_id_for_word(&ch_text));
                    gold_tag_ids.push(single_id);
                }
                2 => {
                    let first = chars[0].to_string();
                    let second = chars[1].to_string();
                    token_ids.push(self.token_id_for_word(&first));
                    gold_tag_ids.push(begin_id);
                    token_ids.push(self.token_id_for_word(&second));
                    gold_tag_ids.push(end_id);
                }
                _ => {
                    let last_index = chars.len() - 1;
                    for (index, ch) in chars.into_iter().enumerate() {
                        let ch_text = ch.to_string();
                        token_ids.push(self.token_id_for_word(&ch_text));
                        if index == 0 {
                            gold_tag_ids.push(begin_id);
                        } else if index == last_index {
                            gold_tag_ids.push(end_id);
                        } else {
                            gold_tag_ids.push(inside_id);
                        }
                    }
                }
            }
        }

        if token_ids.is_empty() || token_ids.len() != gold_tag_ids.len() {
            return None;
        }

        Some(TrainingSample {
            token_ids,
            gold_tag_ids,
        })
    }

    fn parse_targets(&self, words: &[String], tag_ids: &[usize]) -> Vec<OutputItem> {
        let mut result = Vec::new();
        for (index, tag_id) in tag_ids.iter().copied().enumerate() {
            let tag = self.id2label.get(tag_id).map(String::as_str).unwrap_or("O");
            let is_new = result.is_empty() || tag.ends_with('B') || tag.ends_with('S');
            if is_new {
                result.push(OutputItem {
                    word: words[index].clone(),
                    tag: strip_tag_suffix(tag).to_string(),
                });
            } else if let Some(last) = result.last_mut() {
                last.word.push_str(&words[index]);
            }
        }
        result
    }
}

fn strip_tag_suffix(tag: &str) -> &str {
    if tag.ends_with("-B") || tag.ends_with("-I") || tag.ends_with("-E") || tag.ends_with("-S") {
        &tag[..tag.len() - 2]
    } else {
        tag
    }
}

fn parse_ini(input: &[u8]) -> Result<HashMap<String, String>, LacError> {
    let content = std::str::from_utf8(input)
        .map_err(|error| LacError::Parse(format!("`args.ini` is not utf8: {error}")))?;
    let mut values = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with(';')
            || line.starts_with('[')
        {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(values)
}

fn parse_ini_usize(values: &HashMap<String, String>, key: &str) -> Result<usize, LacError> {
    values
        .get(key)
        .ok_or_else(|| LacError::Parse(format!("missing key `{key}` in args.ini")))?
        .parse::<usize>()
        .map_err(|error| LacError::Parse(format!("invalid `{key}` in args.ini: {error}")))
}

fn parse_mode_value(raw: &str) -> Option<Mode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "lac" | "tag" | "lexer" => Some(Mode::Lac),
        "seg" | "cutter" => Some(Mode::Seg),
        _ => None,
    }
}

fn infer_mode(parsed_args: &HashMap<String, String>, label_map: &HashMap<i64, String>) -> Mode {
    if let Some(mode) = parsed_args
        .get("model")
        .and_then(|value| parse_mode_value(value))
    {
        return mode;
    }
    if let Some(mode) = parsed_args
        .get("tag_type")
        .and_then(|value| parse_mode_value(value))
    {
        return mode;
    }

    let is_seg_label = |label: &str| matches!(label, "-B" | "-I" | "-E" | "-S");
    let has_only_seg_labels = label_map.values().all(|label| is_seg_label(label.as_str()));
    if has_only_seg_labels {
        Mode::Seg
    } else {
        Mode::Lac
    }
}

fn load_word2id(input: &[u8]) -> Result<HashMap<String, i64>, LacError> {
    let content = std::str::from_utf8(input)
        .map_err(|error| LacError::Parse(format!("`word.dic` is not utf8: {error}")))?;
    let mut output = HashMap::new();
    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some((id, word)) = line.split_once('\t') else {
            continue;
        };
        let id = id
            .parse::<i64>()
            .map_err(|error| LacError::Parse(format!("invalid word id `{id}`: {error}")))?;
        output.insert(word.to_string(), id);
    }
    Ok(output)
}

fn load_q2b(input: &[u8]) -> Result<HashMap<String, String>, LacError> {
    let content = std::str::from_utf8(input)
        .map_err(|error| LacError::Parse(format!("`q2b.dic` is not utf8: {error}")))?;
    let mut output = HashMap::new();
    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some((from, to)) = line.split_once('\t') else {
            continue;
        };
        output.insert(from.to_string(), to.to_string());
    }
    Ok(output)
}

fn load_id2label(input: &[u8]) -> Result<HashMap<i64, String>, LacError> {
    let content = std::str::from_utf8(input)
        .map_err(|error| LacError::Parse(format!("`tag.dic` is not utf8: {error}")))?;
    let mut output = HashMap::new();
    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some((id, tag)) = line.split_once('\t') else {
            continue;
        };
        let id = id
            .parse::<i64>()
            .map_err(|error| LacError::Parse(format!("invalid tag id `{id}`: {error}")))?;
        output.insert(id, tag.to_string());
    }
    Ok(output)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_label2id(input: &[u8]) -> Result<HashMap<String, usize>, LacError> {
    let content = std::str::from_utf8(input)
        .map_err(|error| LacError::Parse(format!("`tag.dic` is not utf8: {error}")))?;
    let mut output = HashMap::new();
    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        let Some((id, tag)) = line.split_once('\t') else {
            continue;
        };
        let id = id
            .parse::<usize>()
            .map_err(|error| LacError::Parse(format!("invalid tag id `{id}`: {error}")))?;
        output.insert(tag.to_string(), id);
    }
    Ok(output)
}

fn read_checked_param<F>(
    name: &str,
    expected: usize,
    read_param: &mut F,
) -> Result<Vec<f32>, LacError>
where
    F: FnMut(&str) -> Result<Vec<u8>, LacError>,
{
    let bytes = read_param(name)?;
    let values = parse_paddle_tensor(&bytes, name)?;
    if values.len() != expected {
        return Err(LacError::InvalidShape {
            name: name.to_string(),
            expected,
            got: values.len(),
        });
    }
    Ok(values)
}

fn parse_paddle_tensor(bytes: &[u8], name: &str) -> Result<Vec<f32>, LacError> {
    if bytes.len() < 20 {
        return Err(LacError::Parse(format!(
            "parameter `{name}` is too small to be a Paddle tensor"
        )));
    }
    let header_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let payload_start = 20 + header_len;
    if payload_start > bytes.len() {
        return Err(LacError::Parse(format!(
            "parameter `{name}` has invalid header length {header_len}"
        )));
    }
    let payload = &bytes[payload_start..];
    if payload.len() % 4 != 0 {
        return Err(LacError::Parse(format!(
            "parameter `{name}` payload size {} is not float32 aligned",
            payload.len()
        )));
    }
    let mut values = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn matmul_add_bias(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0f32; rows * output_dim];
    for row in 0..rows {
        let input_row = row * input_dim;
        let output_row = row * output_dim;
        for out_index in 0..output_dim {
            let mut value = bias[out_index];
            for in_index in 0..input_dim {
                value += input[input_row + in_index] * weight[in_index * output_dim + out_index];
            }
            output[output_row + out_index] = value;
        }
    }
    output
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn dynamic_gru(
    pre_act: &[f32],
    timesteps: usize,
    hidden_size: usize,
    weight: &[f32],
    bias: &[f32],
    reverse: bool,
) -> Vec<f32> {
    let mut output = vec![0f32; timesteps * hidden_size];
    let mut previous = vec![0f32; hidden_size];
    let mut previous_state = vec![0f32; hidden_size];
    let mut current_state = vec![0f32; hidden_size];
    let mut update_gate = vec![0f32; hidden_size];
    let mut reset_gate = vec![0f32; hidden_size];

    let mut step = |timestep: usize, previous: &mut [f32]| {
        let step_offset = timestep * hidden_size * 3;
        let gate_block_offset = hidden_size * hidden_size * 2;
        previous_state.copy_from_slice(previous);

        for hidden_index in 0..hidden_size {
            let mut gate_u = pre_act[step_offset + hidden_index] + bias[hidden_index];
            let mut gate_r = pre_act[step_offset + hidden_size + hidden_index]
                + bias[hidden_size + hidden_index];
            for previous_index in 0..hidden_size {
                let matrix_offset = previous_index * hidden_size * 2;
                let previous_value = previous_state[previous_index];
                gate_u += previous_value * weight[matrix_offset + hidden_index];
                gate_r += previous_value * weight[matrix_offset + hidden_size + hidden_index];
            }
            update_gate[hidden_index] = sigmoid(gate_u);
            reset_gate[hidden_index] = sigmoid(gate_r);
        }

        for hidden_index in 0..hidden_size {
            let mut candidate = pre_act[step_offset + hidden_size * 2 + hidden_index]
                + bias[hidden_size * 2 + hidden_index];
            for previous_index in 0..hidden_size {
                let matrix_offset = gate_block_offset + previous_index * hidden_size;
                candidate += reset_gate[previous_index]
                    * previous_state[previous_index]
                    * weight[matrix_offset + hidden_index];
            }
            let candidate = candidate.tanh();
            current_state[hidden_index] = update_gate[hidden_index] * candidate
                + (1.0 - update_gate[hidden_index]) * previous_state[hidden_index];
            output[timestep * hidden_size + hidden_index] = current_state[hidden_index];
        }
        previous.copy_from_slice(&current_state);
    };

    if reverse {
        for timestep in (0..timesteps).rev() {
            step(timestep, &mut previous);
        }
    } else {
        for timestep in 0..timesteps {
            step(timestep, &mut previous);
        }
    }

    output
}

fn viterbi_decode(
    emission: &[f32],
    timesteps: usize,
    num_labels: usize,
    crf_w: &[f32],
) -> Vec<usize> {
    if timesteps == 0 {
        return Vec::new();
    }

    let mut previous_scores = vec![0f32; num_labels];
    for label in 0..num_labels {
        previous_scores[label] = crf_w[label] + emission[label];
    }

    let mut backpointers = vec![0usize; timesteps * num_labels];
    for step in 1..timesteps {
        let mut current_scores = vec![f32::NEG_INFINITY; num_labels];
        for current_label in 0..num_labels {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_label = 0usize;
            for previous_label in 0..num_labels {
                let transition = crf_w[(previous_label + 2) * num_labels + current_label];
                let score = previous_scores[previous_label] + transition;
                if score > best_score {
                    best_score = score;
                    best_label = previous_label;
                }
            }
            current_scores[current_label] =
                best_score + emission[step * num_labels + current_label];
            backpointers[step * num_labels + current_label] = best_label;
        }
        previous_scores = current_scores;
    }

    let mut best_last_label = 0usize;
    let mut best_last_score = f32::NEG_INFINITY;
    for label in 0..num_labels {
        let score = previous_scores[label] + crf_w[num_labels + label];
        if score > best_last_score {
            best_last_score = score;
            best_last_label = label;
        }
    }

    let mut path = vec![0usize; timesteps];
    path[timesteps - 1] = best_last_label;
    for step in (1..timesteps).rev() {
        let current = path[step];
        path[step - 1] = backpointers[step * num_labels + current];
    }
    path
}

#[cfg(not(target_arch = "wasm32"))]
fn log_sum_exp(values: &[f32]) -> f32 {
    let mut max_value = f32::NEG_INFINITY;
    for value in values {
        if *value > max_value {
            max_value = *value;
        }
    }
    if !max_value.is_finite() {
        return max_value;
    }
    let mut sum = 0f32;
    for value in values {
        sum += (*value - max_value).exp();
    }
    max_value + sum.ln()
}

#[cfg(not(target_arch = "wasm32"))]
fn crf_sequence_score(emission: &[f32], gold: &[usize], num_labels: usize, crf_w: &[f32]) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }

    let mut score = crf_w[gold[0]] + emission[gold[0]];
    for step in 1..gold.len() {
        let previous = gold[step - 1];
        let current = gold[step];
        score += crf_w[(previous + 2) * num_labels + current];
        score += emission[step * num_labels + current];
    }
    score + crf_w[num_labels + gold[gold.len() - 1]]
}

#[cfg(not(target_arch = "wasm32"))]
fn crf_forward(
    emission: &[f32],
    timesteps: usize,
    num_labels: usize,
    crf_w: &[f32],
) -> (Vec<f32>, f32) {
    let mut alpha = vec![f32::NEG_INFINITY; timesteps * num_labels];
    for label in 0..num_labels {
        alpha[label] = crf_w[label] + emission[label];
    }

    let mut score_buffer = vec![0f32; num_labels];
    for step in 1..timesteps {
        let previous_offset = (step - 1) * num_labels;
        let current_offset = step * num_labels;
        for current_label in 0..num_labels {
            for previous_label in 0..num_labels {
                score_buffer[previous_label] = alpha[previous_offset + previous_label]
                    + crf_w[(previous_label + 2) * num_labels + current_label];
            }
            alpha[current_offset + current_label] =
                emission[current_offset + current_label] + log_sum_exp(&score_buffer);
        }
    }

    let last_offset = (timesteps - 1) * num_labels;
    for label in 0..num_labels {
        score_buffer[label] = alpha[last_offset + label] + crf_w[num_labels + label];
    }
    let log_z = log_sum_exp(&score_buffer);
    (alpha, log_z)
}

#[cfg(not(target_arch = "wasm32"))]
fn crf_backward(emission: &[f32], timesteps: usize, num_labels: usize, crf_w: &[f32]) -> Vec<f32> {
    let mut beta = vec![f32::NEG_INFINITY; timesteps * num_labels];
    let last_offset = (timesteps - 1) * num_labels;
    for label in 0..num_labels {
        beta[last_offset + label] = crf_w[num_labels + label];
    }

    let mut score_buffer = vec![0f32; num_labels];
    for step in (0..timesteps - 1).rev() {
        let current_offset = step * num_labels;
        let next_offset = (step + 1) * num_labels;
        for current_label in 0..num_labels {
            for next_label in 0..num_labels {
                score_buffer[next_label] = crf_w[(current_label + 2) * num_labels + next_label]
                    + emission[next_offset + next_label]
                    + beta[next_offset + next_label];
            }
            beta[current_offset + current_label] = log_sum_exp(&score_buffer);
        }
    }

    beta
}

#[cfg(not(target_arch = "wasm32"))]
fn crf_log_likelihood(emission: &[f32], gold: &[usize], num_labels: usize, crf_w: &[f32]) -> f32 {
    if gold.is_empty() {
        return 0.0;
    }
    let (_, log_z) = crf_forward(emission, gold.len(), num_labels, crf_w);
    crf_sequence_score(emission, gold, num_labels, crf_w) - log_z
}

#[cfg(not(target_arch = "wasm32"))]
fn crf_log_likelihood_and_gradient(
    emission: &[f32],
    gold: &[usize],
    num_labels: usize,
    crf_w: &[f32],
    gradient: &mut [f32],
) -> f32 {
    gradient.fill(0.0);
    if gold.is_empty() {
        return 0.0;
    }

    let timesteps = gold.len();
    let (alpha, log_z) = crf_forward(emission, timesteps, num_labels, crf_w);
    let beta = crf_backward(emission, timesteps, num_labels, crf_w);
    let log_likelihood = crf_sequence_score(emission, gold, num_labels, crf_w) - log_z;

    gradient[gold[0]] += 1.0;
    gradient[num_labels + gold[timesteps - 1]] += 1.0;
    for step in 1..timesteps {
        let previous = gold[step - 1];
        let current = gold[step];
        gradient[(previous + 2) * num_labels + current] += 1.0;
    }

    for label in 0..num_labels {
        let log_probability = alpha[label] + beta[label] - log_z;
        gradient[label] -= log_probability.exp();
    }

    let last_offset = (timesteps - 1) * num_labels;
    for label in 0..num_labels {
        let log_probability = alpha[last_offset + label] + beta[last_offset + label] - log_z;
        gradient[num_labels + label] -= log_probability.exp();
    }

    for step in 1..timesteps {
        let previous_offset = (step - 1) * num_labels;
        let current_offset = step * num_labels;
        for previous_label in 0..num_labels {
            let previous_alpha = alpha[previous_offset + previous_label];
            for current_label in 0..num_labels {
                let log_probability = previous_alpha
                    + crf_w[(previous_label + 2) * num_labels + current_label]
                    + emission[current_offset + current_label]
                    + beta[current_offset + current_label]
                    - log_z;
                gradient[(previous_label + 2) * num_labels + current_label] -=
                    log_probability.exp();
            }
        }
    }

    log_likelihood
}

#[cfg(not(target_arch = "wasm32"))]
fn read_file(path: impl AsRef<Path>) -> Result<Vec<u8>, LacError> {
    let path = path.as_ref();
    fs::read(path).map_err(|source| LacError::Io {
        path: path.display().to_string(),
        source,
    })
}

#[cfg(all(test, feature = "bundled-models"))]
mod tests {
    use super::{infer_mode, Lac, Mode, OutputItem};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn lac_mode_matches_reference_example() {
        let lac = Lac::new(Mode::Lac).expect("load bundled lac model");
        let result = lac.run("百度是一家高科技公司");
        assert_eq!(
            result,
            vec![
                OutputItem {
                    word: "百度".to_string(),
                    tag: "ORG".to_string(),
                },
                OutputItem {
                    word: "是".to_string(),
                    tag: "v".to_string(),
                },
                OutputItem {
                    word: "一家".to_string(),
                    tag: "m".to_string(),
                },
                OutputItem {
                    word: "高科技".to_string(),
                    tag: "n".to_string(),
                },
                OutputItem {
                    word: "公司".to_string(),
                    tag: "n".to_string(),
                },
            ]
        );
    }

    #[test]
    fn seg_mode_matches_reference_example() {
        let seg = Lac::new(Mode::Seg).expect("load bundled seg model");
        let words = seg.cut("百度是一家高科技公司");
        assert_eq!(words, vec!["百度", "是", "一家", "高科技", "公司"]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn model_dir(path: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn from_dir_infers_mode_from_model_key() {
        let lac = Lac::from_dir(model_dir("models_general/lac_model")).expect("load lac model");
        assert_eq!(lac.mode(), Mode::Lac);

        let seg = Lac::from_dir(model_dir("models_general/seg_model")).expect("load seg model");
        assert_eq!(seg.mode(), Mode::Seg);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn from_dir_with_mode_overrides_detected_mode() {
        let forced = Lac::from_dir_with_mode(model_dir("models_general/seg_model"), Mode::Lac)
            .expect("load seg model with forced mode");
        assert_eq!(forced.mode(), Mode::Lac);
    }

    #[test]
    fn infer_mode_fallbacks_work() {
        let mut labels = HashMap::new();
        labels.insert(0, "n-B".to_string());
        labels.insert(1, "n-I".to_string());
        labels.insert(2, "O".to_string());

        let mut args = HashMap::new();
        args.insert("model".to_string(), "seg".to_string());
        args.insert("tag_type".to_string(), "tag".to_string());
        assert_eq!(infer_mode(&args, &labels), Mode::Seg);

        args.remove("model");
        assert_eq!(infer_mode(&args, &labels), Mode::Lac);

        let mut seg_labels = HashMap::new();
        seg_labels.insert(0, "-B".to_string());
        seg_labels.insert(1, "-I".to_string());
        seg_labels.insert(2, "-E".to_string());
        seg_labels.insert(3, "-S".to_string());
        args.remove("tag_type");
        assert_eq!(infer_mode(&args, &seg_labels), Mode::Seg);
    }
}
