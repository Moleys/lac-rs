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
    oov_id: i64,
    config: ModelConfig,
    vocab_size: usize,
    num_labels: usize,
    params: ModelParams,
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

        let mut token_ids = Vec::with_capacity(seq_words.len());
        for word in &seq_words {
            let normalized = self
                .q2b
                .get(word)
                .map(String::as_str)
                .unwrap_or(word.as_str());
            let token_id =
                self.word2id
                    .get(normalized)
                    .copied()
                    .unwrap_or(self.oov_id)
                    .clamp(0, (self.vocab_size.saturating_sub(1)) as i64) as usize;
            token_ids.push(token_id);
        }

        let tag_ids = self.predict_tags(&token_ids);
        self.parse_targets(&seq_words, &tag_ids)
    }

    fn predict_tags(&self, token_ids: &[usize]) -> Vec<usize> {
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

        let emission = matmul_add_bias(
            &features,
            steps,
            hidden * 2,
            self.num_labels,
            &self.params.emission_w,
            &self.params.emission_b,
        );

        viterbi_decode(&emission, steps, self.num_labels, &self.params.crf_w)
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
