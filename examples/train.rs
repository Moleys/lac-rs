use lac_rs::{Lac, TrainOptions};
use std::path::PathBuf;

fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> T {
    args.get(index)
        .and_then(|raw| raw.parse::<T>().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root_dir.join("models_general/lac_model"));
    let train_data = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| root_dir.join("data/lac_train_chivi.tsv"));
    let eval_data = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| root_dir.join("data/lac_test_chivi.tsv"));

    let options = TrainOptions {
        epochs: parse_arg(&args, 4, 1usize),
        learning_rate: parse_arg(&args, 5, 0.02f32),
        max_samples: args.get(6).and_then(|raw| raw.parse::<usize>().ok()),
    };

    let mut model = Lac::from_dir(&model_dir).expect("load model");
    let train_report = model
        .train_from_tsv(&train_data, options)
        .expect("train from dataset");
    println!(
        "train  samples={} tokens={} acc={:.4} avg_ll={:.4}",
        train_report.samples,
        train_report.tokens,
        train_report.token_accuracy,
        train_report.average_log_likelihood
    );

    let eval_report = model
        .evaluate_tsv(&eval_data)
        .expect("evaluate from dataset");
    println!(
        "eval   samples={} tokens={} acc={:.4} avg_ll={:.4}",
        eval_report.samples,
        eval_report.tokens,
        eval_report.token_accuracy,
        eval_report.average_log_likelihood
    );
}
