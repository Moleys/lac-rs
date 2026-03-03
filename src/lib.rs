mod runtime;

#[cfg(feature = "bundled-models")]
mod bundled;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod wasm_api;

pub use runtime::{Lac, LacError, Mode, OutputItem, TrainOptions, TrainReport};

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub use wasm_api::WasmLac;
