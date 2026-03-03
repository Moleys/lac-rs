use crate::{Lac, Mode};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmLac {
    inner: Lac,
}

#[wasm_bindgen]
impl WasmLac {
    #[wasm_bindgen(constructor)]
    pub fn new(mode: Option<String>) -> Result<WasmLac, JsValue> {
        let selected_mode = match mode.as_deref() {
            Some("seg") | Some("SEG") => Mode::Seg,
            _ => Mode::Lac,
        };
        let inner =
            Lac::new(selected_mode).map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run_js(&self, text: &str) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.run(text))
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen(js_name = cut)]
    pub fn cut_js(&self, text: &str) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.cut(text))
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }
}
