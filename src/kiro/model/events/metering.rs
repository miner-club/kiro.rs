//! Kiro per-request credit metering event.

use serde::Deserialize;

use crate::kiro::parser::error::ParseResult;
use crate::kiro::parser::frame::Frame;

use super::base::EventPayload;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeteringEvent {
    /// Credits consumed by this request.
    #[serde(default)]
    pub usage: Option<f64>,
    #[serde(default)]
    pub input_tokens: Option<i32>,
    #[serde(default)]
    pub output_tokens: Option<i32>,
}

impl EventPayload for MeteringEvent {
    fn from_frame(frame: &Frame) -> ParseResult<Self> {
        frame.payload_as_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_metering_credits() {
        let event: MeteringEvent =
            serde_json::from_str(r#"{"usage":1.25,"inputTokens":100,"outputTokens":20}"#).unwrap();
        assert_eq!(event.usage, Some(1.25));
        assert_eq!(event.input_tokens, Some(100));
        assert_eq!(event.output_tokens, Some(20));
    }
}
