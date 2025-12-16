//! ML refinement modules (Phase 2)
//!
//! Optional ONNX model inference for edge case correction.

#[cfg(feature = "ml")]
pub mod onnx_model;

#[cfg(feature = "ml")]
pub mod refinement;

#[cfg(feature = "ml")]
pub mod edge_cases;

