//! Sample windowing and buffering utilities

/// Buffer for windowed audio processing
#[derive(Debug)]
pub struct SampleBuffer {
    /// Buffer data
    data: Vec<f32>,
    /// Current position
    position: usize,
}

impl SampleBuffer {
    /// Create a new sample buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            position: 0,
        }
    }
    
    /// Add samples to buffer
    pub fn push(&mut self, samples: &[f32]) {
        self.data.extend_from_slice(samples);
    }
    
    /// Get next window of samples
    pub fn next_window(&mut self, window_size: usize) -> Option<Vec<f32>> {
        if self.position + window_size > self.data.len() {
            return None;
        }
        
        let window = self.data[self.position..self.position + window_size].to_vec();
        self.position += window_size;
        Some(window)
    }
}

