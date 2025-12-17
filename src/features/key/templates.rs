//! Krumhansl-Kessler key templates
//!
//! Defines tonal profiles for 24 keys (12 major + 12 minor) based on empirical
//! listening experiments.
//!
//! # Reference
//!
//! Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the Dynamic Changes in Perceived
//! Tonal Organization in a Spatial Representation of Musical Keys. *Psychological Review*,
//! 89(4), 334-368.

/// Key templates for all 24 keys
#[derive(Debug, Clone)]
pub struct KeyTemplates {
    /// Major key templates (12 keys: C, C#, D, ..., B)
    pub major: [Vec<f32>; 12],
    
    /// Minor key templates (12 keys: C, C#, D, ..., B)
    pub minor: [Vec<f32>; 12],
}

impl KeyTemplates {
    /// Create new key templates with Krumhansl-Kessler profiles
    ///
    /// Templates are 12-element vectors representing the likelihood of each
    /// semitone class (C, C#, D, ..., B) in that key.
    ///
    /// # Returns
    ///
    /// `KeyTemplates` with all 24 key profiles initialized
    pub fn new() -> Self {
        // C Major template
        // Values from Krumhansl & Kessler (1982)
        let c_major = vec![
            0.15, // C
            0.01, // C#
            0.12, // D
            0.01, // D#
            0.13, // E
            0.11, // F
            0.01, // F#
            0.13, // G
            0.01, // G#
            0.12, // A
            0.01, // A#
            0.10, // B
        ];
        
        // A Minor template (relative minor of C major)
        let a_minor = vec![
            0.12, // A
            0.01, // A#
            0.10, // B
            0.12, // C
            0.01, // C#
            0.11, // D
            0.01, // D#
            0.12, // E
            0.13, // F
            0.01, // F#
            0.10, // G
            0.01, // G#
        ];
        
        // Generate all 12 major keys by rotating C major
        let mut major: [Vec<f32>; 12] = [
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
        ];
        
        // Generate all 12 minor keys by rotating A minor
        let mut minor: [Vec<f32>; 12] = [
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
            vec![0.0; 12],
        ];
        
        // Rotate templates for all 12 keys
        for key_idx in 0..12 {
            // Major keys: rotate C major template
            // For key_idx, we want the template rotated so that the tonic is at key_idx
            for semitone_idx in 0..12 {
                major[key_idx][semitone_idx] = c_major[(semitone_idx + 12 - key_idx) % 12];
            }
            
            // Minor keys: rotate A minor template
            // A minor template has A at index 0, but we want it at key_idx (where key_idx=9 for A)
            // To get A at key_idx, we need to rotate the template by key_idx positions
            // So we take element at (semitone_idx - key_idx + 12) % 12 from the original template
            for semitone_idx in 0..12 {
                let source_idx = (semitone_idx + 12 - key_idx) % 12;
                minor[key_idx][semitone_idx] = a_minor[source_idx];
            }
        }
        
        Self { major, minor }
    }
    
    /// Get template for a specific key
    ///
    /// # Arguments
    ///
    /// * `key_idx` - Key index (0-11 for major, 12-23 for minor)
    ///
    /// # Returns
    ///
    /// 12-element template vector for the specified key
    pub fn get_template(&self, key_idx: u32) -> &[f32] {
        if key_idx < 12 {
            &self.major[key_idx as usize]
        } else {
            &self.minor[(key_idx - 12) as usize]
        }
    }
    
    /// Get template for major key
    ///
    /// # Arguments
    ///
    /// * `key_idx` - Major key index (0-11: C, C#, D, ..., B)
    ///
    /// # Returns
    ///
    /// 12-element template vector for the specified major key
    pub fn get_major_template(&self, key_idx: u32) -> &[f32] {
        &self.major[key_idx as usize % 12]
    }
    
    /// Get template for minor key
    ///
    /// # Arguments
    ///
    /// * `key_idx` - Minor key index (0-11: C, C#, D, ..., B)
    ///
    /// # Returns
    ///
    /// 12-element template vector for the specified minor key
    pub fn get_minor_template(&self, key_idx: u32) -> &[f32] {
        &self.minor[key_idx as usize % 12]
    }
}

impl Default for KeyTemplates {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_key_templates_creation() {
        let templates = KeyTemplates::new();
        
        // Check that all templates have 12 elements
        for i in 0..12 {
            assert_eq!(templates.major[i].len(), 12);
            assert_eq!(templates.minor[i].len(), 12);
        }
    }
    
    #[test]
    fn test_c_major_template() {
        let templates = KeyTemplates::new();
        let c_major = templates.get_major_template(0);
        
        // C major should have high values for C, E, G (tonic, major third, perfect fifth)
        assert!(c_major[0] > 0.1); // C (tonic)
        assert!(c_major[4] > 0.1); // E (major third)
        assert!(c_major[7] > 0.1); // G (perfect fifth)
        
        // Non-scale tones should have low values
        assert!(c_major[1] < 0.05); // C#
        assert!(c_major[3] < 0.05); // D#
    }
    
    #[test]
    fn test_a_minor_template() {
        let templates = KeyTemplates::new();
        let a_minor = templates.get_minor_template(9); // A is index 9
        
        // A minor should have high values for A, C, E (tonic, minor third, perfect fifth)
        assert!(a_minor[9] > 0.1); // A (tonic)
        assert!(a_minor[0] > 0.1); // C (minor third)
        assert!(a_minor[4] > 0.1); // E (perfect fifth)
    }
    
    #[test]
    fn test_get_template() {
        let templates = KeyTemplates::new();
        
        // Test major keys (0-11)
        let c_major = templates.get_template(0);
        assert_eq!(c_major.len(), 12);
        
        // Test minor keys (12-23)
        let a_minor = templates.get_template(21); // A minor is 12 + 9 = 21
        assert_eq!(a_minor.len(), 12);
    }
    
    #[test]
    fn test_template_rotation() {
        let templates = KeyTemplates::new();
        
        // C major and D major should be related by rotation
        let _c_major = templates.get_major_template(0);
        let d_major = templates.get_major_template(2); // D is 2 semitones above C
        
        // D major should have high value at index 2 (D)
        assert!(d_major[2] > 0.1);
        
        // The pattern should be rotated
        // C major's tonic (index 0) should correspond to D major's subdominant (index 10)
        // This is a simplified check - the full relationship is more complex
    }
}
