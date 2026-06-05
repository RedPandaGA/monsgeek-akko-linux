// Audio Reactive LED Mode
// Captures system audio and maps frequency spectrum to keyboard RGB colors

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use spectrum_analyzer::scaling::divide_by_N_sqrt;
use spectrum_analyzer::windows::hann_window;
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::led_stream::send_full_frame;
use crate::notify::keymap::MATRIX_LEN;

/// Number of frequency bands to analyze
const NUM_BANDS: usize = 8;

/// FFT sample size (must be power of 2)
const FFT_SIZE: usize = 2048;

/// Matrix dimensions
const COLS: usize = 16;
const ROWS: usize = 6;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Per-band sensitivity multipliers
#[derive(Clone, Debug)]
pub struct BandSensitivity {
    pub values: [f32; NUM_BANDS],
}

impl Default for BandSensitivity {
    fn default() -> Self {
        Self {
            values: [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 2.2, 3.0],
        }
    }
}

/// Per-band smoothing factors
#[derive(Clone, Debug)]
pub struct BandSmoothing {
    pub values: [f32; NUM_BANDS],
}

impl Default for BandSmoothing {
    fn default() -> Self {
        Self {
            // Bass: fast (low smoothing), treble: slower (higher smoothing)
            values: [0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3],
        }
    }
}

/// A named frequency band set (low, high, weight) for each of the 8 bands
#[derive(Clone, Debug)]
pub struct FrequencySet {
    pub name: String,
    /// (low_hz, high_hz, weight) for each band
    pub bands: [(f32, f32, f32); NUM_BANDS],
}

impl FrequencySet {
    pub fn default_set() -> Self {
        Self {
            name: "default".to_string(),
            bands: [
                (20.0,    100.0,   2.5),
                (60.0,    150.0,   1.5),
                (150.0,   400.0,   1.2),
                (400.0,   1000.0,  1.0),
                (1000.0,  2500.0,  1.0),
                (2500.0,  6000.0,  1.2),
                (6000.0,  12000.0, 1.5),
                (12000.0, 20000.0, 2.0),
            ],
        }
    }

    pub fn drums_set() -> Self {
        Self {
            name: "drums".to_string(),
            bands: [
                // Kick drum body: 50-100Hz
                (50.0,   100.0,  3.0),
                // Kick attack + bass guitar: 100-200Hz
                (100.0,  200.0,  2.0),
                // Snare body + low toms: 200-400Hz
                (200.0,  400.0,  1.5),
                // Snare crack + upper toms: 400-800Hz
                (400.0,  800.0,  1.5),
                // Hi-hat closed + rimshot: 800-3000Hz
                (800.0,  3000.0, 1.2),
                // Hi-hat open + cymbals body: 3000-7000Hz
                (3000.0, 7000.0, 1.3),
                // Cymbal shimmer: 7000-12000Hz
                (7000.0, 12000.0, 1.8),
                // Cymbal air + room: 12000-20000Hz
                (12000.0, 20000.0, 2.5),
            ],
        }
    }
}

/// A named palette of colors for dazzle mode
#[derive(Clone, Debug)]
pub struct DazzlePalette {
    pub name: String,
    pub colors: Vec<(u8, u8, u8)>,
}

impl DazzlePalette {
    pub fn pick(&self, idx: usize) -> (u8, u8, u8) {
        self.colors[idx % self.colors.len()]
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }
}

/// Full audio reactive configuration — loaded from ~/.config/monsgeek-akko/audio.toml
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// Audio source device name
    pub source: Option<String>,
    /// Color mode: "spectrum", "dazzle", "dazzleband", "solid", "gradient"
    pub color_mode: String,
    /// Active dazzle palette name
    pub dazzle_palette: String,
    /// Available dazzle palettes
    pub palettes: Vec<DazzlePalette>,
    /// Base hue for solid/gradient modes (0-360)
    pub base_hue: f32,
    /// Global sensitivity multiplier
    pub sensitivity: f32,
    /// Smoothing factor (kept for config compat, per-band smoothing takes precedence)
    pub smoothing: f32,
    /// Bass amplification of other bands (0.0 = off, 1.0 = strong)
    pub bass_amplify: f32,
    /// Target FPS
    pub fps: u32,
    /// Per-band sensitivity
    pub band_sensitivity: BandSensitivity,
    /// Per-band smoothing
    pub band_smoothing: BandSmoothing,
    /// Active frequency set name
    pub freq_set: String,
    /// Named frequency sets
    pub freq_sets: Vec<FrequencySet>,
    /// Active layout name
    pub layout: String,
    /// Named column layouts
    pub layouts: Vec<(String, [usize; COLS])>,
    pub global_mix: f32,
    pub local_mix: f32,
    pub global_decay: f32,
    pub band_decay: f32,
    pub response_curve: f32,
    /// How many frames each band holds its color in dazzleband mode
    pub dazzle_band_frames: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            source: None,
            color_mode: "spectrum".to_string(),
            dazzle_palette: "rainbow".to_string(),
            palettes: vec![
                DazzlePalette { name: "rainbow".to_string(),      colors: vec![(255,0,0),(255,127,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(139,0,255),(255,0,255)] },
                DazzlePalette { name: "reddish".to_string(),      colors: vec![(255,0,0),(220,20,20),(255,80,0),(200,0,50),(255,30,30)] },
                DazzlePalette { name: "bluish".to_string(),       colors: vec![(0,0,255),(0,100,255),(0,200,255),(0,255,255),(30,30,220)] },
                DazzlePalette { name: "purplish".to_string(),     colors: vec![(148,0,211),(180,0,255),(100,0,200),(220,50,255),(75,0,130)] },
                DazzlePalette { name: "fire".to_string(),         colors: vec![(255,0,0),(255,60,0),(255,120,0),(255,200,0),(255,255,0)] },
                DazzlePalette { name: "redwhiteblue".to_string(), colors: vec![(255,0,0),(255,255,255),(0,0,255),(200,0,0),(255,255,255),(0,0,200)] },
                DazzlePalette { name: "ember".to_string(),        colors: vec![(180,10,0),(220,40,0),(255,80,0),(255,140,0),(255,200,20)] },
                DazzlePalette { name: "sunset".to_string(),       colors: vec![(255,80,50),(255,30,80),(255,120,0),(255,180,0),(220,80,20)] },
                DazzlePalette { name: "volcano".to_string(),      colors: vec![(120,0,0),(200,20,0),(255,60,0),(255,160,0),(255,240,180)] },
                DazzlePalette { name: "arctic".to_string(),       colors: vec![(255,255,255),(180,230,255),(80,180,255),(20,100,220),(0,40,180)] },
                DazzlePalette { name: "aurora".to_string(),       colors: vec![(0,200,150),(0,230,80),(100,0,200),(180,0,255),(255,255,255)] },
                DazzlePalette { name: "ocean".to_string(),        colors: vec![(0,20,100),(0,60,200),(0,160,220),(0,210,180),(180,240,240)] },
                DazzlePalette { name: "neon".to_string(),         colors: vec![(255,0,100),(0,255,60),(0,230,255),(255,230,0),(255,0,200)] },
                DazzlePalette { name: "candy".to_string(),        colors: vec![(255,100,180),(150,255,180),(200,150,255),(255,240,100),(255,140,120)] },
                DazzlePalette { name: "toxic".to_string(),        colors: vec![(0,255,60),(180,255,0),(0,255,200),(255,255,0),(200,255,100)] },
                DazzlePalette { name: "void".to_string(),         colors: vec![(40,0,80),(20,0,120),(80,0,100),(10,0,60),(120,0,180)] },
                DazzlePalette { name: "blood".to_string(),        colors: vec![(80,0,0),(160,0,0),(220,0,20),(255,20,20),(120,0,10)] },
                DazzlePalette { name: "midnight".to_string(),     colors: vec![(0,10,60),(0,30,80),(20,0,80),(40,0,100),(0,60,80)] },
                DazzlePalette { name: "christmas".to_string(),    colors: vec![(220,0,0),(255,30,30),(255,255,255),(0,160,40),(255,200,0)] },
                DazzlePalette { name: "halloween".to_string(),    colors: vec![(220,80,0),(150,0,180),(255,120,0),(60,180,0),(180,0,120)] },
                DazzlePalette { name: "synthwave".to_string(),    colors: vec![(255,0,120),(180,0,255),(0,220,255),(120,0,220),(255,60,180)] },
            ],
            base_hue: 0.0,
            sensitivity: 1.0,
            smoothing: 0.15,
            bass_amplify: 0.5,
            fps: 20,
            band_sensitivity: BandSensitivity::default(),
            band_smoothing: BandSmoothing::default(),
            freq_set: "default".to_string(),
            freq_sets: vec![
                FrequencySet::default_set(),
                FrequencySet::drums_set(),
            ],
            layout: "default".to_string(),
            layouts: vec![
                ("default".to_string(),   [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]),
                ("symmetric".to_string(), [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]),
                ("treble".to_string(),    [3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7]),
            ],
            global_mix: 0.4,
            local_mix: 0.6,
            global_decay: 0.998,
            band_decay: 0.999,
            response_curve: 0.6,
            dazzle_band_frames: 8,
        }
    }
}

impl AudioConfig {
    /// Load from ~/.config/monsgeek-akko/audio.toml, falling back to defaults
    pub fn load() -> Self {
        let mut cfg = AudioConfig::default();
        let path = dirs::home_dir()
            .unwrap_or_default()
            .join(".config/monsgeek-akko/audio.toml");

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                eprintln!("No config file found, using defaults");
                return cfg;
            }
        };

        let mut section = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() { continue; }

            if line.starts_with('[') {
                section = line.trim_matches(|c| c == '[' || c == ']').to_string();
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 { continue; }
            let key = parts[0].trim();
            let val_raw = parts[1].split('#').next().unwrap_or("").trim();
            let val = val_raw.trim_matches('"');

            match section.as_str() {
                "" => match key {
                    "source"             => cfg.source = Some(val.to_string()),
                    "color_mode"         => cfg.color_mode = val.to_string(),
                    "dazzle_palette"     => cfg.dazzle_palette = val.to_string(),
                    "layout"             => cfg.layout = val.to_string(),
                    "freq_set"           => cfg.freq_set = val.to_string(),
                    "sensitivity"        => cfg.sensitivity = val.parse().unwrap_or(cfg.sensitivity),
                    "smoothing"          => cfg.smoothing = val.parse().unwrap_or(cfg.smoothing),
                    "bass_amplify"       => cfg.bass_amplify = val.parse().unwrap_or(cfg.bass_amplify),
                    "fps"                => cfg.fps = val.parse().unwrap_or(cfg.fps),
                    "global_mix"         => cfg.global_mix = val.parse().unwrap_or(cfg.global_mix),
                    "local_mix"          => cfg.local_mix = val.parse().unwrap_or(cfg.local_mix),
                    "global_decay"       => cfg.global_decay = val.parse().unwrap_or(cfg.global_decay),
                    "band_decay"         => cfg.band_decay = val.parse().unwrap_or(cfg.band_decay),
                    "response_curve"     => cfg.response_curve = val.parse().unwrap_or(cfg.response_curve),
                    "dazzle_band_frames" => cfg.dazzle_band_frames = val.parse().unwrap_or(cfg.dazzle_band_frames),
                    _ => {}
                },
                "band_sensitivity" => match key {
                    "sub_bass"   => cfg.band_sensitivity.values[0] = val.parse().unwrap_or(1.0),
                    "bass"       => cfg.band_sensitivity.values[1] = val.parse().unwrap_or(1.0),
                    "low_mid"    => cfg.band_sensitivity.values[2] = val.parse().unwrap_or(1.1),
                    "mid"        => cfg.band_sensitivity.values[3] = val.parse().unwrap_or(1.2),
                    "upper_mid"  => cfg.band_sensitivity.values[4] = val.parse().unwrap_or(1.4),
                    "presence"   => cfg.band_sensitivity.values[5] = val.parse().unwrap_or(1.6),
                    "brilliance" => cfg.band_sensitivity.values[6] = val.parse().unwrap_or(2.2),
                    "air"        => cfg.band_sensitivity.values[7] = val.parse().unwrap_or(3.0),
                    _ => {}
                },
                "band_smoothing" => match key {
                    "sub_bass"   => cfg.band_smoothing.values[0] = val.parse().unwrap_or(0.05),
                    "bass"       => cfg.band_smoothing.values[1] = val.parse().unwrap_or(0.05),
                    "low_mid"    => cfg.band_smoothing.values[2] = val.parse().unwrap_or(0.1),
                    "mid"        => cfg.band_smoothing.values[3] = val.parse().unwrap_or(0.15),
                    "upper_mid"  => cfg.band_smoothing.values[4] = val.parse().unwrap_or(0.2),
                    "presence"   => cfg.band_smoothing.values[5] = val.parse().unwrap_or(0.25),
                    "brilliance" => cfg.band_smoothing.values[6] = val.parse().unwrap_or(0.3),
                    "air"        => cfg.band_smoothing.values[7] = val.parse().unwrap_or(0.3),
                    _ => {}
                },
                "freq_sets" => {
                    // Parse: name = [low, high, weight, low, high, weight, ...] (24 values)
                    let stripped = val.trim_matches(|c| c == '[' || c == ']');
                    let parsed: Vec<f32> = stripped.split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parsed.len() == NUM_BANDS * 3 {
                        let mut bands = [(0.0f32, 0.0f32, 0.0f32); NUM_BANDS];
                        for i in 0..NUM_BANDS {
                            bands[i] = (parsed[i*3], parsed[i*3+1], parsed[i*3+2]);
                        }
                        if let Some(existing) = cfg.freq_sets.iter_mut().find(|s| s.name == key) {
                            existing.bands = bands;
                        } else {
                            cfg.freq_sets.push(FrequencySet { name: key.to_string(), bands });
                        }
                    } else {
                        eprintln!("Warning: freq_set '{}' needs {} values (low,high,weight per band), got {}",
                            key, NUM_BANDS * 3, parsed.len());
                    }
                },
                "layouts" => {
                    let stripped = val.trim_matches(|c| c == '[' || c == ']');
                    let parsed: Vec<usize> = stripped.split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parsed.len() == 16 {
                        let mut arr = [0usize; COLS];
                        for i in 0..16 { arr[i] = parsed[i].min(NUM_BANDS - 1); }
                        if let Some(existing) = cfg.layouts.iter_mut().find(|(n, _)| n == key) {
                            existing.1 = arr;
                        } else {
                            cfg.layouts.push((key.to_string(), arr));
                        }
                    } else {
                        eprintln!("Warning: layout '{}' needs 16 entries, got {}", key, parsed.len());
                    }
                },
                "palettes" => {
                    // Parse: name = [[r,g,b],[r,g,b],...]
                    let stripped = val.trim().trim_matches(|c| c == '[' || c == ']');
                    let mut colors: Vec<(u8, u8, u8)> = Vec::new();
                    for entry in stripped.split(']') {
                        let entry = entry.trim().trim_matches(|c| c == '[' || c == ',');
                        let nums: Vec<u8> = entry.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() == 3 {
                            colors.push((nums[0], nums[1], nums[2]));
                        }
                    }
                    if !colors.is_empty() {
                        if let Some(existing) = cfg.palettes.iter_mut().find(|p| p.name == key) {
                            existing.colors = colors;
                        } else {
                            cfg.palettes.push(DazzlePalette { name: key.to_string(), colors });
                        }
                    } else {
                        eprintln!("Warning: palette '{}' has no valid colors", key);
                    }
                },
                _ => {}
            }
        }

        eprintln!("Loaded config from {}", path.display());
        eprintln!("  mode={} palette={} sens={} smooth={} bass_amp={} fps={}",
            cfg.color_mode, cfg.dazzle_palette, cfg.sensitivity,
            cfg.smoothing, cfg.bass_amplify, cfg.fps);
        eprintln!("  band_sens={:?}", cfg.band_sensitivity.values);
        eprintln!("  layout={} available={:?}", cfg.layout,
            cfg.layouts.iter().map(|(n,_)| n.as_str()).collect::<Vec<_>>());
        cfg
    }

    /// Get the active frequency set
    pub fn active_freq_set(&self) -> &FrequencySet {
        self.freq_sets.iter()
            .find(|s| s.name == self.freq_set)
            .unwrap_or(&self.freq_sets[0])
    }

    /// Get the active column layout
    pub fn active_layout(&self) -> &[usize; COLS] {
        self.layouts.iter()
            .find(|(name, _)| name == &self.layout)
            .map(|(_, bands)| bands)
            .unwrap_or(&self.layouts[0].1)
    }

    /// Get the active dazzle palette
    pub fn active_palette(&self) -> &DazzlePalette {
        self.palettes.iter()
            .find(|p| p.name == self.dazzle_palette)
            .unwrap_or(&self.palettes[0])
    }
}

// ─── Audio State ──────────────────────────────────────────────────────────────

pub struct AudioState {
    pub bands: Mutex<[f32; NUM_BANDS]>,
    pub running: AtomicBool,
    pub sample_rate: AtomicU32,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            bands: Mutex::new([0.0; NUM_BANDS]),
            running: AtomicBool::new(false),
            sample_rate: AtomicU32::new(44100),
        }
    }
}

impl AudioState {
    pub fn get_bands(&self) -> [f32; NUM_BANDS] { *self.bands.lock().unwrap() }
    pub fn set_bands(&self, b: [f32; NUM_BANDS]) { *self.bands.lock().unwrap() = b; }
    pub fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    pub fn stop(&self) { self.running.store(false, Ordering::SeqCst); }
}

// ─── Audio Capture ────────────────────────────────────────────────────────────

pub struct AudioCapture {
    pub state: Arc<AudioState>,
    _stream: Box<dyn std::any::Any>,
}

impl AudioCapture {
    pub fn start(config: &AudioConfig) -> Result<Self, String> {
        let state = Arc::new(AudioState::default());
        let host = cpal::default_host();

        std::env::set_var("ALSA_DEBUG", "0");
        eprintln!("(Ignoring ALSA warnings below - they're harmless)");

        let audio_device = find_audio_device(&host, config.source.as_deref())?;
        let audio_config = audio_device
            .default_input_config()
            .map_err(|e| format!("Failed to get audio config: {e}"))?;

        let sample_rate = audio_config.sample_rate().0;
        state.sample_rate.store(sample_rate, Ordering::SeqCst);

        let sample_buffer: Arc<Mutex<Vec<f32>>> =
            Arc::new(Mutex::new(Vec::with_capacity(FFT_SIZE * 4)));
        let buf_clone = Arc::clone(&sample_buffer);

        let stream = audio_device
            .build_input_stream(
                &audio_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut buf) = buf_clone.lock() {
                        buf.extend_from_slice(data);
                        if buf.len() > FFT_SIZE * 8 {
                            let drain = buf.len() - FFT_SIZE * 4;
                            buf.drain(..drain);
                        }
                    }
                },
                |err| eprintln!("Audio stream error: {err}"),
                None,
            )
            .map_err(|e| format!("Failed to build audio stream: {e}"))?;

        stream.play().map_err(|e| format!("Failed to start audio stream: {e}"))?;
        state.running.store(true, Ordering::SeqCst);

        let state_clone = Arc::clone(&state);
        let _smoothing = config.smoothing;
        let band_sens = config.band_sensitivity.clone();
        let band_smooth = config.band_smoothing.clone();
        let freq_set = config.active_freq_set().clone();
        let global_mix = config.global_mix;
        let local_mix = config.local_mix;
        let global_decay = config.global_decay;
        let band_decay = config.band_decay;
        let response_curve = config.response_curve;

        thread::spawn(move || {
            let mut smoothed = [0.0f32; NUM_BANDS];
            let mut global_ref = 0.01f32;
            let mut band_ref = [0.01f32; NUM_BANDS];
            let interval = Duration::from_millis(8);

            while state_clone.running.load(Ordering::SeqCst) {
                let start = Instant::now();
                let sr = state_clone.sample_rate.load(Ordering::SeqCst);

                let samples: Vec<f32> = {
                    if let Ok(buf) = sample_buffer.lock() {
                        let len = buf.len();
                        if len >= FFT_SIZE {
                            buf[len - FFT_SIZE..].to_vec()
                        } else {
                            vec![0.0; FFT_SIZE]
                        }
                    } else {
                        vec![0.0; FFT_SIZE]
                    }
                };

                let raw = analyze_spectrum(
                    &samples,
                    sr,
                    &band_sens,
                    &freq_set,
                    &mut global_ref,
                    &mut band_ref,
                    global_mix,
                    local_mix,
                    global_decay,
                    band_decay,
                    response_curve,
                );

                // Per-band asymmetric smoothing: attack fast, decay at per-band rate
                for i in 0..NUM_BANDS {
                    let s = band_smooth.values[i];
                    if raw[i] > smoothed[i] {
                        smoothed[i] = smoothed[i] * (s * 0.5) + raw[i] * (1.0 - s * 0.5);
                    } else {
                        smoothed[i] = smoothed[i] * s + raw[i] * (1.0 - s);
                    }
                }

                state_clone.set_bands(smoothed);

                let elapsed = start.elapsed();
                if elapsed < interval {
                    thread::sleep(interval - elapsed);
                }
            }
        });

        Ok(Self { state, _stream: Box::new(stream) })
    }

    pub fn stop(&self) { self.state.stop(); }
    pub fn get_bands(&self) -> [f32; NUM_BANDS] { self.state.get_bands() }
}

// ─── Spectrum Analysis ────────────────────────────────────────────────────────

fn analyze_spectrum(
        samples: &[f32],
        sample_rate: u32,
        band_sens: &BandSensitivity,
        freq_set: &FrequencySet,
        global_ref: &mut f32,
        band_ref: &mut [f32; NUM_BANDS],
        global_mix: f32,
        local_mix: f32,
        global_decay: f32,
        band_decay: f32,
        response_curve: f32,
    ) -> [f32; NUM_BANDS] {
    let mut bands = [0.0f32; NUM_BANDS];

    if samples.len() < FFT_SIZE { return bands; }

    let max_sample = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_sample < 0.0005 { return bands; }

    let windowed: Vec<f32> = hann_window(&samples[..FFT_SIZE]).to_vec();
    let max_freq = (sample_rate / 2) as f32;
    let freq_limit = FrequencyLimit::Range(20.0, max_freq.min(20000.0));

    let spectrum = match samples_fft_to_spectrum(&windowed, sample_rate, freq_limit, Some(&divide_by_N_sqrt)) {
        Ok(s) => s,
        Err(_) => return bands,
    };

    let band_ranges = &freq_set.bands;

    let mut counts = [0u32; NUM_BANDS];
    for (freq, magnitude) in spectrum.data().iter() {
        let fhz = freq.val();
        for (i, (low, high, _)) in band_ranges.iter().enumerate() {
            if fhz >= *low && fhz < *high {
                bands[i] += magnitude.val();
                counts[i] += 1;
            }
        }
    }

    let max_band = {
        let mut tmp = [0.0f32; NUM_BANDS];
        for i in 0..NUM_BANDS {
            if counts[i] > 0 {
                tmp[i] = (bands[i] / counts[i] as f32) * band_ranges[i].2 * band_sens.values[i];
            }
        }
        tmp.iter().fold(0.0f32, |a, &b| f32::max(a, b))
    };

    if max_band < 0.0005 { return [0.0; NUM_BANDS]; }

    let mut raw_vals = [0.0f32; NUM_BANDS];
    for i in 0..NUM_BANDS {
        if counts[i] > 0 {
            raw_vals[i] = (bands[i] / counts[i] as f32) * band_ranges[i].2 * band_sens.values[i];
        }
    }

    if std::env::var("RUST_LOG").is_ok() {
        eprintln!("[SPEC] max={:.4} raw={:.3?}", max_band, raw_vals);
    }

    *global_ref = (*global_ref).max(max_band);
    *global_ref *= global_decay;
    let global_reference = (*global_ref).max(0.001);

    for i in 0..NUM_BANDS {
        if raw_vals[i] > band_ref[i] {
            band_ref[i] = raw_vals[i];
        } else {
            band_ref[i] *= band_decay;
        }
        let local_reference = band_ref[i].max(0.001);
        let global_norm = (raw_vals[i] / global_reference).min(1.0);
        let local_norm  = (raw_vals[i] / local_reference).min(1.0);
        let hybrid = global_norm * global_mix + local_norm * local_mix;
        bands[i] = hybrid.powf(response_curve);
    }

    bands
}

// ─── Color / Rendering ───────────────────────────────────────────────────────

use monsgeek_keyboard::led::RgbColor;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = RgbColor::from_hsv(h, s, v);
    (c.r, c.g, c.b)
}

/// Hue per column for spectrum/gradient modes (smooth rainbow bass→treble)
const COL_HUE: [f32; COLS] = [
    0.0, 15.0, 30.0,
    45.0, 60.0, 75.0,
    100.0, 120.0,
    150.0, 180.0,
    200.0, 220.0,
    240.0, 260.0,
    280.0,
    300.0,
];

/// Simple LCG pseudo-random number generator
struct SlotRng {
    state: u64,
}

impl SlotRng {
    fn new(seed: u64) -> Self { Self { state: seed ^ 0xdeadbeefcafe } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() as usize) % max
    }
}

fn bands_to_frame(
    bands: &[f32; NUM_BANDS],
    config: &AudioConfig,
    dazzle_seeds: &mut [[u64; ROWS]; COLS],
    band_color_seeds: &mut [u64; NUM_BANDS],
    frame_counter: u32,
) -> [(u8, u8, u8); MATRIX_LEN] {
    let mut leds = [(0u8, 0u8, 0u8); MATRIX_LEN];

    let bass_energy = ((bands[0] + bands[1]) / 2.0).min(1.0);
    let amplify = 1.0 + config.bass_amplify * bass_energy;

    let palette = config.active_palette();
    let layout = config.active_layout();

    // Pre-compute per-band colors for dazzleband mode
    let band_colors: [Option<(u8, u8, u8)>; NUM_BANDS] = {
        let mut bc = [None; NUM_BANDS];
        if config.color_mode == "dazzleband" {
            let interval = config.dazzle_band_frames.max(1);
            let slot = frame_counter / interval;
            for i in 0..NUM_BANDS {
                let stored_slot = (band_color_seeds[i] >> 32) as u32;
                if slot != stored_slot {
                    // New slot: pick a new random color for this band
                    let mut rng = SlotRng::new(
                        band_color_seeds[i] ^ slot as u64 ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15)
                    );
                    let pick = rng.next_usize(palette.len());
                    // Store slot in high 32 bits, color index in low 32 bits
                    band_color_seeds[i] = ((slot as u64) << 32) | (pick as u64);
                }
                let pick = (band_color_seeds[i] & 0xFFFFFFFF) as usize;
                bc[i] = Some(palette.pick(pick));
            }
        }
        bc
    };

    for col in 0..COLS {
        let band_idx = layout[col].min(NUM_BANDS - 1);
        let raw = (bands[band_idx] * config.sensitivity * amplify).min(1.0);

        let lit_rows = (raw * ROWS as f32).round() as usize;

        for row in 0..ROWS {
            let from_bottom = ROWS - 1 - row;
            if from_bottom >= lit_rows { continue; }

            let led_idx = row * COLS + col;

            leds[led_idx] = match config.color_mode.as_str() {
                "solid" => {
                    hsv_to_rgb(config.base_hue, 1.0, raw)
                }
                "gradient" => {
                    let hue = (config.base_hue + COL_HUE[col]) % 360.0;
                    let tip = from_bottom as f32 / lit_rows.max(1) as f32;
                    let bar_brightness = 0.25 + raw * 0.75;
                    let brightness = (0.55 + 0.45 * (1.0 - tip)) * bar_brightness;
                    hsv_to_rgb(hue, 1.0, brightness.min(1.0))
                }
                "dazzle" => {
                    // Each individual LED slot gets a random palette color,
                    // re-randomized when the top of the bar changes
                    if from_bottom == lit_rows.saturating_sub(1) {
                        let mut rng = SlotRng::new(dazzle_seeds[col][row] ^ (raw.to_bits() as u64));
                        dazzle_seeds[col][row] = rng.next();
                    }
                    let mut rng = SlotRng::new(dazzle_seeds[col][row]);
                    let pick = rng.next_usize(palette.len());
                    palette.pick(pick)
                }
                "dazzleband" => {
                    // All columns sharing the same band get the same color.
                    // Color changes every dazzle_band_frames frames.
                    band_colors[band_idx].unwrap_or((255, 255, 255))
                }
                _ => {
                    // "spectrum" — rainbow gradient, brighter at bottom
                    let hue = COL_HUE[col];
                    let tip = from_bottom as f32 / lit_rows.max(1) as f32;
                    let brightness = 0.55 + 0.45 * (1.0 - tip);
                    hsv_to_rgb(hue, 1.0, brightness.min(1.0))
                }
            };
        }
    }

    leds
}

// ─── Main Run Loop ────────────────────────────────────────────────────────────

pub fn run_audio_reactive(
    keyboard: &monsgeek_keyboard::KeyboardInterface,
    config: AudioConfig,
    running: Arc<AtomicBool>,
) -> Result<(), String> {
    println!("Starting audio capture...");
    let capture = AudioCapture::start(&config)?;
    println!("Audio capture started.");

    let frame_duration = Duration::from_millis(1000 / config.fps as u64);

    let mut dazzle_seeds = [[0u64; ROWS]; COLS];
    for col in 0..COLS {
        for row in 0..ROWS {
            dazzle_seeds[col][row] = (col * 100 + row) as u64;
        }
    }

    // Per-band color seeds for dazzleband mode
    // High 32 bits = last frame slot, low 32 bits = palette color index
    let mut band_color_seeds = [0u64; NUM_BANDS];
    for i in 0..NUM_BANDS {
        // Init with impossible slot (0xFFFFFFFF) so first frame always picks a color
        band_color_seeds[i] = 0xFFFFFFFF_00000000u64 | (i as u64);
    }

    let mut frame_counter: u32 = 0;

    running.store(true, Ordering::SeqCst);

    while running.load(Ordering::SeqCst) && capture.state.is_running() {
        let frame_start = Instant::now();

        let bands = capture.get_bands();
        let leds = bands_to_frame(
            &bands,
            &config,
            &mut dazzle_seeds,
            &mut band_color_seeds,
            frame_counter,
        );

        let _ = send_full_frame(keyboard, &leds);

        frame_counter = frame_counter.wrapping_add(1);

        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            thread::sleep(frame_duration - elapsed);
        }
    }

    capture.stop();
    println!("Audio reactive mode stopped");
    Ok(())
}

// ─── Utility / Test Functions ─────────────────────────────────────────────────

pub fn list_audio_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(name) = device.name() {
                devices.push(name);
            }
        }
    }
    devices
}

pub fn test_audio_capture() -> Result<(), String> {
    let host = cpal::default_host();
    let device = find_audio_device(&host, None)?;
    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    println!("Audio device: {name}");
    let config = device.default_input_config().map_err(|e| format!("Config error: {e}"))?;
    println!("Sample rate: {} Hz", config.sample_rate().0);
    println!("Channels: {}", config.channels());
    println!("Sample format: {:?}", config.sample_format());
    Ok(())
}

pub fn test_audio_levels(source: Option<&str>) -> Result<(), String> {
    use std::io::Write;

    let host = cpal::default_host();
    if let Ok(monitor) = get_pulseaudio_monitor() {
        println!("Found monitor source: {monitor}");
        std::env::set_var("PULSE_SOURCE", &monitor);
    }
    let device = find_audio_device(&host, source)?;
    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    println!("Using device: {name}");

    let config = device.default_input_config().map_err(|e| format!("Config error: {e}"))?;
    let sample_rate = config.sample_rate().0;
    println!("Sample rate: {} Hz, channels: {}", sample_rate, config.channels());

    let callback_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let cc = Arc::clone(&callback_count);
    let max_sample = Arc::new(Mutex::new(0.0f32));
    let ms = Arc::clone(&max_sample);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            cc.fetch_add(1, Ordering::Relaxed);
            let local_max = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if let Ok(mut max) = ms.lock() {
                if local_max > *max { *max = local_max; }
            }
        },
        |err| eprintln!("Audio error: {err}"),
        None,
    ).map_err(|e| format!("Failed to build stream: {e}"))?;

    stream.play().map_err(|e| format!("Failed to play: {e}"))?;
    println!("\nListening for 5 seconds...");

    for i in 0..5 {
        thread::sleep(Duration::from_secs(1));
        let callbacks = callback_count.load(Ordering::Relaxed);
        let peak = *max_sample.lock().unwrap();
        let bars = (peak * 50.0).min(50.0) as usize;
        print!("  Second {}: {} callbacks, peak: {:.4} [{}{}]",
            i + 1, callbacks, peak,
            "#".repeat(bars), " ".repeat(50 - bars));
        println!();
        std::io::stdout().flush().ok();
        *max_sample.lock().unwrap() = 0.0;
    }

    drop(stream);
    println!("\nDone. Total callbacks: {}", callback_count.load(Ordering::Relaxed));
    Ok(())
}

fn find_audio_device(host: &cpal::Host, source: Option<&str>) -> Result<cpal::Device, String> {
    if let Some(src) = source {
        eprintln!("Using specified source: {src}");
        std::env::set_var("PULSE_SOURCE", src);
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    if name == "pulse" { return Ok(device); }
                }
            }
        }
        return Err(format!("Specified source '{src}' not found — is 'pulse' device available?"));
    }

    if let Ok(monitor) = get_pulseaudio_monitor() {
        eprintln!("Auto-detected monitor source: {monitor}");
        std::env::set_var("PULSE_SOURCE", &monitor);
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    if name == "pulse" { return Ok(device); }
                }
            }
        }
    }

    host.default_input_device().ok_or_else(|| "No audio input device found".to_string())
}

fn get_pulseaudio_monitor() -> Result<String, String> {
    let output = std::process::Command::new("pactl")
        .args(["list", "sources", "short"])
        .output()
        .map_err(|e| format!("Failed to run pactl: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 && parts[1].contains(".monitor") {
            return Ok(parts[1].to_string());
        }
    }
    Err("No monitor source found".to_string())
}