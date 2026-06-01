//! Reactive mode command handlers (audio, screen).

use super::{setup_interrupt_handler, CmdCtx, CommandResult};

/// Run audio reactive LED mode
pub fn audio(
    ctx: &CmdCtx,
    source_override: Option<&str>,
) -> CommandResult {
    let keyboard = super::open_keyboard(ctx).map_err(|e| format!("Failed to open device: {e}"))?;

    // Load config from file, then apply CLI overrides
    let mut config = iot_driver::audio_reactive::AudioConfig::load();
    if let Some(src) = source_override {
        config.source = Some(src.to_string());
    }

    println!("Starting audio reactive mode on {}...", keyboard.device_name());
    println!("  mode={} palette={} sensitivity={} smoothing={} bass_amplify={} fps={}",
        config.color_mode, config.dazzle_palette, config.sensitivity,
        config.smoothing, config.bass_amplify, config.fps);
    println!("Press Ctrl+C to stop");

    let running = setup_interrupt_handler();
    if let Err(e) = iot_driver::audio_reactive::run_audio_reactive(&keyboard, config, running) {
        eprintln!("Audio reactive error: {e}");
    }
    Ok(())
}

/// Test audio capture - list devices
pub fn audio_test(list: bool) -> CommandResult {
    println!("Available cpal input devices:");
    for name in iot_driver::audio_reactive::list_audio_devices() {
        println!("  - {name}");
    }

    println!("\nPipeWire/PulseAudio monitor sources (recommended --source values):");
    if let Ok(output) = std::process::Command::new("pactl")
        .args(["list", "sources", "short"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut found = false;
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let name = parts[1];
                if name.contains(".monitor") {
                    println!("  - {name}");
                    found = true;
                }
            }
        }
        if !found { println!("  (none found)"); }
    } else {
        println!("  (pactl not available)");
    }

    println!("\nConfig file: ~/.config/monsgeek-akko/audio.toml");

    if !list {
        println!();
        if let Err(e) = iot_driver::audio_reactive::test_audio_capture() {
            eprintln!("Audio test failed: {e}");
        }
    }
    Ok(())
}

/// Show real-time audio levels
pub fn audio_levels(source: Option<&str>) -> CommandResult {
    if let Err(e) = iot_driver::audio_reactive::test_audio_levels(source) {
        eprintln!("Audio levels test failed: {e}");
    }
    Ok(())
}

/// Run screen color reactive LED mode
#[cfg(feature = "screen-capture")]
pub async fn screen(ctx: &CmdCtx, fps: u32) -> CommandResult {
    let fps = fps.clamp(1, 60);
    let keyboard = super::open_keyboard(ctx).map_err(|e| format!("Failed to open device: {e}"))?;
    println!("Starting screen color mode on {}...", keyboard.device_name());
    println!("Press Ctrl+C to stop");
    let running = setup_interrupt_handler();
    if let Err(e) = iot_driver::screen_capture::run_screen_color_mode(&keyboard, running, fps).await {
        eprintln!("Screen color mode error: {e}");
    }
    Ok(())
}
