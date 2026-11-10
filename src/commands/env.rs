use std::path::Path;

use crate::config::Config;
use crate::errors::VvmError;

/// Print shell-evaluable environment variables for the current version.
/// Called by the shell hook after `vvm use` / `vvm install`,
/// or manually by the user: eval "$(vvm env)"
pub fn run(config: &Config, shell: &str) -> Result<(), VvmError> {
    let link = config.current_link();
    if !link.is_symlink() {
        return Ok(());
    }

    let target = std::fs::read_link(&link)?;
    let venv_path = target.join("venv");

    if !venv_path.exists() {
        return Ok(());
    }

    let bin_path = venv_path.join("bin");

    if shell == "fish" {
        print_fish_env(config, &venv_path, &bin_path);
        return Ok(());
    }

    // Output shell commands to eval.
    // Strategy: strip any previous vvm venv bin from PATH, then prepend the new one.
    println!("export VVM_DIR=\"{}\";", config.base_dir.display());
    println!("export VIRTUAL_ENV=\"{}\";", venv_path.display());

    // Remove previous vvm bin from PATH using shell parameter expansion (no sed)
    println!(
        r#"case ":$PATH:" in *":${{__VVM_PREV_BIN:-}}:"*) PATH="${{PATH//$__VVM_PREV_BIN:/}}";; esac;"#
    );
    println!("export __VVM_PREV_BIN=\"{}\";", bin_path.display());
    println!("export PATH=\"{}:$PATH\";", bin_path.display());
    println!("hash -r 2>/dev/null;");

    Ok(())
}

fn print_fish_env(config: &Config, venv_path: &Path, bin_path: &Path) {
    let previous_bin = std::env::var_os("__VVM_PREV_BIN");
    let inherited_path = std::env::var_os("PATH").unwrap_or_default();
    let paths = std::iter::once(bin_path.to_path_buf()).chain(
        std::env::split_paths(&inherited_path)
            .filter(|entry| previous_bin.as_deref() != Some(entry.as_os_str())),
    );
    let path = paths
        .map(|entry| quote_fish(&entry))
        .collect::<Vec<_>>()
        .join(" ");

    println!("set -gx VVM_DIR {};", quote_fish(&config.base_dir));
    println!("set -gx VIRTUAL_ENV {};", quote_fish(venv_path));
    println!("set -gx __VVM_PREV_BIN {};", quote_fish(bin_path));
    println!("set -gx PATH {path};");
}

fn quote_fish(path: &Path) -> String {
    format!(
        "'{}'",
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('\'', "\\'")
    )
}
