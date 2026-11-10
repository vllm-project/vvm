mod commands;
mod config;
mod cuda;
mod errors;
mod fsutil;
mod github;
mod metadata;
mod resolve;
mod venv;

use clap::{Parser, Subcommand};
use std::os::unix::process::CommandExt;

use config::Config;

#[derive(Parser)]
#[command(name = "vvm", about = "vLLM Version Manager", version)]
pub struct Cli {
    /// Show verbose output (uv/git logs)
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Install a vllm version
    Install {
        /// Version specifier: 0.18.0, stable, commit:abc123, branch:main, pr:1234, nightly
        specifier: Option<String>,
        /// CUDA variant: cu128, cu129, cu130, cpu (auto-detected if omitted)
        #[arg(long)]
        cuda: Option<String>,
        /// Python executable to use for the virtual environment
        #[arg(long)]
        python: Option<String>,
        /// Build from source with VLLM_USE_PRECOMPILED=1
        #[arg(long)]
        source: bool,
        /// Path to local vllm repo (editable install with precompiled C++/CUDA)
        #[arg(long)]
        path: Option<String>,
        /// Override commit hash for precompiled wheel (use with --path). Can be "nightly".
        #[arg(long)]
        wheel_commit: Option<String>,
        /// Exact wheel URL or local file path for precompiled wheel (use with --path)
        #[arg(long)]
        wheel_location: Option<String>,
        /// Remove existing install first, then reinstall (for updating nightly/pr/local)
        #[arg(long, short)]
        force: bool,
        /// Compile C++/CUDA extensions from source instead of using precompiled wheel (use with --path)
        #[arg(long)]
        compile: bool,
        /// Checkout a git ref (branch, tag, or commit SHA) in the local repo before installing (use with --path or --repo)
        #[arg(long, short, visible_alias = "ref")]
        branch: Option<String>,
        /// Git URL to clone and install from (e.g. git@github.com:org/vllm.git)
        #[arg(long)]
        repo: Option<String>,
        /// Auto-approve interactive prompts (e.g. compile fallback when no precompiled wheel exists)
        #[arg(long, short)]
        yes: bool,
    },
    /// Switch to an installed vllm version
    Use {
        /// Version specifier (reads .vvmrc if omitted)
        specifier: Option<String>,
    },
    /// List installed versions
    Ls {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// List available remote releases
    LsRemote {
        /// Filter pattern (e.g. "0.17")
        pattern: Option<String>,
        /// Include pre-release versions
        #[arg(long)]
        pre: bool,
    },
    /// Show the currently active version
    Current,
    /// Show detailed info for an installed version
    Info {
        /// Version specifier (uses current version if omitted)
        specifier: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Uninstall one or more vllm versions
    Uninstall {
        /// Versions to uninstall
        #[arg(required = true, num_args = 1..)]
        specifiers: Vec<String>,
    },
    /// Create or list aliases
    Alias {
        /// Alias name (omit to list all aliases)
        name: Option<String>,
        /// Version to alias to
        target: Option<String>,
    },
    /// Remove an alias
    Unalias {
        /// Alias name to remove
        name: String,
    },
    /// Run uv pip in the current version's venv (e.g. vvm pip install fastsafetensors)
    Pip {
        /// Arguments passed to uv pip
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Install an auxiliary library that can't build cleanly via pip
    /// (needs git submodules, vendored CUDA, etc). Available: deepgemm, flashinfer, deepep
    InstallExtra {
        /// Library name (deepgemm, flashinfer, deepep)
        name: String,
        /// Ref to install: commit sha, tag, branch, or a release version like
        /// 0.6.7. Defaults to the pin recorded by the active vllm.
        #[arg(long)]
        r#ref: Option<String>,
        /// Override the upstream repo URL
        #[arg(long)]
        repo: Option<String>,
    },
    /// Print shell environment variables for the current version
    Env {
        /// Shell syntax for the activation commands
        #[arg(long, default_value = "bash", value_parser = ["bash", "zsh", "fish"])]
        shell: String,
    },
    /// Print shell initialization code
    Init {
        /// Shell type: bash, zsh, fish
        shell: String,
    },
    /// Execute a command with a specific vllm version
    Exec {
        /// Version specifier
        specifier: String,
        /// Command and arguments to execute
        #[arg(last = true)]
        cmd: Vec<String>,
    },
    /// List installed version names (for shell completion)
    #[command(hide = true)]
    ListInstalled,
}

fn main() {
    let cli = Cli::parse();
    let config = Config::load();

    if cli.verbose {
        std::env::set_var("VVM_VERBOSE", "1");
    }

    let result = match cli.command {
        Commands::Install {
            specifier,
            cuda,
            python,
            source,
            path,
            wheel_commit,
            wheel_location,
            force,
            compile,
            branch,
            repo,
            yes,
        } => commands::install::run(
            &config,
            specifier.as_deref(),
            cuda.as_deref(),
            python.as_deref(),
            source,
            path.as_deref(),
            wheel_commit.as_deref(),
            wheel_location.as_deref(),
            force,
            compile,
            branch.as_deref(),
            repo.as_deref(),
            yes,
        ),
        Commands::Use { specifier } => commands::use_cmd::run(&config, specifier.as_deref()),
        Commands::Ls { json } => commands::ls::run(&config, json),
        Commands::LsRemote { pattern, pre } => commands::ls_remote::run(pattern.as_deref(), pre),
        Commands::Current => commands::current::run(&config),
        Commands::Info { specifier, json } => {
            commands::info::run(&config, specifier.as_deref(), json)
        }
        Commands::Uninstall { specifiers } => {
            let mut any_err = false;
            for s in &specifiers {
                if let Err(e) = commands::uninstall::run(&config, s) {
                    eprintln!("error: {}", e);
                    any_err = true;
                }
            }
            if any_err {
                std::process::exit(1);
            }
            Ok(())
        }
        Commands::Pip { args } => commands::pip::run(&config, &args),
        Commands::InstallExtra {
            name,
            r#ref: git_ref,
            repo,
        } => commands::install_extra::run(&config, &name, git_ref.as_deref(), repo.as_deref()),
        Commands::Alias { name, target } => {
            commands::alias::run(&config, name.as_deref(), target.as_deref())
        }
        Commands::Unalias { name } => commands::alias::remove(&config, &name),
        Commands::Env { shell } => commands::env::run(&config, &shell),
        Commands::Init { shell } => commands::init::run(&shell),
        Commands::Exec { specifier, cmd } => exec_with_version(&config, &specifier, &cmd),
        Commands::ListInstalled => {
            let versions_dir = config.versions_dir();
            if versions_dir.exists() {
                let mut entries: Vec<String> = std::fs::read_dir(&versions_dir)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .filter(|e| e.path().is_dir())
                            .map(|e| e.file_name().to_string_lossy().to_string())
                            .filter(|name| !name.starts_with('.'))
                            .collect()
                    })
                    .unwrap_or_default();
                entries.sort();
                for name in entries {
                    println!("{}", name);
                }
            }
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("{}: {}", colored::Colorize::red("error"), e);
        std::process::exit(1);
    }
}

fn exec_with_version(
    config: &Config,
    specifier_str: &str,
    cmd: &[String],
) -> Result<(), errors::VvmError> {
    if cmd.is_empty() {
        return Err(errors::VvmError::InvalidSpecifier(
            "No command specified. Usage: vvm exec <version> -- <command>".to_string(),
        ));
    }

    // Try raw string as directory name first (handles local-*, raw dir names, etc.)
    let dir_name = {
        let raw_dir = config.versions_dir().join(specifier_str);
        if raw_dir.exists() {
            specifier_str.to_string()
        } else {
            let specifier = resolve::VersionSpecifier::parse(specifier_str)?;
            specifier.dir_name()
        }
    };
    let version_dir = config.versions_dir().join(&dir_name);

    if !version_dir.exists() {
        return Err(errors::VvmError::VersionNotFound(format!(
            "'{}' is not installed",
            dir_name
        )));
    }

    let venv_path = version_dir.join("venv");
    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    // Resolve the command to run within the venv:
    // - pip/pip3: uv pip (uv is already a required dependency; respects VIRTUAL_ENV)
    // - other commands not in venv/bin: try python -m {cmd}
    // - otherwise: run directly (found in venv/bin or absolute path)
    let cmd_name = &cmd[0];
    let (exe, exec_args): (String, Vec<String>) = if matches!(cmd_name.as_str(), "pip" | "pip3") {
        let mut args = vec!["pip".to_string()];
        args.extend_from_slice(&cmd[1..]);
        ("uv".to_string(), args)
    } else if !cmd_name.contains('/') && !bin_dir.join(cmd_name).exists() {
        let python = bin_dir.join("python").to_string_lossy().to_string();
        let mut args = vec!["-m".to_string(), cmd_name.clone()];
        args.extend_from_slice(&cmd[1..]);
        (python, args)
    } else {
        (cmd_name.clone(), cmd[1..].to_vec())
    };

    let err = std::process::Command::new(&exe)
        .args(&exec_args)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", &venv_path)
        .exec();

    // exec() only returns on failure
    Err(errors::VvmError::CommandFailed {
        cmd: exe,
        detail: err.to_string(),
    })
}
