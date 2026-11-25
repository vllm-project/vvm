use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Process-scoped auto-yes flag: when set, `prompt_compile_fallback` returns
/// true without reading stdin. Set once at the top of `run()` from the `-y`
/// CLI flag so we don't have to thread it through every install path.
static AUTO_YES: AtomicBool = AtomicBool::new(false);

/// Set once the new install is live (current symlink updated). The SIGINT
/// handler checks it so a late Ctrl+C doesn't delete a completed install or
/// resurrect the parked old version over it.
static INSTALL_FINALIZED: AtomicBool = AtomicBool::new(false);

use crate::config::Config;
use crate::cuda::{has_gpu_access, CudaVariant};
use crate::errors::VvmError;
use crate::fsutil;
use crate::fsutil::remove_dir_all_robust;
use crate::github;
use crate::metadata::VersionMetadata;
use crate::resolve::VersionSpecifier;
use crate::venv;

#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &Config,
    specifier_str: Option<&str>,
    cuda_override: Option<&str>,
    python: Option<&str>,
    source: bool,
    local_path: Option<&str>,
    wheel_commit: Option<&str>,
    wheel_location: Option<&str>,
    force: bool,
    compile: bool,
    branch: Option<&str>,
    repo_url: Option<&str>,
    yes: bool,
) -> Result<(), VvmError> {
    AUTO_YES.store(yes, Ordering::Relaxed);

    if local_path.is_some() && repo_url.is_some() {
        return Err(VvmError::InvalidSpecifier(
            "--path and --repo are mutually exclusive".to_string(),
        ));
    }
    let has_local = local_path.is_some() || repo_url.is_some();
    if compile && !has_local {
        return Err(VvmError::InvalidSpecifier(
            "--compile requires --path or --repo".to_string(),
        ));
    }
    if compile && (wheel_commit.is_some() || wheel_location.is_some()) {
        return Err(VvmError::InvalidSpecifier(
            "--compile cannot be used with --wheel-commit or --wheel-location".to_string(),
        ));
    }
    if branch.is_some() && !has_local {
        return Err(VvmError::InvalidSpecifier(
            "--branch requires --path or --repo".to_string(),
        ));
    }

    // If --repo is given, clone/fetch to ~/.vvm/repos/ and use as local path.
    let cloned_path = if let Some(url) = repo_url {
        Some(clone_or_fetch_repo(config, url)?)
    } else {
        None
    };
    let local_path = local_path.or(cloned_path.as_deref());

    // Resolve `stable` to the latest non-prerelease tag (e.g. v0.7.0) so the rest
    // of install treats it like a normal release. Only meaningful without a local
    // source — with --path/--repo the specifier is just a name override.
    let resolved_stable;
    let specifier_str = if local_path.is_none()
        && specifier_str
            .map(|s| s.eq_ignore_ascii_case("stable"))
            .unwrap_or(false)
    {
        println!("  {} latest stable release...", "Resolving".blue());
        let tag = github::latest_stable_release()?;
        println!("  {} {}", "Resolved".blue(), tag.cyan());
        resolved_stable = tag;
        Some(resolved_stable.as_str())
    } else {
        specifier_str
    };

    config.ensure_dirs()?;
    venv::ensure_uv()?;

    let cuda = match cuda_override {
        Some(v) => CudaVariant::from_str_loose(v)
            .ok_or_else(|| VvmError::UnsupportedVariant(v.to_string()))?,
        None => {
            let detected = CudaVariant::detect();
            // Auto-detection landing on cpu means nothing here advertised CUDA.
            // Say so — a silently CPU-only vllm is otherwise only discovered
            // much later, when a model fails to load.
            if matches!(detected, CudaVariant::Cpu) {
                eprintln!(
                    "{} no CUDA detected (nvidia-smi, /usr/local/cuda/version.json, nvcc) — installing the {} build.",
                    "Note:".yellow().bold(),
                    "cpu".yellow()
                );
                eprintln!(
                    "      Pass {} or set {} to target a GPU build from this host.",
                    "--cuda cu130".cyan(),
                    "$VVM_CUDA_VARIANT".cyan()
                );
            }
            detected
        }
    };

    // If --path is given, derive specifier from the local repo's git state
    let (dir_name, specifier, commit_hash) = if let Some(lp) = local_path {
        let repo_path = shellexpand::tilde(lp).to_string();
        let repo = std::path::Path::new(&repo_path);
        if !repo.join(".git").exists() {
            return Err(VvmError::InvalidSpecifier(format!(
                "'{}' is not a git repository",
                lp
            )));
        }
        if let Some(br) = branch {
            crate::resolve::validate_ref("branch name", br)?;
            checkout_branch(repo, br)?;
        }
        let (dir_name, commit) = derive_local_dir_name(repo, specifier_str)?;
        let spec = VersionSpecifier::Commit(commit.clone());
        (dir_name, spec, Some(commit))
    } else {
        let spec_str = specifier_str.ok_or_else(|| {
            VvmError::InvalidSpecifier(
                "Version specifier required (or use --path for local repo)".to_string(),
            )
        })?;
        // Try raw string as directory name first (handles local-*, and users
        // copying directory names from `vvm ls` output like `commit-abc123`).
        // Skip the shortcut for dynamic specifiers (Branch/Pr/Nightly) so the
        // normal flow can resolve the HEAD commit and decide whether to update.
        // Only Nightly actually collides here (its dir name == its spec string),
        // but the check is cheap and future-proof.
        let raw_dir = config.versions_dir().join(spec_str);
        let is_dynamic_spec = matches!(
            VersionSpecifier::parse(spec_str),
            Ok(VersionSpecifier::Branch(_))
                | Ok(VersionSpecifier::Pr(_))
                | Ok(VersionSpecifier::Nightly)
        );
        if raw_dir.exists() && !is_dynamic_spec && !reclaim_abandoned_install(&raw_dir, spec_str)? {
            if !force {
                return auto_use(config, spec_str, &raw_dir);
            }
            // --force on an existing version dir: reconstruct the specifier
            // from its metadata so the user can pass the directory name.
            match spec_from_metadata(&raw_dir) {
                Some((spec, hash)) => (spec_str.to_string(), spec, hash),
                None => {
                    // No / unreadable metadata (partial install, or a `local-*`
                    // install that needs --path). Parse as a specifier — with
                    // an interactive y/N prompt on TTY if the input looks like
                    // a directory-name form (e.g. `commit-abc123` → `commit:abc123`).
                    let spec = parse_with_prompt(spec_str)?;
                    if spec.dir_name() != spec_str {
                        return Err(VvmError::InvalidSpecifier(format!(
                            "'{}' is an existing version directory but has no usable metadata. \
                             Pass the original specifier (e.g. commit:<hash>, branch:<name>, pr:<num>, \
                             or --path <repo> for a local install) to reinstall.",
                            spec_str
                        )));
                    }
                    let commit_hash = resolve_commit(&spec)?;
                    (spec.dir_name(), spec, commit_hash)
                }
            }
        } else {
            let spec = parse_with_prompt(spec_str)?;
            let dir_name = spec.dir_name();
            let commit_hash = resolve_commit(&spec)?;
            (dir_name, spec, commit_hash)
        }
    };

    // For nightly, the wheel index publishes the exact commit it currently
    // points to. Resolving up-front lets us show real SHAs in the Updating
    // message and persist commit_hash in metadata.
    let commit_hash = if matches!(specifier, VersionSpecifier::Nightly) && commit_hash.is_none() {
        resolve_nightly_commit(&wheel_client(), &cuda)
    } else {
        commit_hash
    };

    let version_dir = config.versions_dir().join(&dir_name);

    // Finish the job of any interrupted deletion from a previous run.
    fsutil::sweep_stale_trash(&config.versions_dir());

    // Update/force paths park the old install aside instead of deleting it,
    // so a failed or interrupted install restores it (guard Drop + SIGINT
    // handler below). Removed only after the new install succeeds.
    let mut parked: Option<fsutil::ParkedVersion> = None;

    if version_dir.exists() {
        if force {
            println!(
                "{} existing {} ...",
                "Removing".yellow().bold(),
                dir_name.cyan()
            );
            parked = Some(fsutil::ParkedVersion::park(&version_dir).map_err(VvmError::Io)?);
        } else if reclaim_abandoned_install(&version_dir, &dir_name)? {
            // Wreckage of an interrupted install — removed, fall through and
            // install fresh in this same command.
        } else if has_update(&specifier, &commit_hash, &version_dir) {
            // Dynamic type (branch/pr/nightly) with a newer commit available.
            let old_commit = VersionMetadata::load(&version_dir)
                .ok()
                .and_then(|m| m.commit);
            let old_short = old_commit
                .as_deref()
                .map(|c| &c[..12.min(c.len())])
                .unwrap_or("unknown");
            let new_short = commit_hash
                .as_deref()
                .map(|c| &c[..12.min(c.len())])
                .unwrap_or("latest");
            println!(
                "{} {} ({} → {}) ...",
                "Updating".green().bold(),
                dir_name.cyan(),
                old_short,
                new_short,
            );
            parked = Some(fsutil::ParkedVersion::park(&version_dir).map_err(VvmError::Io)?);
        } else {
            return auto_use(config, &dir_name, &version_dir);
        }
    }

    // Atomically claim this install slot. On Linux, mkdir is an atomic syscall.
    // First process to create the directory owns the install; others wait.
    match std::fs::create_dir(&version_dir) {
        Ok(_) => {
            // We own this install. Create a lock marker so waiting processes
            // can distinguish "in progress" from "stale partial install".
            // Format: "hostname:pid" — on a shared home, waiters may be on
            // OTHER nodes where our pid means nothing (see is_lock_stale).
            let lock_path = version_dir.join(".installing");
            if let Err(e) = std::fs::write(
                &lock_path,
                format!("{}:{}", local_hostname(), std::process::id()),
            ) {
                let _ = std::fs::remove_dir_all(&version_dir);
                return Err(VvmError::Io(e));
            }
            // Heartbeat: bump the lock's mtime every 30s so cross-node waiters
            // can tell an active install from a dead one. Touching through an
            // open fd cannot resurrect the file after install removes it, and
            // the thread dies with the process.
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&lock_path) {
                std::thread::spawn(move || loop {
                    std::thread::sleep(Duration::from_secs(30));
                    if f.set_modified(std::time::SystemTime::now()).is_err() {
                        break;
                    }
                });
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Race: another process created the directory after our existence check.
            return wait_for_install(config, &version_dir, &dir_name);
        }
        Err(e) => return Err(VvmError::Io(e)),
    }

    // Clean up the partial install directory if the user hits Ctrl+C mid-install.
    // Without this, a half-built venv with a stale .installing lock gets left behind.
    // If an old version was parked for this update, rename it back — exit()
    // skips Drop, so the ParkedVersion guard can't do it for us here.
    // handler runs on the signal thread; don't hold locks/FS state that needs the
    // main thread to finalize — just rm -rf and exit(130), which is the canonical
    // SIGINT exit code.
    {
        let cleanup_dir = version_dir.clone();
        let restore = parked.as_ref().map(|p| p.restore_paths());
        let _ = ctrlc::set_handler(move || {
            if !INSTALL_FINALIZED.load(Ordering::Relaxed) {
                let _ = std::fs::remove_dir_all(&cleanup_dir);
                if let Some((parked, original)) = &restore {
                    let _ = std::fs::rename(parked, original);
                }
            }
            std::process::exit(130);
        });
    }

    println!(
        "{} {} (cuda: {})",
        "Installing".green().bold(),
        dir_name.cyan(),
        cuda.as_str().yellow()
    );

    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let spinner = if verbose {
        ProgressBar::hidden()
    } else {
        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(Duration::from_millis(100));
        sp
    };

    // Create venv
    spinner.set_message("Creating virtual environment...");
    let venv_path = version_dir.join("venv");
    if let Err(e) = venv::create_venv(&venv_path, python) {
        spinner.finish_and_clear();
        let _ = std::fs::remove_dir_all(&version_dir);
        return Err(e);
    }

    // Install vllm
    spinner.set_message("Installing vllm...");

    // For --compile installs: whether we reused the kernels a previous compile
    // left in the source tree, and the arch list those kernels were built for.
    let mut reused_kernels = false;
    let mut kernels_arch_list: Option<String> = None;

    let install_result = if let Some(lp) = local_path {
        let repo_path = shellexpand::tilde(lp).to_string();
        if compile {
            match try_reuse_compiled_kernels(config, &venv_path, &repo_path, &cuda, &spinner) {
                Some(arch) => {
                    reused_kernels = true;
                    kernels_arch_list = Some(arch);
                    Ok(())
                }
                None => install_from_local_compiled(&venv_path, &repo_path, &cuda, &spinner)
                    .map(|arch| kernels_arch_list = Some(arch)),
            }
        } else {
            install_from_local(
                &venv_path,
                &repo_path,
                &cuda,
                wheel_commit,
                wheel_location,
                &spinner,
            )
        }
    } else if matches!(specifier, VersionSpecifier::Pr(_)) {
        install_from_pr(&venv_path, &specifier, &cuda, &version_dir, &spinner)
    } else if source {
        install_from_source(&venv_path, &specifier, &cuda, &commit_hash, &spinner)
    } else if matches!(specifier, VersionSpecifier::Nightly) {
        // Nightly: try the selected variant's index before any commit fallback.
        install_nightly_with_fallback(&venv_path, &cuda, &spinner)
    } else if matches!(specifier, VersionSpecifier::Branch(_)) {
        // Branch: try HEAD commit wheel first, then walk back recent branch commits
        install_branch_with_fallback(&venv_path, &specifier, &cuda, &commit_hash, &spinner)
    } else {
        // commit:X (and release) — install_from_wheel looks up the exact wheel.
        // For commit:X on miss, fall back to source-mode install: clone the
        // commit + use a nearby wheel for C++/CUDA (install_from_source handles
        // the drift check and the compile prompt internally).
        let res = install_from_wheel(&venv_path, &specifier, &cuda, &commit_hash);
        if res.is_err() && matches!(specifier, VersionSpecifier::Commit(_)) {
            log_info(
                &spinner,
                format!(
                    "  {} no exact wheel for this commit on {}/{}, falling back to source mode...",
                    "Note:".yellow().bold(),
                    std::env::consts::ARCH,
                    cuda.as_str()
                ),
            );
            install_from_source(&venv_path, &specifier, &cuda, &commit_hash, &spinner)
        } else {
            res
        }
    };

    if let Err(e) = install_result {
        spinner.finish_and_clear();
        let _ = std::fs::remove_dir_all(&version_dir);
        let arch = std::env::consts::ARCH;
        let hint = match &specifier {
            VersionSpecifier::Commit(_) | VersionSpecifier::Branch(_) | VersionSpecifier::Pr(_) => {
                format!(
                    "\nHint: pre-built wheel may not exist for {} / {}.\n\
                     The commit's wheel might still be building — try again in ~1 hour,\n\
                     or use --wheel-commit to specify an earlier commit.",
                    arch,
                    cuda.as_str()
                )
            }
            _ => String::new(),
        };
        return Err(if hint.is_empty() {
            e
        } else {
            VvmError::CommandFailed {
                cmd: "install".to_string(),
                detail: format!("{}{}", e, hint),
            }
        });
    }

    // Verify the installation actually works and capture the installed version string.
    let mut compiled_fallback = false;
    let installed_version = {
        spinner.set_message("Verifying installation...");
        match verify_installation(&venv_path, &cuda, &specifier) {
            Ok(v) => Some(v),
            Err(e) => {
                // In --path mode (non-compile), the precompiled wheel may be
                // unusable at runtime (e.g. libcudart mismatch). Offer to retry
                // with source compilation in the same venv. Reused kernels that
                // fail verification mean the reuse was wrong — recompile without
                // asking, since the user already requested --compile.
                let fallback = match local_path {
                    Some(lp) if !compile || reused_kernels => {
                        let proceed = if reused_kernels {
                            log_info(
                                &spinner,
                                format!(
                                    "  {} reused kernels failed verification ({}), recompiling from source...",
                                    "Warning:".yellow().bold(),
                                    e
                                ),
                            );
                            true
                        } else {
                            prompt_compile_fallback(&spinner, &e.to_string(), OnDecline::Abort)
                        };
                        if proceed {
                            reused_kernels = false; // compiling for real now
                            let repo_path = shellexpand::tilde(lp).to_string();
                            // Uninstall the broken precompiled vllm first — leaving
                            // stale .pth/dist-info around can make the compiled
                            // install resolve to the old package.
                            spinner.set_message("Uninstalling broken vllm...");
                            let _ =
                                venv::run_in_venv(&venv_path, "uv", &["pip", "uninstall", "vllm"]);
                            let retry = install_from_local_compiled(
                                &venv_path, &repo_path, &cuda, &spinner,
                            )
                            .and_then(|arch| {
                                kernels_arch_list = Some(arch);
                                spinner.set_message("Verifying installation...");
                                verify_installation(&venv_path, &cuda, &specifier)
                            });
                            Some(retry)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                match fallback {
                    Some(Ok(v)) => {
                        compiled_fallback = true;
                        Some(v)
                    }
                    Some(Err(e2)) => {
                        spinner.finish_and_clear();
                        let _ = std::fs::remove_dir_all(&version_dir);
                        return Err(e2);
                    }
                    None => {
                        spinner.finish_and_clear();
                        let _ = std::fs::remove_dir_all(&version_dir);
                        return Err(e);
                    }
                }
            }
        }
    };

    // Write metadata
    let mut metadata = build_metadata(&specifier, &dir_name, &cuda, &commit_hash);
    if let Some(v) = installed_version {
        metadata.version = Some(v);
    }
    if let Some(lp) = local_path {
        metadata.source_path = Some(shellexpand::tilde(lp).to_string());
        if compile || compiled_fallback {
            metadata.install_type = "local-compiled".to_string();
            // Record the kernels' arch list so later --compile installs of this
            // repo can reuse them — but not when the build included uncommitted
            // C++/CUDA edits, since metadata.commit wouldn't identify what was
            // actually built. (Reuse installs verified worktree cleanliness.)
            let kernels_dirty = !reused_kernels
                && compiled_drift_worktree(&shellexpand::tilde(lp), "HEAD").is_some();
            metadata.arch_list = if kernels_dirty {
                None
            } else {
                kernels_arch_list.take()
            };
        }
    }
    metadata.save(&version_dir).map_err(VvmError::Io)?;

    spinner.finish_and_clear();

    // Auto-use the newly installed version.
    // Use atomic rename so concurrent installs don't race on remove+create.
    let current_link = config.current_link();
    let tmp_link = config
        .base_dir
        .join(format!(".current.tmp.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp_link); // clean up any leftover tmp from a prior crash
    if let Err(e) = std::os::unix::fs::symlink(&version_dir, &tmp_link)
        .and_then(|_| std::fs::rename(&tmp_link, &current_link))
    {
        let _ = std::fs::remove_dir_all(&version_dir); // removes lock too
        return Err(VvmError::Io(e));
    }

    // The new install is live: stop protecting against rollback, then delete
    // the parked old version (updates/--force only).
    INSTALL_FINALIZED.store(true, Ordering::Relaxed);
    if let Some(p) = parked.take() {
        p.commit();
    }

    // Set default alias if this is the first installation
    set_default_if_first(config, &version_dir)?;

    // Remove lock marker only after current symlink is updated — waiting processes
    // will see a fully ready install (metadata + current link) when they wake up.
    let _ = std::fs::remove_file(version_dir.join(".installing"));

    let mut info_parts = vec![format!("cuda: {}", metadata.cuda.yellow())];
    if let Some(ref v) = metadata.version {
        if !v.is_empty() {
            info_parts.insert(0, format!("version: {}", v.cyan()));
        }
    }
    if let Some(ref commit) = metadata.commit {
        let short = &commit[..12.min(commit.len())];
        info_parts.push(format!("commit: {}", short));
    }
    if reused_kernels {
        info_parts.push("kernels: reused".to_string());
    }
    if let Some(pr) = metadata.pr {
        info_parts.push(format!("pr: #{}", pr));
    }
    println!(
        "{} {} installed and active ({})",
        "✓".green().bold(),
        dir_name.cyan(),
        info_parts.join(", ")
    );
    if std::env::var("VVM_DIR").is_err() {
        println!("  Run: eval \"$(vvm env)\"");
    }

    Ok(())
}

/// Reclaim a version directory left behind by an interrupted install.
///
/// Two facts together make a directory provably dead: no `metadata.json` means
/// the install never finished, and no live `.installing` lock means no process
/// is going to finish it. vvm created the directory, nothing in it is the
/// user's, and nobody owns it — so clean it up and let the caller install,
/// rather than stopping to demand a `--force` re-run of the same command.
///
/// Returns true when the directory was removed.
fn reclaim_abandoned_install(
    version_dir: &std::path::Path,
    dir_name: &str,
) -> Result<bool, VvmError> {
    // A completed install is never wreckage — check first so the common path
    // costs one stat and never reaches the grace period below.
    if version_dir.join("metadata.json").exists() {
        return Ok(false);
    }

    let lock_path = version_dir.join(".installing");
    // A live installer is briefly lockless, between its `create_dir` and its
    // `File::create(".installing")`. Give the lock time to show up before
    // calling the directory abandoned — same grace period as wait_for_install.
    if !lock_path.exists() {
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while !lock_path.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    // Someone is actively installing (possibly on another node) — hands off.
    if lock_path.exists() && !is_lock_stale(&lock_path) {
        return Ok(false);
    }

    println!(
        "{} previous install of {} was interrupted, cleaning up...",
        "Warning:".yellow().bold(),
        dir_name.cyan()
    );
    remove_dir_all_robust(version_dir)?;
    Ok(true)
}

/// Auto-switch to an already-installed version instead of erroring.
fn auto_use(
    config: &Config,
    dir_name: &str,
    version_dir: &std::path::Path,
) -> Result<(), VvmError> {
    let meta_path = version_dir.join("metadata.json");
    if !meta_path.exists() {
        let lock_path = version_dir.join(".installing");
        if lock_path.exists() {
            // Another process is installing — fall through to wait logic
            return wait_for_install(config, version_dir, dir_name);
        }
        return Err(VvmError::InvalidSpecifier(format!(
            "'{}' has a partial install (no metadata, no lock). Re-run the install to clean it up automatically.",
            dir_name
        )));
    }

    // Backfill missing version field (e.g. for nightly installs predating the
    // verify-on-install fix). Cheap python call, only runs when degenerate.
    if let Ok(mut meta) = VersionMetadata::load(version_dir) {
        if meta.version.as_deref().unwrap_or("").is_empty() {
            if let Some(v) = quick_capture_version(&version_dir.join("venv")) {
                meta.version = Some(v);
                let _ = meta.save(version_dir);
            }
        }
    }

    let current_link = config.current_link();
    let tmp_link = config
        .base_dir
        .join(format!(".current.tmp.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp_link);
    std::os::unix::fs::symlink(version_dir, &tmp_link).map_err(VvmError::Io)?;
    std::fs::rename(&tmp_link, &current_link).map_err(VvmError::Io)?;

    println!(
        "{} {} already installed, now active",
        "✓".green().bold(),
        dir_name.cyan()
    );
    if std::env::var("VVM_DIR").is_err() {
        println!("  Run: eval \"$(vvm env)\"");
    }
    Ok(())
}

fn wait_for_install(
    config: &Config,
    version_dir: &std::path::Path,
    dir_name: &str,
) -> Result<(), VvmError> {
    let meta_path = version_dir.join("metadata.json");
    let lock_path = version_dir.join(".installing");

    if meta_path.exists() {
        return auto_use(config, dir_name, version_dir);
    }

    // Give the lock file up to 500 ms to appear — handles the tiny window between
    // the installer's `create_dir` and its subsequent `File::create(".installing")`.
    if !lock_path.exists() {
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        while !lock_path.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    if meta_path.exists() {
        return auto_use(config, dir_name, version_dir);
    }

    if !lock_path.exists() {
        // No lock and no metadata — a previous install crashed and left a partial directory.
        return Err(VvmError::InvalidSpecifier(format!(
            "'{}' has a partial install (no metadata, no lock). Re-run the install to clean it up automatically.",
            dir_name
        )));
    }

    // Another process holds the lock — wait for it to finish.
    println!(
        "{} {} is being installed by another process, waiting...",
        "Waiting".yellow().bold(),
        dir_name.cyan()
    );
    let sp = ProgressBar::new_spinner();
    sp.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    sp.enable_steady_tick(Duration::from_millis(100));
    sp.set_message(format!("Waiting for {} ...", dir_name));

    // Long enough for a full --compile install (~20-40 min on GB300); a dead
    // installer is caught much sooner by is_lock_stale's heartbeat check.
    let timeout = Duration::from_secs(7200);
    let poll = Duration::from_secs(2);
    let started = std::time::Instant::now();

    loop {
        std::thread::sleep(poll);

        if meta_path.exists() {
            sp.finish_and_clear();
            return auto_use(config, dir_name, version_dir);
        }

        if !lock_path.exists() {
            // Re-check metadata: the installer writes metadata THEN removes the
            // lock, so there's a tiny window where we could see the lock gone
            // before the metadata write is visible. A second check closes it.
            if meta_path.exists() {
                sp.finish_and_clear();
                return auto_use(config, dir_name, version_dir);
            }
            sp.finish_and_clear();
            return Err(VvmError::CommandFailed {
                cmd: "install".to_string(),
                detail: format!("The process installing '{}' failed. Try again.", dir_name),
            });
        }

        // Detect if the installing process died (e.g. Ctrl+C, kill, OOM)
        if is_lock_stale(&lock_path) {
            sp.finish_and_clear();
            let _ = std::fs::remove_dir_all(version_dir);
            return Err(VvmError::CommandFailed {
                cmd: "install".to_string(),
                detail: format!(
                    "The process installing '{}' was interrupted. Cleaned up — please retry.",
                    dir_name
                ),
            });
        }

        if started.elapsed() > timeout {
            sp.finish_and_clear();
            return Err(VvmError::CommandFailed {
                cmd: "install".to_string(),
                detail: format!(
                    "Timed out waiting for another process to install '{}'",
                    dir_name
                ),
            });
        }
    }
}

/// Check if a dynamic install type (branch/pr/nightly) has a newer version available.
fn has_update(
    specifier: &VersionSpecifier,
    new_commit: &Option<String>,
    version_dir: &std::path::Path,
) -> bool {
    let is_dynamic = matches!(
        specifier,
        VersionSpecifier::Branch(_) | VersionSpecifier::Pr(_) | VersionSpecifier::Nightly
    );
    if !is_dynamic {
        return false;
    }
    match new_commit {
        Some(new) => {
            // Have a resolved commit — compare with what's installed.
            match VersionMetadata::load(version_dir) {
                Ok(meta) => meta.commit.as_deref() != Some(new),
                Err(_) => true, // can't read metadata, treat as needing update
            }
        }
        None => {
            // Nightly: no commit to compare up-front — always update.
            true
        }
    }
}

/// Check whether the `.installing` lock file was left by a dead process.
/// The lock file contains the PID of the installing process.
/// Installs heartbeat-touch the lock every 30s; consider a lock dead when its
/// mtime hasn't moved for this long. Generous slack for shared-FS mtime lag.
const LOCK_HEARTBEAT_STALE_SECS: u64 = 300;

fn local_hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Heartbeat check: the installer touches the lock's mtime every 30s, so a
/// long-untouched lock means the installer is gone (works across nodes).
fn lock_mtime_stale(lock_path: &std::path::Path) -> bool {
    match std::fs::metadata(lock_path).and_then(|m| m.modified()) {
        Ok(mtime) => std::time::SystemTime::now()
            .duration_since(mtime)
            .map(|age| age.as_secs() > LOCK_HEARTBEAT_STALE_SECS)
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn is_lock_stale(lock_path: &std::path::Path) -> bool {
    let contents = match std::fs::read_to_string(lock_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let contents = contents.trim();
    let (host, pid) = match contents.split_once(':') {
        Some((h, p)) => (Some(h), p.parse::<u32>().ok()),
        None => (None, contents.parse::<u32>().ok()),
    };
    match (host, pid) {
        // Our node: /proc/{pid} exists iff the installer is alive. A pid check
        // is exact — no need to wait out the heartbeat window.
        (Some(h), Some(pid)) if h == local_hostname() => {
            !std::path::Path::new(&format!("/proc/{}", pid)).exists()
        }
        // Another node (shared home): its pids mean nothing here — NEVER judge
        // liveness by /proc, that's how active cross-node installs used to get
        // "cleaned up". Trust the mtime heartbeat instead.
        (Some(_), _) => lock_mtime_stale(lock_path),
        // Old bare-pid format (pre-hostname vvm): the pid may be from another
        // node, so only a *live* local pid is conclusive; otherwise fall back
        // to the mtime heartbeat (old installers don't touch, so this ages out
        // in LOCK_HEARTBEAT_STALE_SECS — acceptable for a legacy format).
        (None, Some(pid)) => {
            if std::path::Path::new(&format!("/proc/{}", pid)).exists() {
                false
            } else {
                lock_mtime_stale(lock_path)
            }
        }
        // Empty/garbage contents — ownerless, treat as stale.
        (None, None) => true,
    }
}

/// Parse a version specifier; if the input looks like a directory-name form
/// (e.g. `commit-abc123`, `pr-1234`, `branch-main`), prompt the user to
/// auto-correct to the canonical `kind:value` form when running in a TTY.
/// In non-interactive contexts the original error (with the suggested form
/// in the message) is returned unchanged so scripts fail fast.
fn parse_with_prompt(spec_str: &str) -> Result<VersionSpecifier, VvmError> {
    use std::io::{self, BufRead, IsTerminal, Write};

    match VersionSpecifier::parse(spec_str) {
        Ok(spec) => Ok(spec),
        Err(e) => {
            let Some(corrected) = crate::resolve::suggest_corrected_specifier(spec_str) else {
                return Err(e);
            };
            if !io::stdin().is_terminal() {
                return Err(e);
            }
            print!(
                "{} '{}' looks like a directory name. Install as '{}' instead? [y/N] ",
                "?".cyan().bold(),
                spec_str,
                corrected.cyan(),
            );
            io::stdout().flush().ok();
            let mut line = String::new();
            io::stdin()
                .lock()
                .read_line(&mut line)
                .map_err(VvmError::Io)?;
            match line.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => VersionSpecifier::parse(&corrected),
                _ => Err(e),
            }
        }
    }
}

/// Reconstruct a `VersionSpecifier` from metadata.json, for `--force` reinstalls
/// when the user passes a directory name instead of the original specifier.
/// Returns None if metadata is missing or represents a type we can't reinstall
/// without extra flags (e.g. `local` installs need `--path`).
fn spec_from_metadata(dir: &Path) -> Option<(VersionSpecifier, Option<String>)> {
    let meta = VersionMetadata::load(dir).ok()?;
    match meta.install_type.as_str() {
        "release" => {
            let v = meta
                .name
                .strip_prefix('v')
                .unwrap_or(&meta.name)
                .to_string();
            Some((VersionSpecifier::Release(v), meta.commit))
        }
        "commit" => {
            let hash = meta.commit.clone()?;
            Some((VersionSpecifier::Commit(hash.clone()), Some(hash)))
        }
        "branch" => Some((VersionSpecifier::Branch(meta.branch?), meta.commit)),
        "pr" => Some((VersionSpecifier::Pr(meta.pr?), meta.commit)),
        "nightly" => Some((VersionSpecifier::Nightly, meta.commit)),
        _ => None,
    }
}

fn resolve_commit(specifier: &VersionSpecifier) -> Result<Option<String>, VvmError> {
    match specifier {
        VersionSpecifier::Release(v) => {
            // Resolve tag → commit SHA and store in metadata so vvm info works offline.
            // A definitive "tag not found" fails fast — the release doesn't exist, so
            // no install path (release index or commit-wheel fallback) can succeed.
            // Transient API failures stay non-fatal: release install works without the SHA.
            let tag = format!("v{}", v);
            match github::tag_commit_sha(&tag) {
                Ok(Some(sha)) => Ok(Some(sha)),
                Ok(None) => Err(release_not_found(v)),
                Err(_) => Ok(None),
            }
        }
        VersionSpecifier::Commit(hash) => {
            let full = github::resolve_commit_sha(hash)?;
            Ok(Some(full))
        }
        VersionSpecifier::Branch(branch) => {
            println!(
                "  {} branch '{}' HEAD commit...",
                "Resolving".blue(),
                branch
            );
            let sha = github::branch_head_commit(branch)?;
            println!("  {} {}", "Resolved".blue(), &sha[..12]);
            Ok(Some(sha))
        }
        VersionSpecifier::Pr(num) => {
            println!("  {} PR #{} HEAD commit...", "Resolving".blue(), num);
            let sha = github::pr_head_commit(*num)?;
            println!("  {} {}", "Resolved".blue(), &sha[..12]);
            Ok(Some(sha))
        }
        _ => Ok(None),
    }
}

/// Build the user-facing error for a release tag that doesn't exist upstream,
/// hinting at the latest release (and nightly, when the request is newer).
fn release_not_found(version: &str) -> VvmError {
    let mut msg = format!(
        "{} — tag 'v{}' does not exist in vllm-project/vllm.",
        version, version
    );
    if let Ok(latest) = github::latest_stable_release() {
        msg.push_str(&format!(" Latest release: {}.", latest));
        if version_newer_than(version, &latest) {
            msg.push_str(&format!(
                "\nvLLM {} is not released yet — `vvm install nightly` tracks the latest main build.",
                version
            ));
        }
    }
    msg.push_str("\nRun `vvm ls-remote` to list available versions.");
    VvmError::VersionNotFound(msg)
}

/// Numeric dot-segment comparison ("0.23.0" > "v0.22.1"). Non-numeric suffixes
/// within a segment are ignored; missing segments compare as lower.
fn version_newer_than(a: &str, b: &str) -> bool {
    fn parts(s: &str) -> Vec<u64> {
        s.trim_start_matches('v')
            .split('.')
            .map(|p| {
                p.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            })
            .collect()
    }
    parts(a) > parts(b)
}

/// Try the variant's nightly index first. If no wheel is available, fall back
/// to recent main commits except for the frozen cu129 compatibility index.
fn install_nightly_with_fallback(
    venv_path: &std::path::Path,
    cuda: &CudaVariant,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let client = wheel_client();
    // The unqualified commit indexes used for historical cu129 wheels may
    // now serve a different CUDA variant. Keep cu129 on its compatibility feed.
    if matches!(cuda, CudaVariant::Cu129) || lookup_wheel_url(&client, "nightly", cuda).is_some() {
        let spec = VersionSpecifier::Nightly;
        return install_from_wheel(venv_path, &spec, cuda, &Some("nightly".to_string()));
    }

    // Nightly unavailable — find a recent main commit with a wheel
    println!(
        "  {} nightly unavailable for {}/{}, searching recent commits...",
        "Note:".yellow().bold(),
        std::env::consts::ARCH,
        cuda.as_str()
    );

    let (good_commit, _url) = find_good_wheel_commit(&client, cuda, None, Some(spinner))
        .ok_or_else(|| VvmError::CommandFailed {
            cmd: "nightly install".to_string(),
            detail: format!(
                "No wheel available for {}/{}. Checked nightly + 30 recent main commits.",
                std::env::consts::ARCH,
                cuda.as_str()
            ),
        })?;

    let commit_spec = VersionSpecifier::Commit(good_commit.clone());
    let hash = Some(good_commit);
    install_from_wheel(venv_path, &commit_spec, cuda, &hash)
}

/// Try the branch HEAD commit wheel first. If unavailable, walk back recent branch
/// commits to find the most recent one that has a precompiled wheel.
fn install_branch_with_fallback(
    venv_path: &std::path::Path,
    specifier: &VersionSpecifier,
    cuda: &CudaVariant,
    commit_hash: &Option<String>,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let branch = match specifier {
        VersionSpecifier::Branch(b) => b.as_str(),
        _ => unreachable!(),
    };

    let head_commit = commit_hash
        .as_ref()
        .expect("branch commit_hash must be resolved before install");

    let client = wheel_client();

    // Fast check: does the HEAD commit have a wheel for the current arch?
    if lookup_wheel_url(&client, head_commit, cuda).is_some() {
        return install_from_wheel(venv_path, specifier, cuda, commit_hash);
    }

    println!(
        "  {} no wheel for HEAD ({}) on {}/{}, searching recent commits...",
        "Note:".yellow().bold(),
        &head_commit[..12],
        std::env::consts::ARCH,
        cuda.as_str()
    );

    // Walk back recent branch commits to find one with a wheel
    let commits =
        github::recent_branch_commits(branch, 30).map_err(|e| VvmError::CommandFailed {
            cmd: "branch commit lookup".to_string(),
            detail: e.to_string(),
        })?;

    let total = commits.len();
    for (i, commit) in commits.iter().enumerate() {
        if commit == head_commit {
            continue; // already checked
        }
        spinner.set_message(format!(
            "Searching precompiled wheel ({}/{}): {}...",
            i + 1,
            total,
            &commit[..8.min(commit.len())]
        ));
        if lookup_wheel_url(&client, commit, cuda).is_some() {
            println!(
                "  {} wheel found at {}",
                "Found:".blue().bold(),
                &commit[..12]
            );

            // We walked back from HEAD; the skipped commits may include C++
            // changes the wheel won't have.
            if let Ok(files) = github::compare_changed_files(commit, head_commit) {
                if has_compiled_changes(&files) {
                    let compiled: Vec<String> = files
                        .into_iter()
                        .filter(|f| has_compiled_changes(std::slice::from_ref(f)))
                        .collect();
                    let responsible = github::commits_in_range_touching(commit, head_commit, |f| {
                        has_compiled_changes(std::slice::from_ref(&f.to_string()))
                    })
                    .unwrap_or_default();
                    let requested = format!(
                        "branch '{}' HEAD ({})",
                        branch,
                        &head_commit[..12.min(head_commit.len())]
                    );
                    let reason =
                        format_range_drift_reason(&requested, commit, &compiled, &responsible);
                    if prompt_compile_fallback(spinner, &reason, OnDecline::InstallPrecompiled) {
                        return compile_at_ref(venv_path, specifier, commit_hash, cuda, spinner);
                    }
                    log_info(
                        spinner,
                        format!(
                            "  {} precompiled install will not include upstream C++/CUDA changes between wheel commit {} and branch '{}' HEAD",
                            "Warning:".yellow().bold(),
                            &commit[..12.min(commit.len())],
                            branch
                        ),
                    );
                }
            }

            let fallback_spec = VersionSpecifier::Commit(commit.clone());
            let fallback_hash = Some(commit.clone());
            return install_from_wheel(venv_path, &fallback_spec, cuda, &fallback_hash);
        }
    }

    let reason = format!(
        "No precompiled wheel for branch '{}' HEAD or {} recent commits on {}/{}.",
        branch,
        commits.len(),
        std::env::consts::ARCH,
        cuda.as_str()
    );
    if prompt_compile_fallback(spinner, &reason, OnDecline::Abort) {
        return compile_at_ref(venv_path, specifier, commit_hash, cuda, spinner);
    }

    Err(VvmError::CommandFailed {
        cmd: "branch install".to_string(),
        detail: format!(
            "No wheel available for branch '{}' on {}/{}. Checked HEAD + {} recent commits.",
            branch,
            std::env::consts::ARCH,
            cuda.as_str(),
            commits.len()
        ),
    })
}

fn install_from_wheel(
    venv_path: &std::path::Path,
    specifier: &VersionSpecifier,
    cuda: &CudaVariant,
    commit_hash: &Option<String>,
) -> Result<(), VvmError> {
    let torch_idx = torch_index_url(cuda);
    let torch_backend = format!("--torch-backend={}", cuda.as_str());

    // fastsafetensors 0.3.x ships only x86_64 manylinux wheels on PyPI; on
    // aarch64 uv falls back to sdist. Its setup.py imports pybind11 but does
    // not list it in [build-system.requires], so PEP 517 isolated builds fail
    // with ModuleNotFoundError. Pre-install pybind11 + base build tools into
    // the venv and pair with --no-build-isolation-package fastsafetensors
    // below so the sdist build picks them up. Harmless on x86_64 (uv will
    // pick the wheel and skip the build entirely).
    let fast_build_deps = vec!["pip", "install", "pybind11", "setuptools", "wheel"];
    venv::run_in_venv(venv_path, "uv", &fast_build_deps)?;

    let is_release = matches!(specifier, VersionSpecifier::Release(_));

    if is_release {
        let version = match specifier {
            VersionSpecifier::Release(v) => v,
            _ => unreachable!(),
        };

        if matches!(cuda, CudaVariant::Cu129) {
            // cu129: PyPI has the default wheel for both x86_64 and aarch64
            let pkg = format!("vllm=={}", version);
            let args = vec![
                "pip",
                "install",
                &pkg,
                "--extra-index-url",
                &torch_idx,
                "--index-strategy",
                "unsafe-best-match",
                &torch_backend,
                "--no-build-isolation-package",
                "fastsafetensors",
            ];
            return venv::run_in_venv(venv_path, "uv", &args);
        }

        // Non-cu129: try the release index first, then fall back to the
        // commit-based wheel for the release tag. This prevents silent
        // fallback to PyPI's cu129 default.
        let vllm_index_url = release_index_url(version, cuda);
        let wheel_url = find_release_wheel_url(&vllm_index_url, version, cuda).or_else(|_| {
            // Release index empty — look for a commit-based wheel at the tag's
            // commit instead. The SHA was resolved up-front (resolve_commit);
            // re-resolve only if that failed (e.g. transient GitHub error).
            eprintln!(
                "{} No release wheel for {} at wheels.vllm.ai/{}/{}, trying commit-based wheel...",
                "Warning:".yellow().bold(),
                cuda.as_str(),
                version,
                cuda.as_str(),
            );
            let hash = match commit_hash {
                Some(h) => h.clone(),
                None => {
                    let tag = format!("v{}", version);
                    crate::github::tag_commit_sha(&tag)?
                        .ok_or_else(|| release_not_found(version))?
                }
            };
            let idx = commit_index_url(&hash, cuda);
            find_wheel_url(&idx, cuda)
        })?;
        let args = vec![
            "pip",
            "install",
            &wheel_url,
            "--extra-index-url",
            &torch_idx,
            "--index-strategy",
            "unsafe-best-match",
            &torch_backend,
            "--no-build-isolation-package",
            "fastsafetensors",
        ];
        return venv::run_in_venv(venv_path, "uv", &args);
    }

    // For nightly/commit/branch/PR: find the exact wheel URL and install directly
    // to bypass uv version resolution (wheels have local version labels like
    // +g65b2f405d.cu130 which uv ignores when resolving by name).
    let hash = commit_hash
        .as_ref()
        .expect("commit_hash must be resolved for commit/branch/pr installs");
    let index_url = commit_index_url(hash, cuda);
    let wheel_url = find_wheel_url(&index_url, cuda)?;

    let args = vec![
        "pip",
        "install",
        &wheel_url,
        "--extra-index-url",
        &torch_idx,
        "--index-strategy",
        "unsafe-best-match",
        &torch_backend,
        "--no-build-isolation-package",
        "fastsafetensors",
    ];
    venv::run_in_venv(venv_path, "uv", &args)
}

/// Fetch wheel URL from metadata.json, matching the current arch.
/// metadata.json is a JSON array of wheel entries with platform_tag and path fields.
fn find_wheel_url(index_url: &str, cuda: &CudaVariant) -> Result<String, VvmError> {
    let arch = std::env::consts::ARCH; // "x86_64" or "aarch64"
    let metadata_url = format!("{}/vllm/metadata.json", index_url.trim_end_matches('/'));

    let resp = wheel_client()
        .get(&metadata_url)
        .send()
        .map_err(|e| VvmError::CommandFailed {
            cmd: "wheel metadata lookup".to_string(),
            detail: e.to_string(),
        })?;

    if !resp.status().is_success() {
        return Err(VvmError::CommandFailed {
            cmd: "wheel metadata lookup".to_string(),
            detail: format!(
                "No precompiled wheel found for this commit + {} (HTTP {}).\n\
                 The wheel may not exist for this variant — try: --cuda cu129",
                cuda.as_str(),
                resp.status()
            ),
        });
    }

    let entries: Vec<serde_json::Value> = resp.json().map_err(|e| VvmError::CommandFailed {
        cmd: "wheel metadata lookup".to_string(),
        detail: format!("Unexpected metadata.json format: {}", e),
    })?;

    for entry in &entries {
        let platform = entry["platform_tag"].as_str().unwrap_or("");
        let path = entry["path"].as_str().unwrap_or("");
        if platform.contains(arch) && !path.is_empty() {
            // path is relative to metadata.json: "../../../{commit}/{filename}"
            // Strip leading "../../../" and prepend base URL
            let suffix = path.trim_start_matches("../").trim_start_matches('/');
            return Ok(format!("https://wheels.vllm.ai/{}", suffix));
        }
    }

    Err(VvmError::CommandFailed {
        cmd: "wheel metadata lookup".to_string(),
        detail: format!("No wheel found for {} in {}", arch, metadata_url),
    })
}

/// Install from a GitHub PR by cloning the PR ref and doing an editable install
/// with precompiled C++/CUDA libs from the merge-base on main.
fn install_from_pr(
    venv_path: &std::path::Path,
    specifier: &VersionSpecifier,
    cuda: &CudaVariant,
    version_dir: &std::path::Path,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let pr_num = match specifier {
        VersionSpecifier::Pr(n) => *n,
        _ => unreachable!(),
    };

    let client = wheel_client();

    // A merged PR needs no drift check at all: upstream CI built a wheel at the
    // merge commit, and that wheel's compiled extensions already contain the
    // PR's C++/CUDA changes. Without this, wheel selection below keys off the
    // PR head's timestamp, lands on a main commit from *before* the merge, and
    // then offers a 15-30 min compile to reproduce kernels that were published
    // hours earlier. Only pin it when the wheel actually exists — a merge
    // commit whose build failed falls back to the normal path.
    let merged_wheel_commit = github::pr_merge_commit(pr_num)
        .ok()
        .flatten()
        .filter(|sha| lookup_wheel_url(&client, sha, cuda).is_some());

    if let Some(sha) = &merged_wheel_commit {
        log_info(
            spinner,
            format!(
                "  PR #{} is merged — using its merge commit's wheel ({})",
                pr_num,
                sha[..12.min(sha.len())].cyan()
            ),
        );
    } else if let Ok(files) = github::pr_changed_files(pr_num) {
        // Still open (or merged without a wheel): a PR touching csrc/ means the
        // precompiled wheel won't exercise the PR's changes. On YES, fall through
        // to compile_at_ref below (which clones, checks out the PR ref, and
        // compiles); on NO, install with the precompiled wheel + a warning.
        if has_compiled_changes(&files) {
            let compiled: Vec<String> = files
                .into_iter()
                .filter(|f| has_compiled_changes(std::slice::from_ref(f)))
                .collect();
            let reason = format_drift_reason(&format!("PR #{}", pr_num), &compiled);
            if prompt_compile_fallback(spinner, &reason, OnDecline::InstallPrecompiled) {
                return compile_at_ref(venv_path, specifier, &None, cuda, spinner);
            }
            log_info(
                spinner,
                format!(
                    "  {} precompiled install will not include PR #{} C++/CUDA changes",
                    "Warning:".yellow().bold(),
                    pr_num
                ),
            );
        }
    }

    let src_dir = version_dir.join("src");
    let src_str = src_dir.to_string_lossy().to_string();

    // Clone the vllm repo and fetch the PR ref
    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let git_quiet: &[&str] = if verbose { &[] } else { &["--quiet"] };

    let mut clone_args = vec!["clone", "--depth", "1"];
    clone_args.extend_from_slice(git_quiet);
    clone_args.extend(["https://github.com/vllm-project/vllm.git", &src_str]);
    venv::run_in_venv(venv_path, "git", &clone_args)?;

    let pr_ref = format!("pull/{}/head:pr-{}", pr_num, pr_num);
    let pr_branch = format!("pr-{}", pr_num);
    let mut fetch_args = vec!["-C", &src_str, "fetch", "--depth", "50"];
    fetch_args.extend_from_slice(git_quiet);
    fetch_args.extend(["origin", &pr_ref]);
    venv::run_in_venv(venv_path, "git", &fetch_args)?;

    let mut co_args = vec!["-C", &src_str, "checkout"];
    co_args.extend_from_slice(git_quiet);
    co_args.push(&pr_branch);
    venv::run_in_venv(venv_path, "git", &co_args)?;

    let good_commit = match merged_wheel_commit {
        Some(sha) => sha,
        None => {
            // Use the PR checkout's HEAD timestamp to find a precompiled wheel from
            // roughly the same time period, reducing code divergence between the PR
            // source and the wheel's compiled extensions.
            let until = local_head_timestamp(&src_str);
            match find_good_wheel_commit(&client, cuda, until.as_deref(), None) {
                Some((commit, _url)) => commit,
                None => {
                    let reason = format!(
                        "No precompiled wheel found for PR #{} on {}/{}.",
                        pr_num,
                        std::env::consts::ARCH,
                        cuda.as_str()
                    );
                    if prompt_compile_fallback(spinner, &reason, OnDecline::Abort) {
                        return compile_at_ref(venv_path, specifier, &None, cuda, spinner);
                    }
                    return Err(VvmError::CommandFailed {
                        cmd: "precompiled wheel lookup".to_string(),
                        detail: format!(
                            "No precompiled wheel found for {}/{}.",
                            std::env::consts::ARCH,
                            cuda.as_str()
                        ),
                    });
                }
            }
        }
    };

    align_torch_to_wheel(&client, &src_dir, &good_commit);

    let torch_backend = format!("--torch-backend={}", cuda.as_str());
    let torch_index = torch_index_url(cuda);
    let nightly_index = nightly_index_url(cuda);

    venv::run_in_venv_with_env(
        venv_path,
        "uv",
        &[
            "pip",
            "install",
            "-e",
            &src_str,
            "--extra-index-url",
            &torch_index,
            "--extra-index-url",
            &nightly_index,
            "--index-strategy",
            "unsafe-best-match",
            &torch_backend,
        ],
        &[
            ("VLLM_USE_PRECOMPILED", "1"),
            ("VLLM_PRECOMPILED_WHEEL_VARIANT", cuda.as_str()),
            ("VLLM_PRECOMPILED_WHEEL_COMMIT", &good_commit),
        ],
    )
}

fn install_from_local(
    venv_path: &std::path::Path,
    repo_path: &str,
    cuda: &CudaVariant,
    wheel_commit: Option<&str>,
    wheel_location: Option<&str>,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let torch_backend = format!("--torch-backend={}", cuda.as_str());
    let torch_index = torch_index_url(cuda);
    let wheel_variant_index = nightly_index_url(cuda);

    let args = vec![
        "pip",
        "install",
        "-e",
        repo_path,
        "--extra-index-url",
        &torch_index,
        "--extra-index-url",
        &wheel_variant_index,
        "--index-strategy",
        "unsafe-best-match",
        &torch_backend,
    ];

    // Merge-base with upstream main: an internal fork carries local commits on
    // top of an older main base, and the precompiled wheel must match that BASE
    // (its compiled _C.so), not HEAD. Computed once and reused for both wheel
    // selection and the compiled-drift check below. None on a shallow clone /
    // missing main ref — callers fall back to HEAD's date.
    let base = merge_base_with_main(repo_path);

    // Determine which precompiled wheel commit + URL to use. We track the URL
    // so we can log it (and skip another HTTP call later).
    let (effective_wheel_commit, effective_wheel_url) = if let Some(wc) = wheel_commit {
        // User-pinned commit: look up URL for logging (best-effort; on failure
        // we skip the log but still proceed with the install).
        let url = lookup_wheel_url(&wheel_client(), wc, cuda);
        (Some(wc.to_string()), url)
    } else if wheel_location.is_some() {
        (None, None) // wheel-location skips commit logic
    } else {
        // Auto-find the wheel for the fork's main base. Prefer the exact
        // merge-base wheel; if it has none yet, walk back recent main commits
        // no newer than the BASE's commit date (not HEAD's — HEAD may be months
        // of fork commits ahead, which would grab a far-too-new main wheel).
        spinner.set_message("Finding precompiled wheel...");
        let client = wheel_client();
        let until = base
            .as_deref()
            .and_then(|b| commit_timestamp(repo_path, b))
            .or_else(|| local_head_timestamp(repo_path));
        let found = base
            .as_deref()
            .and_then(|b| lookup_wheel_url(&client, b, cuda).map(|u| (b.to_string(), u)))
            .or_else(|| find_good_wheel_commit(&client, cuda, until.as_deref(), Some(spinner)));
        match found {
            Some((commit, url)) => {
                if let Some(b) = base.as_deref() {
                    if commit == b {
                        log_info(
                            spinner,
                            format!(
                                "  {} matching fork's main base {}",
                                "Wheel base:".blue().bold(),
                                &commit[..12.min(commit.len())]
                            ),
                        );
                    } else {
                        log_info(
                            spinner,
                            format!(
                                "  {} no wheel at main base {}, using nearest older build {}",
                                "Wheel base:".blue().bold(),
                                &b[..12.min(b.len())],
                                &commit[..12.min(commit.len())]
                            ),
                        );
                    }
                }
                (Some(commit), Some(url))
            }
            None => {
                let err = VvmError::CommandFailed {
                    cmd: "precompiled wheel lookup".to_string(),
                    detail: format!(
                        "No precompiled wheel found for {}/{}. Try --wheel-commit nightly or --cuda cu129.",
                        std::env::consts::ARCH,
                        cuda.as_str()
                    ),
                };
                if prompt_compile_fallback(spinner, &err.to_string(), OnDecline::Abort) {
                    return install_from_local_compiled(venv_path, repo_path, cuda, spinner)
                        .map(|_| ());
                }
                return Err(err);
            }
        }
    };

    // Check torch version compatibility: the precompiled _C.abi3.so must match
    // the installed torch. Can't patch user's local repo, so error early.
    if let Some(ref wc) = effective_wheel_commit {
        let client = wheel_client();
        let src_torch =
            parse_torch_version_from_file(&std::path::Path::new(repo_path).join("pyproject.toml"));
        let wheel_torch = fetch_torch_version_from_github(&client, wc);
        if let (Some(ref src_v), Some(ref wheel_v)) = (&src_torch, &wheel_torch) {
            if src_v != wheel_v {
                let err = VvmError::CommandFailed {
                    cmd: "install".to_string(),
                    detail: format!(
                        "Local repo requires torch {} but precompiled wheel (commit {}) \
                         was built with torch {}.\n\
                         The precompiled _C extensions would be ABI-incompatible.\n\
                         Fix: vvm install --path {} --compile",
                        src_v,
                        &wc[..12.min(wc.len())],
                        wheel_v,
                        repo_path
                    ),
                };
                if prompt_compile_fallback(spinner, &err.to_string(), OnDecline::Abort) {
                    return install_from_local_compiled(venv_path, repo_path, cuda, spinner)
                        .map(|_| ());
                }
                return Err(err);
            }
        }
    }

    // Compiled-drift check: if the fork has C++/CUDA changes between the wheel
    // commit and HEAD, the precompiled _C.so won't include them — the user
    // thinks they're testing their kernels but would actually run the wheel's
    // binaries. Diff against the wheel commit when it's reachable locally
    // (catches both upstream drift below the base and the fork's own changes),
    // else against the always-local merge-base. Default to compiling here.
    let drift_from = match effective_wheel_commit.as_deref() {
        Some(wc) if rev_exists(repo_path, wc) => Some(wc.to_string()),
        _ => base.clone(),
    };
    if let Some(from) = drift_from {
        if let Some(changed) = compiled_drift_since(repo_path, &from) {
            let reason = format_drift_reason("Local repo", &changed);
            match prompt_compile_drift(spinner, &reason) {
                DriftChoice::Compile => {
                    return install_from_local_compiled(venv_path, repo_path, cuda, spinner)
                        .map(|_| ());
                }
                DriftChoice::AbortNonInteractive => {
                    return Err(VvmError::CommandFailed {
                        cmd: "install".to_string(),
                        detail: format!(
                            "{}\nRe-run with --compile to build from source, or -y to auto-compile.",
                            reason
                        ),
                    });
                }
                DriftChoice::Precompiled => {
                    log_info(
                        spinner,
                        format!(
                            "  {} precompiled install will not include local C++/CUDA changes",
                            "Warning:".yellow().bold()
                        ),
                    );
                }
            }
        }
    }

    // env_vars holds &str references, so keep the owning Strings/Option in scope.
    let variant_str = cuda.as_str();

    let mut env_vars: Vec<(&str, &str)> = vec![
        ("VLLM_USE_PRECOMPILED", "1"),
        ("VLLM_PRECOMPILED_WHEEL_VARIANT", variant_str),
    ];

    if let Some(ref c) = effective_wheel_commit {
        env_vars.push(("VLLM_PRECOMPILED_WHEEL_COMMIT", c.as_str()));
    }

    if let Some(loc) = wheel_location {
        env_vars.push(("VLLM_PRECOMPILED_WHEEL_LOCATION", loc));
        log_info(spinner, format!("  {} {}", "Wheel:".blue().bold(), loc));
        spinner.set_message(format!("Installing from wheel-location: {}...", loc));
    }

    if let Some(ref c) = effective_wheel_commit {
        if let Some(ref url) = effective_wheel_url {
            log_info(spinner, format!("  {} {}", "Wheel:".blue().bold(), url));
        }
        spinner.set_message(format!(
            "Installing with wheel {}...",
            &c[..12.min(c.len())]
        ));
    }

    venv::run_in_venv_with_env(venv_path, "uv", &args, &env_vars)
}

/// Install from a local vllm repo by compiling C++/CUDA extensions from source.
/// Uses an editable install (`-e`) so that, like the precompiled `--path` mode,
/// edits to the repo's Python files take effect live without reinstalling. The
/// compiled `_C.abi3.so` lands in the source tree; rebuild kernels by re-running
/// `--compile`. (Trade-off: multiple `--compile` installs of the SAME repo path
/// share that one binary, so they can't be compared side-by-side.)
/// Delete every compiled torch-extension binary (`*.abi3.so`) under
/// `<repo>/vllm/`, recursively. Returns how many files were removed.
/// Used before a --compile build so binaries left by previous installs
/// (wheel extractions, builds of other commits) can't outlive the set of
/// extension targets the current commit actually produces.
fn remove_tree_kernel_binaries(repo_path: &str) -> usize {
    fn walk(dir: &std::path::Path, removed: &mut usize) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, removed);
            } else if path
                .file_name()
                .map(|n| n.to_string_lossy().ends_with(".abi3.so"))
                .unwrap_or(false)
                && std::fs::remove_file(&path).is_ok()
            {
                *removed += 1;
            }
        }
    }
    let mut removed = 0;
    walk(&std::path::Path::new(repo_path).join("vllm"), &mut removed);
    removed
}

/// A previous --compile install of the same repo whose in-tree kernels may be
/// reusable for a new --compile install.
struct ReusableKernels {
    name: String,
    commit: String,
    arch_list: String,
}

/// Find the most recent local-compiled install of `repo_path` with the same
/// CUDA variant that recorded a reusable arch list. Returns None if ANY later
/// install used the same tree — a subsequent precompiled --path install
/// extracts wheel binaries over the tree, and a compile for another CUDA
/// variant replaces them, so the tree would no longer hold that build's
/// kernels.
fn find_reusable_compiled(
    config: &Config,
    repo_path: &str,
    cuda: &CudaVariant,
) -> Option<ReusableKernels> {
    let repo_canon = std::fs::canonicalize(repo_path).ok()?;
    let entries = std::fs::read_dir(config.versions_dir()).ok()?;
    let mut best: Option<(String, ReusableKernels)> = None;
    let mut latest_tree_touch: Option<String> = None;
    for entry in entries.flatten() {
        let meta = match VersionMetadata::load(&entry.path()) {
            Ok(m) => m,
            Err(_) => continue, // includes the in-progress install (no metadata yet)
        };
        let Some(src) = &meta.source_path else {
            continue;
        };
        if std::fs::canonicalize(src).ok().as_deref() != Some(repo_canon.as_path()) {
            continue;
        }
        if latest_tree_touch
            .as_ref()
            .map(|at| meta.installed_at > *at)
            .unwrap_or(true)
        {
            latest_tree_touch = Some(meta.installed_at.clone());
        }
        if meta.install_type != "local-compiled" || meta.cuda != cuda.as_str() {
            continue;
        }
        let (Some(commit), Some(arch)) = (&meta.commit, &meta.arch_list) else {
            continue;
        };
        if best
            .as_ref()
            .map(|(at, _)| meta.installed_at > *at)
            .unwrap_or(true)
        {
            best = Some((
                meta.installed_at.clone(),
                ReusableKernels {
                    name: meta.name.clone(),
                    commit: commit.clone(),
                    arch_list: arch.clone(),
                },
            ));
        }
    }
    let (best_at, found) = best?;
    if latest_tree_touch.map(|at| at > best_at).unwrap_or(false) {
        return None;
    }
    Some(found)
}

/// Minimal valid zip archive (bare end-of-central-directory record). Passed as
/// VLLM_PRECOMPILED_WHEEL_LOCATION so vllm's setup.py takes the precompiled
/// path (skipping build_ext) without downloading anything; extracting zero
/// members is a no-op and the kernels already in the source tree are used.
const EMPTY_ZIP: [u8; 22] = [
    0x50, 0x4B, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Try to satisfy a --compile install without compiling, by reusing the
/// kernels a previous --compile install of the same repo left in the source
/// tree (compiled .so files persist there across git checkouts). Safe only
/// when nothing that feeds the compiled extensions changed since that build:
/// csrc/cmake/CMakeLists.txt identical (worktree included), same torch pin,
/// and the old arch list covers the local GPU.
///
/// Returns the kernels' arch list when the reuse install succeeded; None means
/// "not applicable or not safe — do a real compile". Never fails the install:
/// any error here falls back to compilation.
fn try_reuse_compiled_kernels(
    config: &Config,
    venv_path: &std::path::Path,
    repo_path: &str,
    cuda: &CudaVariant,
    spinner: &ProgressBar,
) -> Option<String> {
    if std::env::var("VVM_FORCE_COMPILE").is_ok() {
        return None;
    }
    let prev = find_reusable_compiled(config, repo_path, cuda)?;
    let prev_short = &prev.commit[..12.min(prev.commit.len())];
    if !rev_exists(repo_path, &prev.commit) {
        return None;
    }
    if let Some(changed) = compiled_drift_worktree(repo_path, &prev.commit) {
        log_info(
            spinner,
            format!(
                "  {} C++/CUDA changed since last compile at {} ({} file{}) — compiling",
                "Note:".yellow().bold(),
                prev_short,
                changed.len(),
                if changed.len() == 1 { "" } else { "s" },
            ),
        );
        return None;
    }
    // Torch ABI guard: a different torch pin means the old .so links against
    // the wrong libtorch.
    let torch_now =
        parse_torch_version_from_file(&std::path::Path::new(repo_path).join("pyproject.toml"));
    let torch_then = torch_version_at_commit(repo_path, &prev.commit);
    if let (Some(now), Some(then)) = (&torch_now, &torch_then) {
        if now != then {
            log_info(
                spinner,
                format!(
                    "  {} torch pin changed since last compile ({} → {}) — compiling",
                    "Note:".yellow().bold(),
                    then,
                    now
                ),
            );
            return None;
        }
    }
    // Arch guard: ~/.vvm may be shared across nodes with different GPUs.
    if prev.arch_list != "all" {
        if let Some(local) = crate::cuda::detect_gpu_arch_list() {
            if !crate::cuda::arch_list_covers(&prev.arch_list, &local) {
                log_info(
                    spinner,
                    format!(
                        "  {} kernels were compiled for arch {} but local GPU needs {} — compiling",
                        "Note:".yellow().bold(),
                        prev.arch_list,
                        local
                    ),
                );
                return None;
            }
        }
    }
    // The compiled artifacts must actually still be in the tree. Which
    // extension names exist varies by commit (e.g. _C vs _C_stable_libtorch),
    // so just require at least one compiled extension at the package root.
    let has_kernels = std::fs::read_dir(std::path::Path::new(repo_path).join("vllm"))
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".abi3.so"))
        })
        .unwrap_or(false);
    if !has_kernels {
        return None;
    }

    log_info(
        spinner,
        format!(
            "  {} no C++/CUDA changes since {} (compiled at {}) — reusing kernels (VVM_FORCE_COMPILE=1 to rebuild)",
            "Note:".green().bold(),
            prev.name.cyan(),
            prev_short,
        ),
    );
    spinner.set_message("Installing (reusing compiled kernels)...");

    let version_dir = venv_path.parent().expect("venv must have parent");
    let stub = version_dir.join("kernels-stub.whl");
    if std::fs::write(&stub, EMPTY_ZIP).is_err() {
        return None;
    }

    let torch_backend = format!("--torch-backend={}", cuda.as_str());
    let torch_index = torch_index_url(cuda);
    let wheel_variant_index = nightly_index_url(cuda);
    let args = [
        "pip",
        "install",
        "-e",
        repo_path,
        "--extra-index-url",
        &torch_index,
        "--extra-index-url",
        &wheel_variant_index,
        "--index-strategy",
        "unsafe-best-match",
        &torch_backend,
    ];
    let stub_str = stub.to_string_lossy().to_string();
    let env_vars: Vec<(&str, &str)> = vec![
        ("VLLM_USE_PRECOMPILED", "1"),
        ("VLLM_PRECOMPILED_WHEEL_LOCATION", &stub_str),
    ];
    let result = venv::run_in_venv_with_env(venv_path, "uv", &args, &env_vars);
    let _ = std::fs::remove_file(&stub);
    match result {
        Ok(()) => Some(prev.arch_list),
        Err(e) => {
            log_info(
                spinner,
                format!(
                    "  {} kernel-reuse install failed ({}), falling back to full compile",
                    "Warning:".yellow().bold(),
                    e
                ),
            );
            None
        }
    }
}

/// On success returns the TORCH_CUDA_ARCH_LIST that was in effect for the
/// build ("all" when unrestricted) so callers can record it in metadata.
fn install_from_local_compiled(
    venv_path: &std::path::Path,
    repo_path: &str,
    cuda: &CudaVariant,
    spinner: &ProgressBar,
) -> Result<String, VvmError> {
    let torch_backend = format!("--torch-backend={}", cuda.as_str());

    // Build env vars for faster compilation
    let mut extra_env: Vec<(&str, String)> = Vec::new();

    // TORCH_CUDA_ARCH_LIST: detect local GPU arch to skip irrelevant architectures.
    // vllm defaults to compiling for all supported archs (7.5-12.1), which is very slow.
    //
    // NGC PyTorch containers pre-export a fat list (e.g. "8.0 8.6 9.0 10.0 11.0 12.0+PTX")
    // that swamps the build. When the env list is a strict superset of the local GPU's
    // arch, narrow it. VVM_KEEP_ARCH_LIST=1 disables this for cross-compile use cases.
    let local_archs = crate::cuda::detect_gpu_arch_list();
    let keep_env = std::env::var("VVM_KEEP_ARCH_LIST").is_ok();
    let arch_in_effect: String = match (std::env::var("TORCH_CUDA_ARCH_LIST"), &local_archs) {
        (Err(_), Some(local)) => {
            extra_env.push(("TORCH_CUDA_ARCH_LIST", local.clone()));
            local.clone()
        }
        (Ok(env_val), Some(local)) if !keep_env => {
            if let Some(narrowed) = crate::cuda::narrow_arch_list_if_superset(&env_val, local) {
                log_info(
                    spinner,
                    format!(
                        "  {} narrowing TORCH_CUDA_ARCH_LIST {:?} → {:?} (local GPU: {}; set VVM_KEEP_ARCH_LIST=1 to disable)",
                        "Note:".yellow().bold(),
                        env_val,
                        narrowed,
                        local,
                    ),
                );
                extra_env.push(("TORCH_CUDA_ARCH_LIST", narrowed.clone()));
                narrowed
            } else {
                env_val
            }
        }
        (Ok(env_val), _) => env_val,
        // No env override and no detectable GPU: vllm builds its full arch set.
        (Err(_), None) => "all".to_string(),
    };

    // Don't set MAX_JOBS or NVCC_THREADS — vllm's setup.py defaults to
    // all CPUs (os.sched_getaffinity) with NVCC_THREADS=1, which is optimal.
    // Setting NVCC_THREADS>1 causes setup.py to divide MAX_JOBS by it,
    // and over-parallelizing nvcc can cause memory pressure.

    // Clear stale CMake cache from the repo — previous build isolation runs
    // cache ninja/cmake paths (in build/temp.*/CMakeCache.txt) that point to
    // now-deleted temp venvs.
    let build_dir = std::path::Path::new(repo_path).join("build");
    if build_dir.exists() {
        spinner.set_message("Clearing stale build cache...");
        let _ = std::fs::remove_dir_all(&build_dir);
    }

    // Remove kernel binaries left in the tree by previous installs. A
    // precompiled --path install extracts wheel .so files (built at a
    // different commit) into the tree, and commits can drop or rename
    // extension targets (e.g. _C → _C_stable_libtorch on CUDA), so a stale
    // binary the current build no longer produces would get loaded alongside
    // the fresh one and double-register torch ops at import. The build
    // regenerates exactly the set this commit defines.
    let removed = remove_tree_kernel_binaries(repo_path);
    if removed > 0 {
        log_info(
            spinner,
            format!(
                "  {} removed {} kernel binar{} from previous installs",
                "Note:".blue().bold(),
                removed,
                if removed == 1 { "y" } else { "ies" },
            ),
        );
    }

    // Step 1: Install build dependencies (torch + build-system.requires) into the
    // venv BEFORE compiling. uv's build isolation installs build deps into a temp
    // venv, but cmake can't execute the ninja binary from the temp path (the Python
    // entry-point wrapper fails). By installing deps into the real venv and using
    // --no-build-isolation, cmake finds ninja on PATH and builds correctly.
    spinner.set_message("Installing build dependencies (torch + build tools)...");
    let build_deps = read_build_requires(repo_path);
    let torch_index = torch_index_url(cuda);
    let mut dep_args = vec![
        "pip",
        "install",
        &torch_backend,
        "--extra-index-url",
        &torch_index,
        "--index-strategy",
        "unsafe-best-match",
    ];
    let dep_strs: Vec<String> = build_deps.iter().map(|s| s.to_string()).collect();
    for dep in &dep_strs {
        dep_args.push(dep);
    }
    venv::run_in_venv(venv_path, "uv", &dep_args)?;

    let env_refs: Vec<(&str, &str)> = extra_env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let env_desc: Vec<String> = extra_env
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();
    if !env_desc.is_empty() {
        spinner.set_message(format!(
            "Compiling C++/CUDA extensions from source ({})...",
            env_desc.join(", ")
        ));
    } else {
        spinner.set_message("Compiling C++/CUDA extensions from source...");
    }

    // version_dir is venv_path's parent
    let version_dir = venv_path.parent().expect("venv must have parent");
    let log_path = version_dir.join("build.log");

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    let verbose = std::env::var("VVM_VERBOSE").is_ok();

    // Step 2: Use --no-build-isolation-package vllm (not a global
    // --no-build-isolation) so vllm builds against the venv's CUDA-enabled
    // torch, while transitive sdists still get isolated build envs and honor
    // their own [build-system].requires.
    let mut command = std::process::Command::new("uv");
    let mut uv_args = vec![
        "pip",
        "install",
        "-e",
        repo_path,
        "--no-build-isolation-package",
        "vllm",
        &torch_backend,
    ];
    if verbose {
        uv_args.push("--verbose");
    }
    command
        .args(&uv_args)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", venv_path);
    for (key, value) in &env_refs {
        command.env(key, value);
    }

    if verbose {
        // Verbose: stream to terminal, tee to log file via shell
        command
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = command.spawn().map_err(|e| VvmError::CommandFailed {
            cmd: "uv pip install (compile)".to_string(),
            detail: e.to_string(),
        })?;

        // Spawn threads to read stdout/stderr and write to both terminal and log
        let log_file = std::fs::File::create(&log_path).ok();
        let log_file = std::sync::Arc::new(std::sync::Mutex::new(log_file));

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let log1 = log_file.clone();
        let log2 = log_file.clone();

        let t1 = std::thread::spawn(move || {
            if let Some(mut reader) = stdout {
                let mut buf = [0u8; 4096];
                loop {
                    match std::io::Read::read(&mut reader, &mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let _ = std::io::Write::write_all(&mut std::io::stdout(), &buf[..n]);
                            if let Ok(mut guard) = log1.lock() {
                                if let Some(ref mut f) = *guard {
                                    let _ = std::io::Write::write_all(f, &buf[..n]);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        });

        let t2 = std::thread::spawn(move || {
            if let Some(mut reader) = stderr {
                let mut buf = [0u8; 4096];
                loop {
                    match std::io::Read::read(&mut reader, &mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let _ = std::io::Write::write_all(&mut std::io::stderr(), &buf[..n]);
                            if let Ok(mut guard) = log2.lock() {
                                if let Some(ref mut f) = *guard {
                                    let _ = std::io::Write::write_all(f, &buf[..n]);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        });

        let _ = t1.join();
        let _ = t2.join();

        let status = child.wait().map_err(|e| VvmError::CommandFailed {
            cmd: "uv pip install (compile)".to_string(),
            detail: e.to_string(),
        })?;

        if !status.success() {
            return Err(VvmError::CommandFailed {
                cmd: "uv pip install (compile)".to_string(),
                detail: format!(
                    "exit code: {:?}. Build log: {}",
                    status.code(),
                    log_path.display()
                ),
            });
        }
    } else {
        // Quiet: capture all output, save to log file
        let output = command.output().map_err(|e| VvmError::CommandFailed {
            cmd: "uv pip install (compile)".to_string(),
            detail: e.to_string(),
        })?;

        // Save combined stdout+stderr to build.log
        if let Ok(mut f) = std::fs::File::create(&log_path) {
            let _ = std::io::Write::write_all(&mut f, &output.stdout);
            let _ = std::io::Write::write_all(&mut f, &output.stderr);
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VvmError::CommandFailed {
                cmd: "uv pip install (compile)".to_string(),
                detail: format!("{}\nBuild log: {}", stderr, log_path.display()),
            });
        }
    }

    Ok(arch_in_effect)
}

/// Ensure `<version_dir>/src` is checked out at the requested ref, then
/// run `install_from_local_compiled` on it. Used as a fallback when no
/// usable precompiled wheel exists for `commit:` / `branch:` / `pr:`.
/// Reuses an existing src checkout (left by install_from_pr / install_from_source)
/// so we don't re-clone if the caller already cloned at the right ref.
fn compile_at_ref(
    venv_path: &std::path::Path,
    specifier: &VersionSpecifier,
    commit_hash: &Option<String>,
    cuda: &CudaVariant,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let version_dir = venv_path.parent().expect("venv_path must have a parent");
    let src_dir = version_dir.join("src");
    let src_str = src_dir.to_string_lossy().into_owned();

    if !src_dir.exists() {
        let verbose = std::env::var("VVM_VERBOSE").is_ok();
        let git_quiet: &[&str] = if verbose { &[] } else { &["--quiet"] };

        spinner.set_message("Cloning vllm source for compile fallback...");
        let mut clone_args = vec!["clone", "--depth", "1"];
        clone_args.extend_from_slice(git_quiet);
        clone_args.extend(["https://github.com/vllm-project/vllm.git", &src_str]);
        venv::run_in_venv(venv_path, "git", &clone_args)?;

        match specifier {
            VersionSpecifier::Pr(n) => {
                let pr_ref = format!("pull/{}/head:pr-{}", n, n);
                let pr_branch = format!("pr-{}", n);
                let mut fetch_args = vec!["-C", &src_str, "fetch", "--depth", "50"];
                fetch_args.extend_from_slice(git_quiet);
                fetch_args.extend(["origin", &pr_ref]);
                venv::run_in_venv(venv_path, "git", &fetch_args)?;
                let mut co_args = vec!["-C", &src_str, "checkout"];
                co_args.extend_from_slice(git_quiet);
                co_args.push(&pr_branch);
                venv::run_in_venv(venv_path, "git", &co_args)?;
            }
            VersionSpecifier::Branch(b) => {
                let refspec = format!("{}:{}", b, b);
                let mut fetch_args = vec!["-C", &src_str, "fetch", "--depth", "1"];
                fetch_args.extend_from_slice(git_quiet);
                fetch_args.extend(["origin", &refspec]);
                venv::run_in_venv(venv_path, "git", &fetch_args)?;
                let mut co_args = vec!["-C", &src_str, "checkout"];
                co_args.extend_from_slice(git_quiet);
                co_args.push(b);
                venv::run_in_venv(venv_path, "git", &co_args)?;
            }
            VersionSpecifier::Commit(_) => {
                let hash = commit_hash
                    .as_ref()
                    .expect("commit_hash must be resolved for Commit specifier");
                // Fetch by SHA — works even for commits not on main, because
                // GitHub has uploadpack.allowReachableSHA1InWant=true.
                let mut fetch_args = vec!["-C", &src_str, "fetch", "origin"];
                fetch_args.extend_from_slice(git_quiet);
                fetch_args.push(hash);
                venv::run_in_venv(venv_path, "git", &fetch_args)?;
                let mut co_args = vec!["-C", &src_str, "checkout"];
                co_args.extend_from_slice(git_quiet);
                co_args.push(hash);
                venv::run_in_venv(venv_path, "git", &co_args)?;
            }
            _ => unreachable!("compile_at_ref only supports Commit/Branch/Pr"),
        }
    }

    install_from_local_compiled(venv_path, &src_str, cuda, spinner).map(|_| ())
}

/// Derive a version directory name from a local git repo's state.
/// Derive a directory name from a git URL.
/// `git@github.com:Org/repo.git`   → `Org-repo`
/// `https://github.com/Org/repo`   → `Org-repo`
fn repo_dir_name(url: &str) -> String {
    // Strip trailing .git
    let s = url.strip_suffix(".git").unwrap_or(url);
    // Take the path component after the host — works for both SSH and HTTPS.
    // SSH: git@host:path  →  path
    // HTTPS: https://host/path  →  path
    let path = s
        .rsplit_once(':')
        .map(|(_, p)| p)
        .or_else(|| {
            // HTTPS: strip scheme + host, keep path after first /
            s.split("//")
                .nth(1)
                .and_then(|rest| rest.split_once('/').map(|(_, p)| p))
        })
        .unwrap_or(s);
    path.replace('/', "-")
}

/// For https://github.com/... URLs when `gh` is installed and authenticated,
/// plug gh in as a credential helper so vvm inherits the user's gh login
/// instead of prompting for a username on /dev/tty. Resets existing helpers
/// first — a misconfigured global helper (e.g. stale keychain entry) would
/// otherwise win over ours.
fn git_github_auth_args(url: &str) -> Vec<String> {
    let is_github = url.starts_with("https://github.com/") || url.starts_with("http://github.com/");
    if !is_github {
        return Vec::new();
    }
    let gh_ready = std::process::Command::new("gh")
        .args(["auth", "token"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !gh_ready {
        return Vec::new();
    }
    vec![
        "-c".to_string(),
        "credential.helper=".to_string(),
        "-c".to_string(),
        "credential.helper=!gh auth git-credential".to_string(),
    ]
}

/// Clone a repo if not cached, or fetch updates if already cached.
/// Returns the local path to the repo.
fn clone_or_fetch_repo(config: &Config, url: &str) -> Result<String, VvmError> {
    use std::process::Command;

    let repos_dir = config.repos_dir();
    std::fs::create_dir_all(&repos_dir).map_err(VvmError::Io)?;

    let name = repo_dir_name(url);
    let repo_path = repos_dir.join(&name);

    let auth_args = git_github_auth_args(url);
    // If we handed auth to gh, make git fail fast (no /dev/tty prompt) when
    // gh can't answer — better a clean error than a hung terminal.
    let fail_fast = !auth_args.is_empty();

    if repo_path.join(".git").exists() {
        println!("  {} {} ...", "Fetching".blue(), name.cyan());
        let mut cmd = Command::new("git");
        cmd.args(&auth_args).args([
            "-C",
            &repo_path.to_string_lossy(),
            "fetch",
            "--all",
            "--prune",
        ]);
        if fail_fast {
            cmd.env("GIT_TERMINAL_PROMPT", "0");
        }
        let output = cmd.output().map_err(|e| VvmError::CommandFailed {
            cmd: "git fetch".to_string(),
            detail: e.to_string(),
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VvmError::CommandFailed {
                cmd: "git fetch".to_string(),
                detail: stderr.trim().to_string(),
            });
        }
    } else {
        println!("  {} {} ...", "Cloning".green().bold(), url.cyan());
        let mut cmd = Command::new("git");
        cmd.args(&auth_args)
            .args(["clone", url, &repo_path.to_string_lossy()]);
        if fail_fast {
            cmd.env("GIT_TERMINAL_PROMPT", "0");
        }
        let output = cmd.output().map_err(|e| VvmError::CommandFailed {
            cmd: "git clone".to_string(),
            detail: e.to_string(),
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut detail = stderr.trim().to_string();
            // GitHub URL, gh not available → suggest the fix.
            let is_github =
                url.starts_with("https://github.com/") || url.starts_with("http://github.com/");
            if is_github && auth_args.is_empty() {
                detail.push_str("\n\nHint: to use your `gh` login for git, run: gh auth setup-git");
            }
            return Err(VvmError::CommandFailed {
                cmd: format!("git clone {}", url),
                detail,
            });
        }
    }

    Ok(repo_path.to_string_lossy().to_string())
}

/// Checkout any git ref — branch, tag, or commit SHA. A SHA gives a detached
/// HEAD, which is fine: the ff-only merge below is best-effort and installs
/// pin the resolved commit in metadata either way.
fn checkout_branch(repo: &std::path::Path, branch: &str) -> Result<(), VvmError> {
    use std::process::Command;

    println!("  {} '{}'...", "Checking out".blue(), branch.yellow());

    let output = Command::new("git")
        .args(["-C", &repo.to_string_lossy(), "checkout", branch])
        .output()
        .map_err(|e| VvmError::CommandFailed {
            cmd: "git checkout".to_string(),
            detail: e.to_string(),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(VvmError::CommandFailed {
            cmd: format!("git checkout {}", branch),
            detail: stderr.trim().to_string(),
        });
    }

    // If there's a remote tracking branch, fast-forward to it so we have
    // the latest code after a fetch. Failure is non-fatal (e.g. local-only
    // branches or user's --path repos with uncommitted work).
    let _ = Command::new("git")
        .args([
            "-C",
            &repo.to_string_lossy(),
            "merge",
            "--ff-only",
            &format!("origin/{}", branch),
        ])
        .output();

    Ok(())
}

fn derive_local_dir_name(
    repo: &std::path::Path,
    name_override: Option<&str>,
) -> Result<(String, String), VvmError> {
    use std::process::Command;

    // Get current commit hash
    let output = Command::new("git")
        .args(["-C", &repo.to_string_lossy(), "rev-parse", "HEAD"])
        .output()
        .map_err(|e| VvmError::CommandFailed {
            cmd: "git rev-parse HEAD".to_string(),
            detail: e.to_string(),
        })?;
    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if commit.is_empty() {
        return Err(VvmError::CommandFailed {
            cmd: "git rev-parse HEAD".to_string(),
            detail: "no commit found".to_string(),
        });
    }

    let short = &commit[..8.min(commit.len())];

    if let Some(name) = name_override {
        // User provided a custom name
        return Ok((name.to_string(), commit));
    }

    // Get current branch name
    let branch_output = Command::new("git")
        .args([
            "-C",
            &repo.to_string_lossy(),
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s == "HEAD" || s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

    let dir_name = if let Some(branch) = branch_output {
        format!("local-{}-{}", branch.replace('/', "-"), short)
    } else {
        format!("local-{}", short)
    };

    Ok((dir_name, commit))
}

fn install_from_source(
    venv_path: &std::path::Path,
    specifier: &VersionSpecifier,
    cuda: &CudaVariant,
    commit_hash: &Option<String>,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    // Clone vllm repo into a temp location inside the version dir
    let src_dir = venv_path
        .parent()
        .expect("venv_path must have a parent")
        .join("src");
    let src_str = src_dir.to_string_lossy().into_owned();

    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let git_quiet: &[&str] = if verbose { &[] } else { &["--quiet"] };

    let mut clone_args = vec!["clone", "--depth", "1"];
    clone_args.extend_from_slice(git_quiet);

    match specifier {
        VersionSpecifier::Release(version) => {
            let tag = format!("v{}", version);
            clone_args.extend(["--branch", &tag]);
            let clone_args_str: Vec<&str> = clone_args
                .iter()
                .copied()
                .chain(["https://github.com/vllm-project/vllm.git", &src_str])
                .collect();
            venv::run_in_venv(venv_path, "git", &clone_args_str)?;
        }
        VersionSpecifier::Branch(branch) => {
            clone_args.extend(["--branch", branch.as_str()]);
            let clone_args_str: Vec<&str> = clone_args
                .iter()
                .copied()
                .chain(["https://github.com/vllm-project/vllm.git", &src_str])
                .collect();
            venv::run_in_venv(venv_path, "git", &clone_args_str)?;
        }
        _ => {
            // Clone default branch, then checkout specific commit
            let clone_args_str: Vec<&str> = clone_args
                .iter()
                .copied()
                .chain(["https://github.com/vllm-project/vllm.git", &src_str])
                .collect();
            venv::run_in_venv(venv_path, "git", &clone_args_str)?;

            if let Some(hash) = commit_hash {
                // Fetch the requested commit by SHA. Works for commits on any
                // ref — including PR branches and forks not reachable from
                // main — because GitHub has uploadpack.allowReachableSHA1InWant=true.
                // (`git fetch --unshallow` only deepens the default branch's
                // history and would fail to find off-main commits.)
                let mut fetch_args = vec!["-C", &src_str, "fetch", "origin"];
                fetch_args.extend_from_slice(git_quiet);
                fetch_args.push(hash);
                venv::run_in_venv(venv_path, "git", &fetch_args)?;
                let mut co_args = vec!["-C", &src_str, "checkout"];
                co_args.extend_from_slice(git_quiet);
                co_args.push(hash);
                venv::run_in_venv(venv_path, "git", &co_args)?;
            }
        }
    }

    // Find a precompiled wheel matching the checkout's age and patch torch
    // versions if needed (same logic as PR installs).
    let until = local_head_timestamp(&src_dir.to_string_lossy());
    let client = wheel_client();
    let (good_commit, _url) = match find_good_wheel_commit(&client, cuda, until.as_deref(), None) {
        Some(v) => v,
        None => {
            let reason = format!(
                "No precompiled wheel found for {}/{}.",
                std::env::consts::ARCH,
                cuda.as_str()
            );
            if prompt_compile_fallback(spinner, &reason, OnDecline::Abort) {
                return compile_at_ref(venv_path, specifier, commit_hash, cuda, spinner);
            }
            return Err(VvmError::CommandFailed {
                cmd: "precompiled wheel lookup".to_string(),
                detail: format!(
                    "No precompiled wheel found for {}/{}.",
                    std::env::consts::ARCH,
                    cuda.as_str()
                ),
            });
        }
    };

    align_torch_to_wheel(&client, &src_dir, &good_commit);

    // Compiled-drift check for commit:/branch: requests: the diff between the
    // wheel commit we actually installed and the requested ref may contain
    // upstream C++/CUDA changes. Note: the requested commit/branch itself may
    // be clean — the drift is from in-range upstream commits, which we list
    // so the user can verify it's not their own ref's changes.
    if let Some(req) = commit_hash.as_ref().filter(|h| *h != &good_commit) {
        match github::compare_changed_files(&good_commit, req) {
            Ok(files) if has_compiled_changes(&files) => {
                let compiled: Vec<String> = files
                    .into_iter()
                    .filter(|f| has_compiled_changes(std::slice::from_ref(f)))
                    .collect();
                let requested = match specifier {
                    VersionSpecifier::Branch(b) => {
                        format!("branch '{}' HEAD ({})", b, &req[..12.min(req.len())])
                    }
                    _ => format!("commit {}", &req[..12.min(req.len())]),
                };
                let responsible = github::commits_in_range_touching(&good_commit, req, |f| {
                    has_compiled_changes(std::slice::from_ref(&f.to_string()))
                })
                .unwrap_or_default();
                let reason =
                    format_range_drift_reason(&requested, &good_commit, &compiled, &responsible);
                // For `commit:X` the user asked for a precise ref — silently
                // installing a nearby wheel is silently-wrong, so declining
                // aborts. Branch still falls through (older wheel + warning)
                // since "approximately the branch" is a reasonable best-effort.
                let on_decline = if matches!(specifier, VersionSpecifier::Commit(_)) {
                    OnDecline::Abort
                } else {
                    OnDecline::InstallPrecompiled
                };
                if prompt_compile_fallback(spinner, &reason, on_decline) {
                    return compile_at_ref(venv_path, specifier, commit_hash, cuda, spinner);
                }
                if matches!(specifier, VersionSpecifier::Commit(_)) {
                    return Err(VvmError::CommandFailed {
                        cmd: "install".to_string(),
                        detail: format!(
                            "C++/CUDA build files differ between the nearest wheel commit ({}) \
                             and {}; declined to compile. Re-run with `-y` or `--compile` to \
                             build from source.",
                            &good_commit[..12.min(good_commit.len())],
                            requested
                        ),
                    });
                }
                log_info(
                    spinner,
                    format!(
                        "  {} precompiled install will not include upstream C++/CUDA changes between wheel commit {} and {}",
                        "Warning:".yellow().bold(),
                        &good_commit[..12.min(good_commit.len())],
                        requested
                    ),
                );
            }
            _ => {}
        }
    }

    // Install with precompiled wheels
    let torch_backend = format!("--torch-backend={}", cuda.as_str());
    let torch_index = torch_index_url(cuda);
    let nightly_index = nightly_index_url(cuda);

    venv::run_in_venv_with_env(
        venv_path,
        "uv",
        &[
            "pip",
            "install",
            "-e",
            &src_str,
            "--extra-index-url",
            &torch_index,
            "--extra-index-url",
            &nightly_index,
            "--index-strategy",
            "unsafe-best-match",
            &torch_backend,
        ],
        &[
            ("VLLM_USE_PRECOMPILED", "1"),
            ("VLLM_PRECOMPILED_WHEEL_VARIANT", cuda.as_str()),
            ("VLLM_PRECOMPILED_WHEEL_COMMIT", &good_commit),
        ],
    )?;

    Ok(())
}

fn release_index_url(version: &str, cuda: &CudaVariant) -> String {
    // wheels.vllm.ai hosts release wheels with CUDA variant subdirs
    format!("https://wheels.vllm.ai/{}/{}", version, cuda.as_str())
}

/// Find the exact wheel URL from a wheels.vllm.ai release index page (PEP 503
/// simple HTML). Returns the full URL so we can install it directly, bypassing
/// uv index resolution and preventing silent fallback to PyPI.
fn find_release_wheel_url(
    index_url: &str,
    version: &str,
    cuda: &CudaVariant,
) -> Result<String, VvmError> {
    let arch = std::env::consts::ARCH; // "x86_64" or "aarch64"
    let base_url = format!("{}/vllm/", index_url.trim_end_matches('/'));

    let resp = wheel_client()
        .get(&base_url)
        .send()
        .map_err(|e| VvmError::CommandFailed {
            cmd: "release wheel lookup".to_string(),
            detail: e.to_string(),
        })?;

    if !resp.status().is_success() {
        return Err(VvmError::CommandFailed {
            cmd: "release wheel lookup".to_string(),
            detail: format!(
                "No vllm {} wheel found for CUDA variant {} (HTTP {})\n\
                 Check: {}\n\
                 Try a different variant with: vvm install {} --cuda cu129",
                version,
                cuda.as_str(),
                resp.status(),
                base_url,
                version,
            ),
        });
    }

    let body = resp.text().unwrap_or_default();

    // Parse href="..." links from the PEP 503 simple index HTML, match arch.
    // Wheel filenames use manylinux tags (e.g. manylinux_2_31_x86_64), so
    // match on raw arch string, consistent with find_wheel_url().
    for line in body.lines() {
        if let Some(start) = line.find("href=\"") {
            let rest = &line[start + 6..];
            if let Some(end) = rest.find('"') {
                let href = &rest[..end];
                if href.ends_with(".whl") && href.contains(arch) {
                    // hrefs are relative (e.g. "../../{commit}/vllm-...whl")
                    let url = if href.starts_with("http") {
                        href.to_string()
                    } else {
                        let suffix = href.trim_start_matches("../").trim_start_matches('/');
                        format!("https://wheels.vllm.ai/{}", suffix)
                    };
                    return Ok(url);
                }
            }
        }
    }

    Err(VvmError::CommandFailed {
        cmd: "release wheel lookup".to_string(),
        detail: format!(
            "No vllm {} wheel found for {}/{} at {}\n\
             The release may not have a prebuilt wheel for this variant yet.\n\
             Try a different variant with: vvm install {} --cuda cu129",
            version,
            cuda.as_str(),
            arch,
            base_url,
            version,
        ),
    })
}

fn nightly_index_url(cuda: &CudaVariant) -> String {
    format!("https://wheels.vllm.ai/nightly/{}", cuda.as_str())
}

fn commit_index_url(commit: &str, cuda: &CudaVariant) -> String {
    if commit == "nightly" {
        return nightly_index_url(cuda);
    }
    match cuda {
        CudaVariant::Cu129 => format!("https://wheels.vllm.ai/{}", commit),
        other => format!("https://wheels.vllm.ai/{}/{}", commit, other.as_str()),
    }
}

/// Read [build-system].requires from a repo's pyproject.toml using Python's
/// built-in tomllib (3.11+). Falls back to a sensible default if parsing fails.
fn read_build_requires(repo_path: &str) -> Vec<String> {
    let pyproject = std::path::Path::new(repo_path).join("pyproject.toml");
    let script = format!(
        "import tomllib, json, sys; \
         d = tomllib.load(open('{}', 'rb')); \
         print(json.dumps(d.get('build-system', {{}}).get('requires', [])))",
        pyproject.display()
    );
    if let Ok(output) = std::process::Command::new("python3")
        .args(["-c", &script])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(deps) = serde_json::from_str::<Vec<String>>(stdout.trim()) {
                if !deps.is_empty() {
                    return deps;
                }
            }
        }
    }
    // Fallback defaults matching vllm's typical build-system.requires
    vec![
        "cmake>=3.26".to_string(),
        "ninja".to_string(),
        "packaging".to_string(),
        "setuptools>=77".to_string(),
        "setuptools-scm>=8".to_string(),
        "torch".to_string(),
        "wheel".to_string(),
        "jinja2".to_string(),
    ]
}

fn torch_index_url(cuda: &CudaVariant) -> String {
    format!("https://download.pytorch.org/whl/{}", cuda.as_str())
}

/// Look up the arch-matched wheel URL at a given commit (or "nightly").
/// Returns None if metadata.json is absent, malformed, or has no entry for
/// the current architecture. Stricter than a HEAD check — a commit whose
/// metadata.json only lists x86_64 wheels returns None on aarch64.
fn lookup_wheel_url(
    client: &reqwest::blocking::Client,
    commit: &str,
    cuda: &CudaVariant,
) -> Option<String> {
    let arch = std::env::consts::ARCH;
    let index_url = commit_index_url(commit, cuda);
    let metadata_url = format!("{}/vllm/metadata.json", index_url.trim_end_matches('/'));
    let resp = client.get(&metadata_url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let entries: Vec<serde_json::Value> = resp.json().ok()?;
    for entry in &entries {
        let platform = entry["platform_tag"].as_str().unwrap_or("");
        let path = entry["path"].as_str().unwrap_or("");
        if platform.contains(arch) && !path.is_empty() {
            let suffix = path.trim_start_matches("../").trim_start_matches('/');
            return Some(format!("https://wheels.vllm.ai/{}", suffix));
        }
    }
    None
}

/// Resolve what commit the nightly wheel index currently points to.
/// Parses the 40-char hex segment from the wheel `path` in metadata.json
/// (e.g. `../../39d5fa96a7c687f9ed7e14a5a52064965356cede/...whl`).
fn resolve_nightly_commit(
    client: &reqwest::blocking::Client,
    cuda: &CudaVariant,
) -> Option<String> {
    let arch = std::env::consts::ARCH;
    let index_url = commit_index_url("nightly", cuda);
    let metadata_url = format!("{}/vllm/metadata.json", index_url.trim_end_matches('/'));
    let resp = client.get(&metadata_url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let entries: Vec<serde_json::Value> = resp.json().ok()?;
    for entry in &entries {
        let platform = entry["platform_tag"].as_str().unwrap_or("");
        let path = entry["path"].as_str().unwrap_or("");
        if platform.contains(arch) && !path.is_empty() {
            for seg in path.split('/') {
                if seg.len() == 40 && seg.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Some(seg.to_string());
                }
            }
        }
    }
    None
}

/// Print a persistent info line that shows up both under the spinner (non-verbose)
/// and in verbose mode where the spinner is hidden (plain eprintln).
fn log_info(spinner: &ProgressBar, msg: String) {
    if std::env::var("VVM_VERBOSE").is_ok() {
        eprintln!("{}", msg);
    } else {
        spinner.println(msg);
    }
}

/// What declining the compile prompt does at a given call site, so the
/// [y/N] line can say it. "n" quietly meaning "install the wheel anyway"
/// at one site and "abort" at another is how a user ends up watching an
/// install they thought they had just cancelled.
#[derive(Clone, Copy)]
enum OnDecline {
    /// Proceed with the precompiled wheel (a warning explains what it lacks).
    InstallPrecompiled,
    /// Fail the install.
    Abort,
}

impl OnDecline {
    fn hint(self) -> &'static str {
        match self {
            OnDecline::InstallPrecompiled => "install precompiled wheel anyway",
            OnDecline::Abort => "abort",
        }
    }
}

/// Interactive prompt offering to retry with `--compile` when no usable
/// precompiled wheel is available. Returns false when stdin isn't a TTY
/// so non-interactive runs keep failing loudly.
fn prompt_compile_fallback(spinner: &ProgressBar, reason: &str, on_decline: OnDecline) -> bool {
    use std::io::{self, BufRead, IsTerminal, Write};
    if AUTO_YES.load(Ordering::Relaxed) {
        spinner.suspend(|| {
            eprintln!();
            eprintln!("{}", reason.yellow());
            eprintln!(
                "{} -y given, proceeding with source compilation (this may take 15-30 min).",
                "auto:".cyan().bold()
            );
        });
        return true;
    }
    if !io::stdin().is_terminal() {
        return false;
    }
    spinner.suspend(|| {
        eprintln!();
        eprintln!("{}", reason.yellow());
        eprint!(
            "{} Compile from source? This may take 15-30 min. [y = compile / N = {}] ",
            "?".cyan().bold(),
            on_decline.hint()
        );
        let _ = io::stderr().flush();
        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line).is_err() {
            return false;
        }
        matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
    })
}

/// What to do when a local `--path` repo has C++/CUDA changes the precompiled
/// wheel can't include. Compiling is the recommended default here (unlike the
/// generic no-wheel fallback), since the whole point of a fork checkout is to
/// run its own kernels.
enum DriftChoice {
    /// Compile from source (`-y`, or the interactive default).
    Compile,
    /// Install the precompiled wheel anyway (user explicitly declined compile).
    Precompiled,
    /// Non-interactive and no `-y`: refuse to silently install a wrong wheel
    /// or to auto-start a 15-30 min build without consent.
    AbortNonInteractive,
}

/// Prompt for the local-repo C++/CUDA drift case, defaulting to compile.
/// `-y` compiles; on a TTY the prompt defaults to Yes (Enter compiles);
/// non-interactive without `-y` returns `AbortNonInteractive` so the caller
/// can error out instead of installing a precompiled wheel missing the changes.
fn prompt_compile_drift(spinner: &ProgressBar, reason: &str) -> DriftChoice {
    use std::io::{self, BufRead, IsTerminal, Write};
    if AUTO_YES.load(Ordering::Relaxed) {
        spinner.suspend(|| {
            eprintln!();
            eprintln!("{}", reason.yellow());
            eprintln!(
                "{} -y given, compiling from source (this may take 15-30 min).",
                "auto:".cyan().bold()
            );
        });
        return DriftChoice::Compile;
    }
    if !io::stdin().is_terminal() {
        return DriftChoice::AbortNonInteractive;
    }
    spinner.suspend(|| {
        eprintln!();
        eprintln!("{}", reason.yellow());
        eprint!(
            "{} Compile from source? This may take 15-30 min. [Y/n] ",
            "?".cyan().bold()
        );
        let _ = io::stderr().flush();
        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line).is_err() {
            // EOF/error on a TTY — fall back to the recommended default.
            return DriftChoice::Compile;
        }
        match line.trim().to_lowercase().as_str() {
            "n" | "no" => DriftChoice::Precompiled,
            _ => DriftChoice::Compile, // empty (Enter) and "y"/"yes" both compile
        }
    })
}

/// Files whose modification means the precompiled `_C.abi3.so` won't reflect
/// the user's/PR's/commit's changes — running against such a wheel is
/// silently-wrong, not just suboptimal. These paths cover the build roots
/// that actually affect the compiled extension.
fn has_compiled_changes(files: &[String]) -> bool {
    files
        .iter()
        .any(|f| f.starts_with("csrc/") || f.starts_with("cmake/") || f == "CMakeLists.txt")
}

/// Detect C++/CUDA build-file changes between `from_sha` and HEAD in a local
/// git repo. `from_sha` is the commit the precompiled wheel was built from
/// (the fork's main base, or the nearest older commit with a wheel) — any
/// csrc/cmake change since then means the borrowed `_C.so` won't reflect it.
/// Returns None if git can't answer (e.g. `from_sha` absent in a shallow
/// clone) or there are no such changes, so callers fail silent rather than
/// raising a false alarm.
fn compiled_drift_since(repo_path: &str, from_sha: &str) -> Option<Vec<String>> {
    let diff = std::process::Command::new("git")
        .args([
            "-C",
            repo_path,
            "diff",
            "--name-only",
            &format!("{}..HEAD", from_sha),
            "--",
            "csrc",
            "cmake",
            "CMakeLists.txt",
        ])
        .output()
        .ok()?;
    if !diff.status.success() {
        return None;
    }
    let files: Vec<String> = String::from_utf8_lossy(&diff.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

/// Like `compiled_drift_since`, but diffs `from` against the WORKING TREE so
/// uncommitted csrc/cmake edits count too (kernel-reuse must not trust a dirty
/// tree). Same None-on-git-failure bias — callers should guard with
/// `rev_exists` first when a false "no drift" would be unsafe.
fn compiled_drift_worktree(repo_path: &str, from: &str) -> Option<Vec<String>> {
    let diff = std::process::Command::new("git")
        .args([
            "-C",
            repo_path,
            "diff",
            "--name-only",
            from,
            "--",
            "csrc",
            "cmake",
            "CMakeLists.txt",
        ])
        .output()
        .ok()?;
    if !diff.status.success() {
        return None;
    }
    let files: Vec<String> = String::from_utf8_lossy(&diff.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

/// Torch requirement in pyproject.toml as of `commit` (via `git show`), for
/// the kernel-reuse ABI guard.
fn torch_version_at_commit(repo_path: &str, commit: &str) -> Option<String> {
    let out = std::process::Command::new("git")
        .args([
            "-C",
            repo_path,
            "show",
            &format!("{}:pyproject.toml", commit),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_torch_version(&String::from_utf8_lossy(&out.stdout))
}

/// Compose a human-readable reason string for the compile-fallback prompt,
/// listing up to 8 changed files and indicating when truncated. Used for
/// PR / local-repo drift where the changes belong to the requested ref itself.
fn format_drift_reason(context: &str, files: &[String]) -> String {
    let mut lines = vec![format!(
        "{} modifies C++/CUDA build files — the precompiled wheel won't \
         reflect these changes at runtime:",
        context
    )];
    for f in files.iter().take(8) {
        lines.push(format!("  - {}", f));
    }
    if files.len() > 8 {
        lines.push(format!("  ...and {} more", files.len() - 8));
    }
    lines.join("\n")
}

/// Compose a reason string for range-diff drift (branch fallback, source-mode
/// commit install): the requested ref itself may be clean, but in-range
/// upstream commits between the wheel commit and the requested ref modified
/// C++/CUDA build files. List the responsible commits so the user can tell
/// it's not their requested ref's changes that are flagged.
fn format_range_drift_reason(
    requested: &str,
    wheel_commit: &str,
    files: &[String],
    responsible: &[(String, String)],
) -> String {
    let mut lines = vec![format!(
        "C++/CUDA build files differ between the precompiled wheel commit ({}) \
         and {} — the wheel won't reflect these changes at runtime.",
        &wheel_commit[..12.min(wheel_commit.len())],
        requested
    )];
    if !responsible.is_empty() {
        lines
            .push("Upstream commits in this range that modified C++/CUDA build files:".to_string());
        for (sha, subject) in responsible.iter().take(6) {
            let short = &sha[..12.min(sha.len())];
            let trimmed: String = subject.chars().take(72).collect();
            lines.push(format!("  - {} {}", short, trimmed));
        }
        if responsible.len() > 6 {
            lines.push(format!("  ...and {} more", responsible.len() - 6));
        }
    }
    lines.push("Affected files:".to_string());
    for f in files.iter().take(8) {
        lines.push(format!("  - {}", f));
    }
    if files.len() > 8 {
        lines.push(format!("  ...and {} more", files.len() - 8));
    }
    lines.join("\n")
}

/// Build a lightweight HTTP client for wheel server checks (short timeout,
/// connection reuse across the 30-commit fallback loop).
fn wheel_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("vvm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build HTTP client")
}

/// Find a recent main branch commit that has a precompiled wheel for the
/// current arch. Returns (commit_sha, wheel_url). If `until` is given
/// (ISO 8601), only considers commits at or before that time.
/// `spinner` receives per-commit progress when present.
fn find_good_wheel_commit(
    client: &reqwest::blocking::Client,
    cuda: &CudaVariant,
    until: Option<&str>,
    spinner: Option<&ProgressBar>,
) -> Option<(String, String)> {
    let commits = github::recent_main_commits_until(until, 30).ok()?;
    let total = commits.len();
    for (i, commit) in commits.iter().enumerate() {
        if let Some(sp) = spinner {
            sp.set_message(format!(
                "Searching precompiled wheel ({}/{}): {}...",
                i + 1,
                total,
                &commit[..8.min(commit.len())]
            ));
        }
        if let Some(url) = lookup_wheel_url(client, commit, cuda) {
            return Some((commit.clone(), url));
        }
    }
    None
}

/// Get the committer date of HEAD in the given git repo (ISO 8601).
fn local_head_timestamp(repo_path: &str) -> Option<String> {
    commit_timestamp(repo_path, "HEAD")
}

/// Committer date (ISO 8601) of an arbitrary revision in the given git repo.
/// Returns None if git can't resolve the rev (e.g. not present in a shallow clone).
fn commit_timestamp(repo_path: &str, rev: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", repo_path, "log", "-1", "--format=%cI", rev])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let ts = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if ts.is_empty() {
        None
    } else {
        Some(ts)
    }
}

/// Candidate refs that may point at upstream vllm `main`, in priority order.
/// An internal fork usually keeps `origin/main` synced to upstream; some users
/// add upstream as a separate remote, and a plain checkout has a local `main`.
const MAIN_BASE_REFS: &[&str] = &["upstream/main", "origin/main", "main"];

/// Find the merge-base between the local repo's HEAD and upstream `main`.
/// Tries `MAIN_BASE_REFS` in order and returns the first that resolves —
/// this is the commit on main the fork diverged from, whose precompiled wheel
/// matches the fork's C++/CUDA source. Returns None if git can't answer
/// (no matching ref, shallow clone missing the base, etc.) so callers can
/// fall back gracefully rather than scaring the user with a false alarm.
fn merge_base_with_main(repo_path: &str) -> Option<String> {
    for base_ref in MAIN_BASE_REFS {
        let output = std::process::Command::new("git")
            .args(["-C", repo_path, "merge-base", "HEAD", base_ref])
            .output()
            .ok()?;
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !sha.is_empty() {
                return Some(sha);
            }
        }
    }
    None
}

/// Check whether a revision exists in the local repo's object database.
/// Used to decide whether a wheel commit can serve as a local diff base.
fn rev_exists(repo_path: &str, rev: &str) -> bool {
    std::process::Command::new("git")
        .args([
            "-C",
            repo_path,
            "cat-file",
            "-e",
            &format!("{}^{{commit}}", rev),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Extract the torch version requirement (e.g. "2.11.0") from a pyproject.toml
/// file on disk. Looks for lines like `"torch == 2.11.0"` or `"torch==2.11.0"`.
/// If the source tree's torch requirement differs from the precompiled wheel's,
/// patch all torch-related version pins (torch, torchaudio, torchvision) in the
/// source to match the wheel. The precompiled _C.abi3.so must load against the
/// same torch it was compiled with; the source code itself is almost always
/// compatible with the wheel's torch version.
fn align_torch_to_wheel(
    client: &reqwest::blocking::Client,
    src_dir: &std::path::Path,
    wheel_commit: &str,
) {
    let src_torch = parse_torch_version_from_file(&src_dir.join("pyproject.toml"));
    let wheel_torch = fetch_torch_version_from_github(client, wheel_commit);
    if let (Some(ref src_v), Some(ref wheel_v)) = (&src_torch, &wheel_torch) {
        if src_v != wheel_v {
            eprintln!(
                "  {} torch {} (precompiled wheel) != {} (source) — patching",
                "Pinning".yellow(),
                wheel_v.cyan(),
                src_v.cyan(),
            );
            patch_torch_versions(client, src_dir, wheel_commit);
        }
    }
}

fn parse_torch_version_from_file(path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_torch_version(&content)
}

/// Fetch pyproject.toml from GitHub at a specific commit and extract the torch
/// version requirement.
fn fetch_torch_version_from_github(
    client: &reqwest::blocking::Client,
    commit: &str,
) -> Option<String> {
    let url = format!(
        "https://raw.githubusercontent.com/vllm-project/vllm/{}/pyproject.toml",
        commit
    );
    let resp = client.get(&url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let content = resp.text().ok()?;
    parse_torch_version(&content)
}

/// Parse torch version from pyproject.toml content. Matches patterns like:
/// `"torch == 2.11.0"`, `"torch==2.10.0"`, `"torch >= 2.11.0"`.
/// Only matches bare "torch", not "torchaudio" or "torchvision".
fn parse_torch_version(content: &str) -> Option<String> {
    parse_pkg_version(content, "torch")
}

/// Patch torch-related version pins across the source tree to match the
/// precompiled wheel's requirements. Patches pyproject.toml, requirements/cuda.txt,
/// and requirements/build.txt for torch, torchaudio, and torchvision.
fn patch_torch_versions(
    client: &reqwest::blocking::Client,
    src_dir: &std::path::Path,
    wheel_commit: &str,
) {
    // Fetch the wheel commit's cuda.txt to get the exact versions it was built with
    let url = format!(
        "https://raw.githubusercontent.com/vllm-project/vllm/{}/requirements/cuda.txt",
        wheel_commit
    );
    let wheel_cuda_txt = match client.get(&url).send().ok().and_then(|r| r.text().ok()) {
        Some(t) => t,
        None => return,
    };

    // Read PR's cuda.txt once before patching
    let pr_cuda_txt = match std::fs::read_to_string(src_dir.join("requirements/cuda.txt")) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Collect all replacements first, then apply to each file
    let packages = ["torch", "torchaudio", "torchvision"];
    let mut replacements: Vec<(String, String)> = Vec::new();

    for pkg in &packages {
        let wheel_ver = parse_pkg_version(&wheel_cuda_txt, pkg);
        let pr_ver = parse_pkg_version(&pr_cuda_txt, pkg);

        if let (Some(wv), Some(pv)) = (wheel_ver, pr_ver) {
            if wv != pv {
                replacements.push((format!("{}=={}", pkg, pv), format!("{}=={}", pkg, wv)));
                replacements.push((format!("{} == {}", pkg, pv), format!("{} == {}", pkg, wv)));
            }
        }
    }

    if replacements.is_empty() {
        return;
    }

    let files = [
        src_dir.join("pyproject.toml"),
        src_dir.join("requirements/cuda.txt"),
        src_dir.join("requirements/build.txt"),
    ];
    for file in &files {
        if let Ok(content) = std::fs::read_to_string(file) {
            let mut patched = content;
            for (old, new) in &replacements {
                patched = patched.replace(old.as_str(), new.as_str());
            }
            let _ = std::fs::write(file, patched);
        }
    }
}

/// Extract a package version pin from requirements text (e.g. "torch==2.10.0").
/// Matches exact package name only — "torch" won't match "torchaudio".
fn parse_pkg_version(content: &str, package: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line
            .trim()
            .trim_matches(|c: char| c == '"' || c == ',' || c == '\'' || c.is_whitespace());
        // Match exact package name followed by operator (not prefix match)
        if let Some(rest) = trimmed.strip_prefix(package) {
            let rest = rest.trim_start();
            // Must start with an operator, not a letter (rejects "torchaudio" when package="torch")
            for op in ["==", ">=", "~="] {
                if let Some(ver) = rest.strip_prefix(op) {
                    let ver = ver.trim_start();
                    // Take version up to first whitespace or comment
                    let ver = ver.split_whitespace().next().unwrap_or(ver);
                    return Some(ver.to_string());
                }
            }
        }
    }
    None
}

fn build_metadata(
    specifier: &VersionSpecifier,
    dir_name: &str,
    cuda: &CudaVariant,
    commit_hash: &Option<String>,
) -> VersionMetadata {
    let now = chrono::Utc::now().to_rfc3339();
    match specifier {
        VersionSpecifier::Release(v) => VersionMetadata {
            install_type: "release".to_string(),
            name: dir_name.to_string(),
            version: Some(v.clone()),
            commit: commit_hash.clone(),
            pr: None,
            branch: None,
            source_path: None,
            cuda: cuda.to_string(),
            installed_at: now,
            arch_list: None,
        },
        VersionSpecifier::Commit(_) => VersionMetadata {
            install_type: "commit".to_string(),
            name: dir_name.to_string(),
            version: None,
            commit: commit_hash.clone(), // use the resolved full 40-char SHA, not the raw user input
            pr: None,
            branch: None,
            source_path: None,
            cuda: cuda.to_string(),
            installed_at: now,
            arch_list: None,
        },
        VersionSpecifier::Branch(b) => VersionMetadata {
            install_type: "branch".to_string(),
            name: dir_name.to_string(),
            version: None,
            commit: commit_hash.clone(),
            pr: None,
            branch: Some(b.clone()),
            source_path: None,
            cuda: cuda.to_string(),
            installed_at: now,
            arch_list: None,
        },
        VersionSpecifier::Pr(n) => VersionMetadata {
            install_type: "pr".to_string(),
            name: dir_name.to_string(),
            version: None,
            commit: commit_hash.clone(),
            pr: Some(*n),
            branch: None,
            source_path: None,
            cuda: cuda.to_string(),
            installed_at: now,
            arch_list: None,
        },
        VersionSpecifier::Nightly => VersionMetadata {
            install_type: "nightly".to_string(),
            name: dir_name.to_string(),
            version: None,
            commit: commit_hash.clone(),
            pr: None,
            branch: None,
            source_path: None,
            cuda: cuda.to_string(),
            installed_at: now,
            arch_list: None,
        },
    }
}

/// Read `vllm.__version__` without the C extension check. Used to backfill
/// missing metadata.version on auto_use for installs predating verify-on-nightly.
fn quick_capture_version(venv_path: &std::path::Path) -> Option<String> {
    let python = venv_path.join("bin/python");
    let output = std::process::Command::new(&python)
        .args(["-I", "-c", "import vllm; print(vllm.__version__, end='')"])
        .env("VIRTUAL_ENV", venv_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let v = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// Verify the installation by importing vllm's compiled CUDA extensions
/// (catches CUDA library mismatches and ABI breaks across install types) and
/// reading the version in the same python spawn — two separate python
/// startups is ~1–2s of wasted cold-start.
///
/// Which extensions exist varies by vllm version (≥ 0.24.0 renamed `vllm._C`
/// to `vllm._C_stable_libtorch`; forks add their own), so instead of
/// hardcoding names, import every top-level compiled extension actually
/// present (vllm/*.abi3.so). Loading all of them also catches duplicate op
/// registration from stale binaries left in an editable-install source tree.
const VERIFY_SCRIPT: &str = r#"
import importlib, pathlib, sys
import vllm
if vllm.__file__ is None:
    sys.stderr.write(
        'import vllm resolved to a namespace package, not the installed vllm --\n'
        'no directory on sys.path holds vllm/__init__.py, only bare vllm/ dirs at:\n'
        '  %s\n'
        'sys.path: %s\n' % (list(getattr(vllm, '__path__', [])), sys.path)
    )
    sys.exit(3)
exts = sorted(f.name.split('.')[0] for f in pathlib.Path(vllm.__file__).parent.glob('*.abi3.so'))
assert exts, 'no compiled vllm extensions (vllm/*.abi3.so) found'
for ext in exts:
    importlib.import_module('vllm.' + ext)
print(vllm.__version__, end='')
"#;

fn verify_installation(
    venv_path: &std::path::Path,
    expected_cuda: &CudaVariant,
    specifier: &VersionSpecifier,
) -> Result<String, VvmError> {
    let python = venv_path.join("bin/python");

    // For CPU variant, or when running on a login node without GPU access
    // (nvidia-smi unavailable), skip the C extension check — libcudart won't
    // load without a GPU driver even if the wheel is correct.
    let skip_c_check = matches!(expected_cuda, CudaVariant::Cpu) || !has_gpu_access();
    let script = if skip_c_check {
        "import vllm; print(vllm.__version__, end='')"
    } else {
        VERIFY_SCRIPT
    };

    // -I (isolated mode): verification must test the venv, not the caller's
    // shell. Clusters routinely export PYTHONPATH/PYTHONHOME via module/conda
    // setups, and `python -c` also puts the cwd on sys.path — any of which can
    // shadow the venv's vllm (a stray bare `vllm/` dir resolves as a namespace
    // package with `__file__ = None`). The venv's own site-packages, including
    // editable-install .pth wiring, is unaffected by -I.
    let output = match std::process::Command::new(&python)
        .args(["-I", "-c", script])
        .env("VIRTUAL_ENV", venv_path)
        .output()
    {
        Ok(o) => o,
        Err(_) => return Ok(String::new()), // Can't run python, skip check
    };

    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let arch = std::env::consts::ARCH;
    let variant = expected_cuda.as_str();
    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let stderr_excerpt = if verbose {
        stderr.to_string()
    } else {
        stderr.chars().take(500).collect::<String>()
    };

    let hint = if stderr.contains("resolved to a namespace package") {
        format!(
            "vllm did not import from the venv — python found only bare vllm/ directories \
             (a namespace package), so the install's own vllm is unreachable:\n{}\n\
             Verification ran with `python -I`, so this is not shell pollution: the venv's \
             vllm package itself is missing or its editable source tree is gone.",
            stderr_excerpt
        )
    } else if stderr.contains("libcudart") || stderr.contains("cannot open shared object") {
        let followup = match specifier {
            VersionSpecifier::Nightly => format!(
                "Nightly wheel may not be available for {}/{}.\n\
                 Check: https://wheels.vllm.ai/nightly/{}/vllm/",
                arch, variant, variant
            ),
            VersionSpecifier::Pr(n) => format!(
                "PR #{} precompiled wheel may not match {}. Try: --wheel-commit nightly",
                n, variant
            ),
            _ => format!("Wheel may not exist for {}/{}.", arch, variant),
        };
        format!(
            "CUDA library mismatch — the installed vllm was not built for {}.\n\
             Requested variant: {}, arch: {}.\n\
             {}\n\n\
             --- vllm C extension import stderr ---\n{}",
            variant, variant, arch, followup, stderr_excerpt
        )
    } else if stderr.contains("undefined symbol") {
        let pr_hint = if let VersionSpecifier::Pr(n) = specifier {
            format!(
                "\nThe precompiled C extensions are incompatible with the installed PyTorch.\n\
                 This typically happens when the PR requires a newer PyTorch than the \
                 available precompiled wheels were built with.\n\
                 Workaround: clone the repo, checkout PR #{}, and install with:\n  \
                 vvm install --path <repo> --compile",
                n
            )
        } else {
            String::new()
        };
        format!(
            "vllm C extension import failed (ABI mismatch):\n{}{}",
            stderr_excerpt, pr_hint
        )
    } else {
        format!("vllm C extension import failed:\n{}", stderr_excerpt)
    };

    Err(VvmError::CommandFailed {
        cmd: "installation verification".to_string(),
        detail: hint,
    })
}

fn set_default_if_first(config: &Config, version_dir: &Path) -> Result<(), VvmError> {
    let default_link = config.aliases_dir().join("default");
    if !default_link.exists() {
        // Use create_new semantics: if another concurrent install races here and
        // creates the symlink first, treat it as success (AlreadyExists is fine).
        match std::os::unix::fs::symlink(version_dir, &default_link) {
            Ok(_) => println!("  {} as default version", "Set".blue()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(VvmError::Io(e)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        commit_index_url, commit_timestamp, compiled_drift_since, compiled_drift_worktree,
        find_reusable_compiled, merge_base_with_main, nightly_index_url, rev_exists,
        torch_version_at_commit, version_newer_than,
    };
    use crate::config::Config;
    use crate::cuda::CudaVariant;
    use crate::metadata::VersionMetadata;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn nightly_wheel_index_uses_the_cuda_variant() {
        for (variant, tag) in [
            (CudaVariant::Cu128, "cu128"),
            (CudaVariant::Cu129, "cu129"),
            (CudaVariant::Cu130, "cu130"),
            (CudaVariant::Cpu, "cpu"),
        ] {
            let expected = format!("https://wheels.vllm.ai/nightly/{tag}");
            assert_eq!(nightly_index_url(&variant), expected);
            assert_eq!(commit_index_url("nightly", &variant), expected);
        }
    }

    #[test]
    fn commit_wheel_index_preserves_the_historical_cu129_layout() {
        let commit = "77154ca6116ae34cbfe314beef1016bde9b63989";
        assert_eq!(
            commit_index_url(commit, &CudaVariant::Cu129),
            format!("https://wheels.vllm.ai/{commit}")
        );
        assert_eq!(
            commit_index_url(commit, &CudaVariant::Cu130),
            format!("https://wheels.vllm.ai/{commit}/cu130")
        );
    }

    /// Self-cleaning temp directory (mirrors tests/concurrent_install.rs).
    struct TempRepo {
        path: PathBuf,
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Run a git command in the repo, asserting success. Pins author/committer
    /// identity via env so commits work on machines without a global git config.
    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .args(["-C", &dir.to_string_lossy()])
            .args(args)
            .env("GIT_AUTHOR_NAME", "vvm-test")
            .env("GIT_AUTHOR_EMAIL", "vvm@test")
            .env("GIT_COMMITTER_NAME", "vvm-test")
            .env("GIT_COMMITTER_EMAIL", "vvm@test")
            .output()
            .expect("git should be runnable in tests");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn write(dir: &Path, rel: &str, contents: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, contents).unwrap();
    }

    fn rev_parse(dir: &Path, rev: &str) -> String {
        let out = Command::new("git")
            .args(["-C", &dir.to_string_lossy(), "rev-parse", rev])
            .output()
            .unwrap();
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Build a repo with a `main` branch (carrying csrc/foo.cu) and a `feature`
    /// branch on top, returning (repo, main_tip_sha).
    fn setup_repo() -> (TempRepo, String) {
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vvm-mb-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let repo = TempRepo { path };
        let d = &repo.path;

        git(d, &["init", "--quiet"]);
        // Normalize the initial branch name to `main` across git versions.
        write(d, "csrc/foo.cu", "// base kernel\n");
        write(d, "README.md", "base\n");
        git(d, &["add", "."]);
        git(d, &["commit", "--quiet", "-m", "base"]);
        git(d, &["branch", "-M", "main"]);
        let main_tip = rev_parse(d, "HEAD");

        // Diverge onto a feature branch.
        git(d, &["checkout", "--quiet", "-b", "feature"]);
        (repo, main_tip)
    }

    #[test]
    fn merge_base_resolves_local_main() {
        let (repo, main_tip) = setup_repo();
        let d = repo.path.to_string_lossy().to_string();
        // upstream/main and origin/main don't exist; falls through to `main`.
        assert_eq!(merge_base_with_main(&d).as_deref(), Some(main_tip.as_str()));
    }

    #[test]
    fn drift_none_without_csrc_change_then_some_after() {
        let (repo, main_tip) = setup_repo();
        let d = repo.path.to_string_lossy().to_string();

        // A non-csrc commit on feature: no compiled drift.
        write(&repo.path, "vllm/util.py", "x = 1\n");
        git(&repo.path, &["add", "."]);
        git(&repo.path, &["commit", "--quiet", "-m", "py change"]);
        assert!(compiled_drift_since(&d, &main_tip).is_none());

        // Now touch a csrc file: drift detected and lists the file.
        write(&repo.path, "csrc/foo.cu", "// edited kernel\n");
        git(&repo.path, &["add", "."]);
        git(&repo.path, &["commit", "--quiet", "-m", "kernel change"]);
        let drift = compiled_drift_since(&d, &main_tip).expect("expected drift");
        assert_eq!(drift, vec!["csrc/foo.cu".to_string()]);
    }

    /// Write a metadata.json for a fake installed version under versions/.
    #[allow(clippy::too_many_arguments)]
    fn fake_version(
        base: &Path,
        name: &str,
        install_type: &str,
        cuda: &str,
        source_path: Option<&str>,
        commit: Option<&str>,
        arch_list: Option<&str>,
        installed_at: &str,
    ) {
        let dir = base.join("versions").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let meta = VersionMetadata {
            install_type: install_type.to_string(),
            name: name.to_string(),
            version: None,
            commit: commit.map(str::to_string),
            pr: None,
            branch: None,
            source_path: source_path.map(str::to_string),
            cuda: cuda.to_string(),
            installed_at: installed_at.to_string(),
            arch_list: arch_list.map(str::to_string),
        };
        meta.save(&dir).unwrap();
    }

    #[test]
    fn find_reusable_compiled_picks_latest_matching_install() {
        let (repo, _) = setup_repo();
        let base = repo.path.join("vvm-base");
        std::fs::create_dir_all(base.join("versions")).unwrap();
        let config = Config {
            base_dir: base.clone(),
        };
        let repo_str = repo.path.to_string_lossy().to_string();
        let cuda = CudaVariant::Cu130;

        // No candidates yet.
        assert!(find_reusable_compiled(&config, &repo_str, &cuda).is_none());

        // Ignored as candidates: precompiled --path install (type "commit"),
        // wrong cuda, missing arch_list, different repo. All dated BEFORE the
        // real candidates so they don't trip the latest-tree-touch guard.
        fake_version(
            &base,
            "precompiled",
            "commit",
            "cu130",
            Some(&repo_str),
            Some("aaa"),
            Some("9.0"),
            "2025-12-05T00:00:00Z",
        );
        fake_version(
            &base,
            "wrong-cuda",
            "local-compiled",
            "cu129",
            Some(&repo_str),
            Some("bbb"),
            Some("9.0"),
            "2025-12-06T00:00:00Z",
        );
        fake_version(
            &base,
            "no-arch",
            "local-compiled",
            "cu130",
            Some(&repo_str),
            Some("ccc"),
            None,
            "2025-12-07T00:00:00Z",
        );
        fake_version(
            &base,
            "other-repo",
            "local-compiled",
            "cu130",
            Some("/nonexistent/elsewhere"),
            Some("ddd"),
            Some("9.0"),
            "2026-01-08T00:00:00Z",
        );
        assert!(find_reusable_compiled(&config, &repo_str, &cuda).is_none());

        // Two matches: the most recent wins. Repo path spelled with a
        // trailing `/.` to exercise canonicalization.
        fake_version(
            &base,
            "older",
            "local-compiled",
            "cu130",
            Some(&repo_str),
            Some("eee"),
            Some("8.0"),
            "2026-01-01T00:00:00Z",
        );
        let repo_dotted = format!("{}/.", repo_str);
        fake_version(
            &base,
            "newer",
            "local-compiled",
            "cu130",
            Some(&repo_dotted),
            Some("fff"),
            Some("9.0"),
            "2026-01-02T00:00:00Z",
        );
        let found = find_reusable_compiled(&config, &repo_str, &cuda).expect("expected a match");
        assert_eq!(found.name, "newer");
        assert_eq!(found.commit, "fff");
        assert_eq!(found.arch_list, "9.0");

        // A LATER install of the same tree (here a precompiled --path install,
        // which extracts wheel binaries over the tree) invalidates reuse.
        fake_version(
            &base,
            "later-precompiled",
            "commit",
            "cu130",
            Some(&repo_str),
            Some("ggg"),
            None,
            "2026-01-09T00:00:00Z",
        );
        assert!(find_reusable_compiled(&config, &repo_str, &cuda).is_none());
    }

    #[test]
    fn remove_tree_kernel_binaries_only_touches_abi3_so() {
        let (repo, _) = setup_repo();
        write(&repo.path, "vllm/_C.abi3.so", "stale");
        write(
            &repo.path,
            "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
            "stale",
        );
        write(&repo.path, "vllm/__init__.py", "");
        write(&repo.path, "vllm/libs/libfoo.so", "keep"); // plain .so untouched
        let repo_str = repo.path.to_string_lossy().to_string();
        assert_eq!(super::remove_tree_kernel_binaries(&repo_str), 2);
        assert!(!repo.path.join("vllm/_C.abi3.so").exists());
        assert!(!repo
            .path
            .join("vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so")
            .exists());
        assert!(repo.path.join("vllm/__init__.py").exists());
        assert!(repo.path.join("vllm/libs/libfoo.so").exists());
    }

    #[test]
    fn worktree_drift_catches_uncommitted_kernel_edits() {
        let (repo, main_tip) = setup_repo();
        let d = repo.path.to_string_lossy().to_string();

        // Python-only commit: neither committed nor worktree drift.
        write(&repo.path, "vllm/util.py", "x = 1\n");
        git(&repo.path, &["add", "."]);
        git(&repo.path, &["commit", "--quiet", "-m", "py change"]);
        assert!(compiled_drift_worktree(&d, &main_tip).is_none());

        // Uncommitted csrc edit: invisible to the committed-range check,
        // caught by the worktree check.
        write(&repo.path, "csrc/foo.cu", "// dirty edit\n");
        assert!(compiled_drift_since(&d, &main_tip).is_none());
        let drift = compiled_drift_worktree(&d, &main_tip).expect("expected worktree drift");
        assert_eq!(drift, vec!["csrc/foo.cu".to_string()]);
    }

    #[test]
    fn torch_version_read_from_historic_commit() {
        let (repo, main_tip) = setup_repo();
        let d = repo.path.to_string_lossy().to_string();

        // No pyproject.toml at main tip.
        assert_eq!(torch_version_at_commit(&d, &main_tip), None);

        // Add a pyproject with a torch pin on the feature branch; the value is
        // read from the commit, not the (later-modified) worktree.
        write(
            &repo.path,
            "pyproject.toml",
            "[build-system]\nrequires = [\n    \"torch == 2.9.0\",\n]\n",
        );
        git(&repo.path, &["add", "."]);
        git(&repo.path, &["commit", "--quiet", "-m", "pin torch"]);
        let pinned = rev_parse(&repo.path, "HEAD");
        write(
            &repo.path,
            "pyproject.toml",
            "[build-system]\nrequires = [\n    \"torch == 2.10.0\",\n]\n",
        );
        assert_eq!(
            torch_version_at_commit(&d, &pinned).as_deref(),
            Some("2.9.0")
        );
    }

    #[test]
    fn version_newer_than_compares_numeric_segments() {
        assert!(version_newer_than("0.23.0", "v0.22.1"));
        assert!(version_newer_than("0.23.0", "0.22.1"));
        assert!(!version_newer_than("0.22.1", "v0.22.1"));
        assert!(!version_newer_than("0.21.0", "v0.22.1"));
        // Missing segments compare as lower; non-numeric suffixes ignored.
        assert!(version_newer_than("0.23", "0.22.9"));
        assert!(version_newer_than("0.23.0rc1", "0.22.1"));
    }

    #[test]
    fn commit_timestamp_and_rev_exists() {
        let (repo, main_tip) = setup_repo();
        let d = repo.path.to_string_lossy().to_string();

        assert!(commit_timestamp(&d, "HEAD")
            .map(|t| !t.is_empty())
            .unwrap_or(false));
        assert!(rev_exists(&d, &main_tip));
        assert!(!rev_exists(&d, "0000000000000000000000000000000000000000"));
    }
}
