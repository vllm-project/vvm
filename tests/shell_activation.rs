use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
    data: PathBuf,
}

impl Fixture {
    fn new(label: &str, current: bool) -> Self {
        let root = std::env::temp_dir().join(format!(
            "vvm shell {label} {}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let data = root.join("data with spaces");
        for version in ["first", "second"] {
            std::fs::create_dir_all(data.join("versions").join(version).join("venv/bin")).unwrap();
        }
        if current {
            std::os::unix::fs::symlink(data.join("versions/first"), data.join("current")).unwrap();
        }
        Self { root, data }
    }

    fn run(&self, shell: &str, script: &str) {
        let binary = Path::new(env!("CARGO_BIN_EXE_vvm"));
        let mut paths = vec![binary.parent().unwrap().to_path_buf()];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        let mut command = Command::new(shell);
        match shell {
            "fish" => command.args(["--no-config", "-c"]),
            "bash" => command.args(["--noprofile", "--norc", "-c"]),
            "zsh" => command.args(["-f", "-c"]),
            _ => unreachable!(),
        };
        let output = command
            .arg(script)
            .current_dir(&self.root)
            .env("VVM_DIR", &self.data)
            .env("PATH", std::env::join_paths(paths).unwrap())
            .env("VVM_TEST_SHELL", shell)
            .env("VVM_FIRST", self.data.join("versions/first/venv"))
            .env("VVM_SECOND", self.data.join("versions/second/venv"))
            .env("VIRTUAL_ENV", "unrelated environment")
            .env_remove("__VVM_PREV_BIN")
            .output()
            .unwrap_or_else(|error| panic!("cannot run {shell}: {error}; install fish and zsh"));
        assert!(
            output.status.success(),
            "{shell} exited {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn fish_activates_and_switches_versions() {
    let fixture = Fixture::new("fish 'quotes' \\ $", true);
    fixture.run(
        "fish",
        r#"
        vvm init fish | source
        test "$VIRTUAL_ENV" = "$VVM_FIRST"; or exit 10
        test "$PATH[1]" = "$VVM_FIRST/bin"; or exit 11
        vvm use second; or exit 12
        test "$VIRTUAL_ENV" = "$VVM_SECOND"; or exit 13
        test "$PATH[1]" = "$VVM_SECOND/bin"; or exit 14
        contains -- "$VVM_FIRST/bin" $PATH; and exit 15
        vvm use second; or exit 16
        set -l occurrences 0
        for entry in $PATH
            if test "$entry" = "$VVM_SECOND/bin"
                set occurrences (math $occurrences + 1)
            end
        end
        test $occurrences -eq 1; or exit 17
        "#,
    );
}

#[test]
fn fish_without_current_keeps_the_environment() {
    let fixture = Fixture::new("empty", false);
    fixture.run(
        "fish",
        r#"
        set -l previous_path "$PATH"
        vvm init fish | source
        test "$VIRTUAL_ENV" = 'unrelated environment'; or exit 20
        test "$PATH" = "$previous_path"; or exit 21
        set -q __VVM_PREV_BIN; and exit 22
        exit 0
        "#,
    );
}

#[test]
fn bash_and_zsh_environment_output_remains_compatible() {
    let fixture = Fixture::new("posix", true);
    for shell in ["bash", "zsh"] {
        fixture.run(
            shell,
            r#"
            eval "$(vvm env)"
            test "$VIRTUAL_ENV" = "$VVM_FIRST" || exit 30
            case "$PATH" in "$VVM_FIRST/bin:"*) ;; *) exit 31;; esac
            vvm use second || exit 32
            eval "$(vvm env --shell "$VVM_TEST_SHELL")"
            test "$VIRTUAL_ENV" = "$VVM_SECOND" || exit 33
            case "$PATH" in "$VVM_SECOND/bin:"*) ;; *) exit 34;; esac
            case ":$PATH:" in *":$VVM_FIRST/bin:"*) exit 35;; esac
            "#,
        );
        std::fs::remove_file(fixture.data.join("current")).unwrap();
        std::os::unix::fs::symlink(
            fixture.data.join("versions/first"),
            fixture.data.join("current"),
        )
        .unwrap();
    }
}
