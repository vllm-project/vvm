use std::io::Write;

use clap::CommandFactory;
use clap_complete::{generate, Shell};

use crate::errors::VvmError;

/// Output shell initialization code (wrapper function + tab completions).
pub fn run(shell: &str) -> Result<(), VvmError> {
    match shell {
        "bash" => {
            print_bash_init();
            print_completions(Shell::Bash);
            // Override clap's static `complete -F _vvm vvm` with our dynamic wrapper
            println!("complete -F __vvm_complete vvm");
        }
        "zsh" => {
            print_zsh_init();
            print_completions(Shell::Zsh);
        }
        "fish" => {
            print_fish_init();
            print_completions(Shell::Fish);
        }
        _ => {
            return Err(VvmError::InvalidSpecifier(format!(
                "unsupported shell: {}. Supported: bash, zsh, fish",
                shell
            )));
        }
    }
    Ok(())
}

fn print_completions(shell: Shell) {
    let mut cmd = crate::Cli::command();
    let mut buf = Vec::new();
    generate(shell, &mut cmd, "vvm", &mut buf);
    // The completions reference the real `vvm` binary. For bash/zsh, clap generates
    // `complete -F ... vvm` which works even though we wrap it as a shell function,
    // because `command vvm` is still the binary that clap introspects.
    std::io::stdout().write_all(&buf).ok();
}

fn print_bash_init() {
    print!(
        r#"# vvm shell integration (bash)

vvm() {{
    local cmd="${{1:-}}"
    command vvm "$@"
    local exit_code=$?
    case "$cmd" in
        use|install)
            if [ $exit_code -eq 0 ]; then
                eval "$(command vvm env)"
            fi
            ;;
    esac
    return $exit_code
}}

# Dynamic completion for subcommands that take installed version names
__vvm_installed_versions() {{
    command vvm list-installed 2>/dev/null
}}

__vvm_complete() {{
    local cur="${{COMP_WORDS[COMP_CWORD]}}"
    local cmd="${{COMP_WORDS[1]}}"
    case "$cmd" in
        use|uninstall|exec)
            COMPREPLY=( $(compgen -W "$(__vvm_installed_versions)" -- "$cur") )
            return
            ;;
    esac
    # Fall back to clap-generated completions
    _vvm "$1" "$2" "$3"
}}

# Auto-switch on cd (optional, reads .vvmrc)
__vvm_cd() {{
    builtin cd "$@" || return
    if [ -f .vvmrc ]; then
        vvm use
    fi
}}

# Uncomment to enable auto-switch:
# alias cd=__vvm_cd

# Load current version if set
eval "$(command vvm env)"

"#
    );
}

fn print_zsh_init() {
    print!(
        r#"# vvm shell integration (zsh)

vvm() {{
    local cmd="${{1:-}}"
    command vvm "$@"
    local exit_code=$?
    case "$cmd" in
        use|install)
            if [ $exit_code -eq 0 ]; then
                eval "$(command vvm env)"
            fi
            ;;
    esac
    return $exit_code
}}

# Dynamic completion for subcommands that take installed version names
__vvm_installed_versions() {{
    command vvm list-installed 2>/dev/null
}}

_vvm_dynamic() {{
    local state
    _arguments '1: :->cmd' '*: :->arg'
    case $state in
        arg)
            case $words[2] in
                use|uninstall|exec)
                    _values 'installed versions' $(__vvm_installed_versions)
                    return
                    ;;
            esac
            ;;
    esac
    _vvm
}}

# Auto-switch on cd (reads .vvmrc)
__vvm_chpwd() {{
    if [ -f .vvmrc ]; then
        vvm use
    fi
}}

# Uncomment to enable auto-switch:
# add-zsh-hook chpwd __vvm_chpwd

# Load current version if set
eval "$(command vvm env)"

"#
    );
}

fn print_fish_init() {
    print!(
        r#"# vvm shell integration (fish)

function vvm
    set -l cmd $argv[1]
    command vvm $argv
    set -l exit_code $status
    switch "$cmd"
        case use install
            if test $exit_code -eq 0
                eval (command vvm env --shell fish)
            end
    end
    return $exit_code
end

# Dynamic completion for subcommands that take installed version names
complete -c vvm -n '__fish_seen_subcommand_from use uninstall exec' -f -a '(command vvm list-installed 2>/dev/null)'

# Load current version if set
eval (command vvm env --shell fish)

"#
    );
}
