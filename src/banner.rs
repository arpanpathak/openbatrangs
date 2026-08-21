const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub fn banner_text() -> String {
    let batarang = r#"
            /\          /\
           /  \        /  \
          /    \      /    \
         /      \    /      \
        <        \  /        >
         \        \/        /
          \                /
           \              /
            \            /
             \          /
              \        /
               \      /
                \    /
                 \  /
                  \/
"#;

    let skyline = r#"
       ██  ██  ████   ██     ██████   ██
      ████ ████ █████  ███   ████████  ███
     ████████████████████████████████████████
"#;

    let gotham = r#"        ██████  ██████  ████████ ██   ██  █████  ███    ███
       ██      ██    ██    ██    ██   ██ ██   ██ ████  ████
       ██      ████████    ██    ███████ ███████ ██ ████ ██
       ██      ██    ██    ██    ██   ██ ██   ██ ██  ██  ██
        ██████ ██    ██    ██    ██   ██ ██   ██ ██      ██
"#;

    format!(
        "\x1b[48;5;17m{BLUE}{skyline}{RESET}\n\
         {PURPLE}{batarang}{RESET}\n\
         {CYAN}{BOLD}{gotham}{RESET}\n\
         {YELLOW}{BOLD}   ⚡ GOTHAM · SEATTLE NIGHTS — AGENTIC CODING TERMINAL{RESET}\n\
         {DIM}   local models · auto-discovery · no cloud{RESET}\n\
         {RESET}"
    )
}

pub fn print_banner() {
    print!("{}", banner_text());
}
