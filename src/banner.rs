const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub fn print_banner() {
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

    print!("\x1b[48;5;17m"); // Gotham night background
    println!("{BLUE}{skyline}{RESET}");
    println!("{PURPLE}{batarang}{RESET}");
    println!("{CYAN}{BOLD}{gotham}{RESET}");
    println!("{YELLOW}{BOLD}   ⚡ GOTHAM · SEATTLE NIGHTS — AGENTIC CODING TERMINAL{RESET}");
    println!("{DIM}   local models · auto-discovery · no cloud{RESET}");
    print!("{RESET}");
}
